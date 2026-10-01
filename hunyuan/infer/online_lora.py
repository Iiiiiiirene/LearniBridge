#推理代码
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
import time
import types
from learnibridge.hunyuan_environment import isolate_float_checkpoint_imports
from learnibridge.cli_compat import parse_native_arguments
from learnibridge.schedule import (
    add_interval_argument, resolve_inference_schedule, validate_cached_source,
    validate_interval_adapters,
)

isolate_float_checkpoint_imports()

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from typing import Any, List, Tuple, Optional, Union, Dict
import torch
import torch.nn.functional as F
import json
import numpy as np

# Import LoRA utilities from train script
from learnibridge.lora import (
    LoRALinear,
    inject_lora,
    find_linear_paths,
    freeze_except_lora,
    load_adapter_state,
    set_lora_enabled,
)


def load_lora_adapter(
    block: torch.nn.Module,
    lora_path: Path,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    device: torch.device = None,
) -> None:
    """
    Load LoRA adapter weights into a block.
    
    Args:
        block: The target block to inject LoRA into
        lora_path: Path to the LoRA checkpoint file
        target_modules: List of module paths to inject LoRA
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        device: Device to load the weights to
    """
    # Inject LoRA if not already injected
    available_paths = find_linear_paths(block)
    if target_modules == ["auto"]:
        target_paths = [p for p in available_paths if p.split(".")[-1] in {"linear1", "linear2"}]
        target_paths = target_paths or available_paths
    else:
        target_paths = target_modules
        missing = [p for p in target_paths if p not in available_paths]
        if missing:
            raise ValueError(f"Requested target modules not found: {missing}")
    
    # Check if LoRA is already injected
    has_lora = any(isinstance(m, LoRALinear) for m in block.modules())
    if not has_lora:
        inject_lora(block, target_paths, rank=rank, alpha=alpha, dropout=dropout)
    
    # Load state dict
    map_location = device if device is not None else "cpu"
    payload = torch.load(lora_path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    
    # Move state dict to correct device
    if device is not None:
        state_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in state_dict.items()}
    
    # Load weights
    load_adapter_state(block, {
        name: value for name, value in state_dict.items()
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    })
    set_lora_enabled(block, True)
    logger.info(f"Loaded LoRA adapter from {lora_path}")


def lora_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    text_states: torch.Tensor = None,
    text_mask: torch.Tensor = None,
    text_states_2: Optional[torch.Tensor] = None,
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
    guidance: torch.Tensor = None,
    return_dict: bool = True,
    replace_steps: set = None,
    lora_dir: str = "adapters",
    feature_root: str = "feature/0",
    block_idx: int = 39,
    target_modules: List[str] = None,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Forward pass with LoRA adapter support.
    
    Args:
        replace_steps: Set of timesteps to use LoRA replacement
        lora_dir: Directory containing LoRA checkpoint files (block39_step{i}.pt)
        feature_root: Root directory for feature comparison files
        block_idx: Index of the block to apply LoRA (default: 39)
        target_modules: List of module paths for LoRA injection
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    if replace_steps is None:
        replace_steps = set()
    if 0 in replace_steps:
        raise ValueError("Step zero must compute the initial cache.")
    if target_modules is None:
        target_modules = ["auto"]
    
    out = {}
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // self.patch_size[0],
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )

    # Prepare modulation vectors
    vec = self.time_in(t)
    
    # Text modulation
    vec = vec + self.vector_in(text_states_2)

    # Guidance modulation
    if self.guidance_embed:
        if guidance is None:
            raise ValueError(
                "Didn't get guidance strength for guidance distilled model."
            )
        vec = vec + self.guidance_in(guidance)

    # Embed image and text
    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(
            f"Unsupported text_projection: {self.text_projection}"
        )

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    # Compute cu_seqlens and max_seqlen for flash attention
    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
    
    # Get current timestep
    current_step = self.cnt
    if current_step == 0:
        self._learnibridge_source_step = None
        self.pre = None
        self.previous_residual = None
    set_lora_enabled(self.single_blocks, False)
    
    if current_step in replace_steps:
        schedule = getattr(self, "_learnibridge_schedule", None)
        validate_cached_source(schedule, current_step, getattr(self, "_learnibridge_source_step", None))
        # Try to load and use LoRA adapter
        lora_path = None
        if lora_dir:
            lora_path = Path(lora_dir) / f"block{block_idx}_step{current_step}.pt"
            if not lora_path.exists():
                lora_path = None
        
        if lora_path is not None and lora_path.exists():
            # Load LoRA and run only block39
            logger.info(f"Using LoRA adapter for step {current_step}: {lora_path}")
            
            # Get block39
            if block_idx >= len(self.single_blocks):
                raise ValueError(f"block_idx {block_idx} is out of range")
            
            block39 = self.single_blocks[block_idx]
            
            # Load LoRA adapter
            device = next(block39.parameters()).device
            load_lora_adapter(
                block39,
                lora_path,
                target_modules=target_modules,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                device=device,
            )
            
            # Use previous block39 output as input
            if not hasattr(self, 'pre') or self.pre is None:
                logger.warning(f"self.pre is not set at step {current_step}. Cannot use LoRA, falling back to residual.")
                if hasattr(self, 'previous_residual') and self.previous_residual is not None:
                    img += self.previous_residual
                else:
                    logger.error(f"No residual available at step {current_step}. This step should not be in replace_steps.")
                # Skip LoRA processing and continue to final layer
            else:
                x_input = self.pre  # Input from previous normal run
                
                # Run only block39 with LoRA
                x = block39(
                    x_input,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                )
                
                # Compare with ground truth and calculate MSE
                # if feature_root:
                #     gt_path = Path(feature_root) / f"single_block{block_idx}_step_{current_step}.pt"
                #     if gt_path.exists():
                #         gt_data = torch.load(gt_path, map_location=x.device)
                #         gt_data = gt_data.to(x.dtype)
                #         mse_loss = F.mse_loss(x, gt_data)
                #         logger.info(f"Step {current_step} MSE: {mse_loss.item():.6f}")
                #     else:
                #         logger.warning(f"Ground truth file not found: {gt_path}")
                
                # Extract image part
                img = x[:, :img_seq_len, ...]
        else:
            if schedule is not None:
                raise FileNotFoundError(f"Missing scheduled Hunyuan adapter for step {current_step}.")
            # No LoRA file found, use residual
            if hasattr(self, 'previous_residual') and self.previous_residual is not None:
                img += self.previous_residual
                logger.info(f"Step {current_step} in replace_steps but no LoRA found, using residual")
            else:
                logger.warning(f"Step {current_step} in replace_steps but no LoRA and no residual available")
    else:
        # Normal forward pass
        ori_img = img.clone()
        
        # Pass through DiT blocks
        for _, block in enumerate(self.double_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]
            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for block_idx_inner, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                ]
                x = block(*single_block_args)
                # Save block38 output (for reference)
            if block_idx_inner == 38:
                self.pre = x.clone()
                self._learnibridge_source_step = current_step

        img = x[:, :img_seq_len, ...]
        self.previous_residual = img - ori_img
    
    # Final layer
    img = self.final_layer(img, vec)
    img = self.unpatchify(img, tt, th, tw)
    
    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0
    
    if return_dict:
        out["x"] = img
        return out
    return img


def main():
    # Parse LoRA-specific arguments first
    parser = argparse.ArgumentParser(description="Online inference with LoRA adapters")
    parser.add_argument("--lora-dir", type=str, default=None, help="Directory containing LoRA checkpoint files")
    parser.add_argument("--feature-root", type=str, default="feature/0", help="Root directory for feature comparison")
    parser.add_argument("--replace-steps", type=int, nargs="+", help="Legacy explicit replacement steps; mutually exclusive with --N.")
    add_interval_argument(parser)
    parser.add_argument("--block-idx", type=int, default=39, help="Index of the block to apply LoRA")
    parser.add_argument("--target-modules", type=str, nargs="+", default=["auto"], help="Target modules for LoRA injection")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=64.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    
    # Parse args and merge with hyvideo config
    args, unknown = parser.parse_known_args()
    hyvideo_args = parse_native_arguments(parse_args, unknown)
    
    # Merge arguments
    for key, value in vars(args).items():
        if not hasattr(hyvideo_args, key):
            setattr(hyvideo_args, key, value)
    
    # Use hyvideo_args as the main args object
    args = hyvideo_args
    try:
        schedule = resolve_inference_schedule(args.skip_interval, args.infer_steps, args.replace_steps)
        if schedule is not None:
            if args.block_idx != 39:
                raise ValueError("--N mode calibrates the original final block 39.")
            validate_interval_adapters(
                args.lora_dir, schedule, "hunyuan", rank=args.lora_rank, alpha=args.lora_alpha,
            )
            args.replace_steps = schedule.replace_steps
        elif args.replace_steps is None:
            args.replace_steps = [2,3,4,5,7,8,9,10,11,12,14,15,16,17,18,19,21,22,23,24,25,26,28,29,30,31,32,33,35,36,37,38,39,40,42,43,44,45,46,48]
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    
    print(f"Arguments: {args}")
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`model_base` not exists: {models_root_path}")
    
    # Create save folder
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args
    
    # Setup LoRA forward
    replace_steps_set = set(args.replace_steps) if args.replace_steps else set()
    
    # Create a partial function with the arguments
    def lora_forward_wrapper(self, x, t, text_states=None, text_mask=None, text_states_2=None,
                             freqs_cos=None, freqs_sin=None, guidance=None, return_dict=True):
        return lora_forward(
            self, x, t, text_states, text_mask, text_states_2,
            freqs_cos, freqs_sin, guidance, return_dict,
            replace_steps=replace_steps_set,
            lora_dir=args.lora_dir,
            feature_root=args.feature_root,
            block_idx=args.block_idx,
            target_modules=args.target_modules,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    
    # Initialize transformer state
    transformer = hunyuan_video_sampler.pipeline.transformer
    transformer.cnt = 0
    transformer.num_steps = args.infer_steps
    transformer._learnibridge_schedule = schedule
    if schedule is not None:
        logger.info(f"N={schedule.interval}; full steps={schedule.full_steps}; calibrated steps={schedule.replace_steps}")
    transformer.previous_residual = None
    transformer.pre = None
    transformer.pre_block38 = None
    if hasattr(transformer, "_hf_hook") and hasattr(transformer, "_old_forward"):
        transformer._old_forward = types.MethodType(lora_forward_wrapper, transformer)
    else:
        transformer.forward = types.MethodType(lora_forward_wrapper, transformer)
    
    # Start sampling
    logger.info(f"Starting inference with LoRA adapters. Replace steps: {replace_steps_set}")
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path_file = f"{save_path}/seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}_on_{time_flag}.mp4"
            save_videos_grid(sample, save_path_file, fps=24)
            logger.info(f'Sample saved to: {save_path_file}')


if __name__ == "__main__":
    main()
