import os
import time
import argparse
import types
from pathlib import Path
from loguru import logger
from datetime import datetime
from learnibridge.cli_compat import parse_native_arguments
from learnibridge.hunyuan_environment import isolate_float_checkpoint_imports

isolate_float_checkpoint_imports()

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from typing import Any, List, Tuple, Optional, Union, Dict
import torch
import json
import numpy as np

def save_feature(feature: torch.Tensor, save_path: str, feature_name: str, time_step: int) -> None:
    """
    Save feature tensor to specified path.
    
    Args:
        feature: Feature tensor to save.
        save_path: Directory path to save the feature.
        feature_name: Name of the feature (used in filename).
        time_step: Current timestep (used in filename).
    """
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{feature_name}_step_{time_step}.pt"
    torch.save(feature, os.path.join(save_path, file_name))
    
def teacache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Timestep tensor (for filename labeling)
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
        save_path: str = "feature/5",  # Path to save features
        time_step: int = 0,  # Current inference step (for filenames)
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:                                  
    # Create save directory if it doesn't exist
    import os
    os.makedirs(save_path, exist_ok=True)
    # Extract timestep scalar from tensor
    time_step = self.cnt.item() if isinstance(self.cnt, torch.Tensor) else self.cnt

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
            raise ValueError("Guidance strength missing for distilled model.")
        vec = vec + self.guidance_in(guidance)

    # Embed image and text
    img = self.img_in(img)

    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    # Compute attention sequence lengths
    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    # Pass through DiT blocks (non-Teacache)
    for block_idx, block in enumerate(self.double_blocks):
        img, txt = block(
            img, txt, vec, cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv, freqs_cis
        )

    # Merge and pass through single blocks (non-Teacache)
    x = torch.cat((img, txt), 1)
    # ===== Save inference metadata for TeaCache offline inference =====
    meta = {
        "vec": vec.detach().cpu(),
        "txt_seq_len": txt_seq_len,
        "cu_seqlens_q": cu_seqlens_q.detach().cpu(),
        "cu_seqlens_kv": cu_seqlens_kv.detach().cpu(),
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "freqs_cos": freqs_cos.detach().cpu() if freqs_cos is not None else None,
        "freqs_sin": freqs_sin.detach().cpu() if freqs_sin is not None else None,
    }
    torch.save(meta, os.path.join(save_path, f"meta_step_{time_step}.pt"))
    
    if len(self.single_blocks) > 0:
        for block_idx, block in enumerate(self.single_blocks):
            x = block(
                x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin)
            )
            # Save 11: single block outputs
            selected_blocks = getattr(self, "_learnibridge_cache_blocks", None)
            if (selected_blocks is None and block_idx >= 36) or (
                selected_blocks is not None and block_idx in selected_blocks
            ):
                save_feature(x.detach().cpu(), save_path, f"single_block{block_idx}", time_step)

    img = x[:, :img_seq_len, ...]

    # Final layer processing
    img = self.final_layer(img, vec)
    # # Save 12: Final layer output
    # save_feature(img.detach().cpu(), save_path, "final_layer", time_step)

    img = self.unpatchify(img, tt, th, tw)
    
    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0  

    if return_dict:
        out["x"] = img
        return out
    return img

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feature-dir", type=Path, default=Path("features/hunyuan"))
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--cache-blocks", type=int, nargs="+")
    cache_args, remaining = parser.parse_known_args()
    args = parse_native_arguments(parse_args, remaining)
    prompts = (
        [line.strip() for line in cache_args.prompt_file.read_text().splitlines() if line.strip()]
        if cache_args.prompt_file else [args.prompt]
    )
    if cache_args.max_prompts is not None:
        if cache_args.max_prompts < 1:
            parser.error("--max-prompts must be positive.")
        prompts = prompts[:cache_args.max_prompts]
    if not prompts or not all(prompts):
        parser.error("Supply a nonempty --prompt or --prompt-file.")
    feature_root = cache_args.feature_dir.resolve()
    if feature_root.exists() and any(feature_root.iterdir()):
        raise FileExistsError(f"Use an empty feature directory: {feature_root}")
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    transformer = hunyuan_video_sampler.pipeline.transformer
    if cache_args.cache_blocks is not None and any(
        block < 0 or block >= len(transformer.single_blocks) for block in cache_args.cache_blocks
    ):
        parser.error("--cache-blocks contains an invalid block index.")
    transformer._learnibridge_cache_blocks = (
        set(cache_args.cache_blocks) if cache_args.cache_blocks is not None else None
    )
    transformer.num_steps = args.infer_steps
    def cache_forward(current, *arguments, **keywords):
        return teacache_forward(
            current, *arguments, save_path=current._learnibridge_feature_dir, **keywords
        )
    if hasattr(transformer, "_hf_hook") and hasattr(transformer, "_old_forward"):
        transformer._old_forward = types.MethodType(cache_forward, transformer)
    else:
        transformer.forward = types.MethodType(cache_forward, transformer)
    
    for prompt_index, prompt in enumerate(prompts):
        directory = feature_root / str(prompt_index)
        directory.mkdir(parents=True, exist_ok=False)
        transformer.cnt = 0
        transformer._learnibridge_feature_dir = str(directory)
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
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
            embedded_guidance_scale=args.embedded_cfg_scale,
        )
        (directory / "metadata.json").write_text(json.dumps({
            "prompt": prompt, "seed": args.seed, "steps": args.infer_steps,
            "cache_blocks": cache_args.cache_blocks,
        }, indent=2) + "\n")
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for video_index, sample in enumerate(outputs["samples"]):
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
                video_path = str(Path(save_path) / f"{time_flag}_{prompt_index}_{video_index}.mp4")
                save_videos_grid(sample.unsqueeze(0), video_path, fps=24)
                logger.info(f"Sample saved to: {video_path}")

if __name__ == "__main__":
    main()
