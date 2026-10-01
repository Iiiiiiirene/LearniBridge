"""
Flux LoRA Online Generation Script
- On replace_steps: skip full forward, load LoRA adapter for block 37 only
- On other steps: run full forward and cache residual
"""
import argparse
import math
import os
import logging
import types
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ========================= Config =========================
MODEL_PATH = os.environ.get("FLUX_MODEL_PATH", "")
ADAPTER_DIR = "adapters/ablation_w2"
PROMPT_FILE = str(Path(__file__).resolve().parents[1] / "prompts/test.txt")
OUTPUT_DIR = "ablation/w2"
NUM_INFERENCE_STEPS = 50
SEED = 42
REPLACE_STEPS = {2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20,21,
                 23,24,25,26,27,28,30,31,32,33,34,35,37,38,39,40,41,42,
                 44,45,46,47,48,49}


# ========================= LoRA =========================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        if rank > 0:
            device = base.weight.device
            self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=device))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=device))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.rank == 0:
            return result
        shape = x.shape
        x_2d = self.dropout(x.reshape(-1, shape[-1]))
        lora = x_2d @ self.lora_a.t()
        lora = lora @ self.lora_b.t()
        lora = lora.view(*shape[:-1], self.base.out_features)
        return result + lora * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _locate_attr(module: nn.Module, path: str):
    parts = path.split(".")
    parent = module
    for name in parts[:-1]:
        parent = getattr(parent, name)
    return parent, parts[-1]


def inject_lora(module: nn.Module, targets: Sequence[str], rank: int = 256, alpha: float = 512.0, dropout: float = 0.0):
    loras = []
    for path in targets:
        parent, name = _locate_attr(module, path)
        base = getattr(parent, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} is not an nn.Linear (got {type(base).__name__})")
        lora = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, name, lora)
        loras.append(lora)
    return loras


def find_linear_paths(module: nn.Module, prefix: str = "") -> list:
    paths = []
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            paths.append(child_prefix)
        else:
            paths.extend(find_linear_paths(child, child_prefix))
    return paths


from learnibridge.flux_compat import forward_single_block
from learnibridge.lora import LoRALinear, inject_lora, load_adapter_state, set_lora_enabled
from learnibridge.schedule import (
    add_interval_argument, resolve_inference_schedule, validate_cached_source,
    validate_checkpoint_schedule, validate_interval_adapters,
)


# ========================= Custom Forward =========================
def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    save_path=ADAPTER_DIR,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    if self.cnt == 0:
        self._learnibridge_source_step = None
        self.pre = None
        self.previous_residual = None
    set_lora_enabled(self.single_transformer_blocks, False)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    if self.cnt in self.replace_steps:
        schedule = getattr(self, "_learnibridge_schedule", None)
        validate_cached_source(schedule, self.cnt, getattr(self, "_learnibridge_source_step", None))
        lora_path = Path(save_path) / f"blocks37_step{self.cnt}.pt"
        if not lora_path.exists():
            if schedule is not None:
                raise FileNotFoundError(f"Missing scheduled adapter: {lora_path}")
            if getattr(self, "previous_residual", None) is not None:
                hidden_states = hidden_states + self.previous_residual
            else:
                logger.warning("previous_residual is None; skip residual add.")
        else:
            logger.info(f"Loading LoRA from {lora_path}")
            checkpoint = torch.load(lora_path, map_location=self.device)
            if schedule is not None:
                validate_checkpoint_schedule(checkpoint, schedule, "flux", self.cnt)
            block_idx = checkpoint.get("block_idx", 37)
            target_block = self.single_transformer_blocks[block_idx]

            has_lora = any(isinstance(m, LoRALinear) for m in target_block.modules())
            if not has_lora:
                available_paths = find_linear_paths(target_block)
                target_paths = checkpoint.get("target_modules") or [p for p in available_paths if p.split(".")[-1] in {"linear1", "linear2", "lin"}]
                if not target_paths:
                    target_paths = available_paths
                inject_lora(
                    target_block, target_paths, rank=checkpoint.get("rank", 32),
                    alpha=checkpoint.get("alpha", 64.0), dropout=0.0,
                )

            load_adapter_state(target_block, {
                name: value for name, value in checkpoint["state_dict"].items()
                if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
            })
            set_lora_enabled(target_block, True)
            target_block.eval()
            target_block = target_block.to(torch.float16)

            # Use cached hidden_states from block 36 (self.pre)
            if not hasattr(self, "pre") or self.pre is None:
                logger.warning("self.pre not yet cached; falling back to residual addition.")
                if getattr(self, "previous_residual", None) is not None:
                    hidden_states = hidden_states + self.previous_residual
            else:
                with torch.no_grad():
                    hidden_states = forward_single_block(
                        target_block, self.pre, temb, image_rotary_emb,
                        joint_attention_kwargs, text_length=encoder_hidden_states.shape[1],
                    )
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    else:
        # Full forward pass
        ori_hidden_states = hidden_states.clone()
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = forward_single_block(
                block, hidden_states, temb, image_rotary_emb,
                joint_attention_kwargs, text_length=encoder_hidden_states.shape[1],
            )
            if index_block == 36:
                self.pre = hidden_states.clone()
                self._learnibridge_source_step = self.cnt

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        self.previous_residual = hidden_states - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ========================= Main =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Original cached-residual/LoRA inference.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", default=PROMPT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--replace-steps", type=int, nargs="+")
    add_interval_argument(parser)
    args = parser.parse_args()
    try:
        schedule = resolve_inference_schedule(args.skip_interval, args.num_steps, args.replace_steps)
        if schedule is not None:
            validate_interval_adapters(args.adapter_dir, schedule, "flux")
            args.replace_steps = schedule.replace_steps
        elif args.replace_steps is None:
            args.replace_steps = sorted(REPLACE_STEPS)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if not args.model_path or not Path(args.model_path).is_dir():
        parser.error("Supply an existing local --model-path.")
    if schedule is None and (not args.adapter_dir or not any(
        (Path(args.adapter_dir) / f"blocks37_step{step}.pt").is_file() for step in args.replace_steps
    )):
        parser.error("No trained adapter in --adapter-dir is used by the requested replacement steps.")
    pipeline = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, local_files_only=True)
    pipeline.to("cuda")
    def configured_forward(current, *arguments, **keywords):
        return teacache_forward(current, *arguments, save_path=args.adapter_dir, **keywords)
    pipeline.transformer.forward = types.MethodType(configured_forward, pipeline.transformer)
    pipeline.transformer.cnt = 0
    pipeline.transformer.num_steps = args.num_steps
    pipeline.transformer.replace_steps = set(args.replace_steps)
    pipeline.transformer._learnibridge_schedule = schedule
    if schedule is not None:
        logger.info(f"N={schedule.interval}; full steps={schedule.full_steps}; calibrated steps={schedule.replace_steps}")
    pipeline.transformer.previous_residual = None
    pipeline.transformer.pre = None
    os.makedirs(args.output_dir, exist_ok=True)
    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file, "r", encoding="utf-8") as stream:
            prompts = [line.strip() for line in stream if line.strip()]

    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
        result = pipeline(
            prompt,
            num_inference_steps=args.num_steps,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
        img = result.images[0]

        safe_prompt = "".join(c if c.isalnum() or c in "_-" else "_" for c in prompt[:50])
        filename = f"{i}_online_{safe_prompt}.png"
        save_path = os.path.join(args.output_dir, filename)
        img.save(save_path)
        print(f"Saved: {save_path}")
