import copy
import inspect
from pathlib import Path

import torch

from .flux_compat import forward_single_block
from .lora import inject_lora, load_adapter_state


def bound_arguments(forward, arguments, keywords):
    bound = inspect.signature(forward).bind(*arguments, **keywords)
    bound.apply_defaults()
    return bound.arguments


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {name: to_cpu(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(item) for item in value)
    return value


def combined_hidden(arguments):
    hidden = arguments["hidden_states"]
    encoder = arguments.get("encoder_hidden_states")
    return torch.cat((encoder, hidden), dim=1) if encoder is not None else hidden


class FeatureRecorder:
    """Record only the final-block input/output and conditioning, without replacing forward."""

    def __init__(self, transformer, output_dir, block_idx=None):
        self.transformer = transformer
        self.output_dir = Path(output_dir)
        self.block_idx = len(transformer.single_transformer_blocks) - 1 if block_idx is None else block_idx
        if self.block_idx != len(transformer.single_transformer_blocks) - 1 or self.block_idx < 1:
            raise ValueError("Feature extraction requires the final single block and at least two single blocks.")
        self.step = 0
        self.text_length = None
        self.handles = [
            transformer.register_forward_pre_hook(self._before_transformer, with_kwargs=True),
            transformer.single_transformer_blocks[self.block_idx].register_forward_pre_hook(
                self._before_block, with_kwargs=True
            ),
            transformer.single_transformer_blocks[self.block_idx].register_forward_hook(
                self._after_block, with_kwargs=True
            ),
        ]

    def _save(self, name, value):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{name}_step_{self.step}.pt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite features: {destination}")
        torch.save(to_cpu(value), destination)

    def _before_transformer(self, module, arguments, keywords):
        values = bound_arguments(module.forward, arguments, keywords)
        self.text_length = values["encoder_hidden_states"].shape[1]

    def _before_block(self, module, arguments, keywords):
        values = bound_arguments(module.forward, arguments, keywords)
        self._save(f"singleblock_{self.block_idx - 1}_output", combined_hidden(values))
        self._save("temb", values["temb"])
        self._save("image_rotary_emb", values.get("image_rotary_emb"))
        self._save("text_length", self.text_length)
        if values.get("joint_attention_kwargs"):
            self._save("joint_attention_kwargs", values["joint_attention_kwargs"])

    def _after_block(self, module, arguments, keywords, output):
        combined = torch.cat(output, dim=1) if isinstance(output, tuple) else output
        self._save(f"singleblock_{self.block_idx}_output", combined)
        self.step += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


class FluxBridge:
    """Run the original transformer on full steps and a separate calibrated block on skipped steps."""

    def __init__(self, transformer, checkpoints, num_steps):
        if num_steps < 1 or any(step <= 0 or step >= num_steps for step in checkpoints):
            raise ValueError("Replacement steps must be in [1, num_steps); step zero is always full compute.")
        self.transformer = transformer
        self.original_forward = transformer.forward
        self.num_steps = num_steps
        self.checkpoints = checkpoints
        self.block_idx = len(transformer.single_transformer_blocks) - 1
        self.adapter_block = None
        configuration = None
        for step, checkpoint in sorted(checkpoints.items()):
            if checkpoint.get("format_version") != 1:
                raise ValueError("Use adapters produced by the canonical trainer (format_version=1).")
            if checkpoint["block_idx"] != self.block_idx or step not in checkpoint["steps"]:
                raise ValueError(f"Checkpoint metadata does not match block/step {step}.")
            current = (checkpoint["rank"], checkpoint["alpha"], checkpoint["dropout"],
                       tuple(checkpoint["target_modules"]))
            if configuration is not None and current != configuration:
                raise ValueError("All adapters in a run must have the same rank, alpha, dropout, and target modules.")
            configuration = current
        if configuration is not None:
            rank, alpha, dropout, targets = configuration
            self.adapter_block = copy.deepcopy(transformer.single_transformer_blocks[self.block_idx])
            inject_lora(self.adapter_block, targets, rank=rank, alpha=alpha, dropout=dropout)
            self.adapter_block.requires_grad_(False).eval()
            for checkpoint in checkpoints.values():
                load_adapter_state(self.adapter_block, checkpoint["state_dict"])
        self.reset()
        self.handle = transformer.single_transformer_blocks[self.block_idx].register_forward_pre_hook(
            self._cache_input, with_kwargs=True
        )
        transformer.forward = self.forward

    def reset(self):
        self.step = 0
        self.cached_input = None
        self.cached_step = None
        self.text_length = None
        self.full_steps = []
        self.calibrated_steps = []

    def _cache_input(self, module, arguments, keywords):
        values = bound_arguments(module.forward, arguments, keywords)
        self.cached_input = combined_hidden(values).detach().clone()
        self.cached_step = self.step

    def forward(self, *arguments, **keywords):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        if self.step >= self.num_steps:
            raise RuntimeError("Call bridge.reset() before a new diffusion trajectory.")
        values = bound_arguments(self.original_forward, arguments, keywords)
        self.text_length = values["encoder_hidden_states"].shape[1]
        if self.step not in self.checkpoints:
            output = self.original_forward(*arguments, **keywords)
            self.full_steps.append(self.step)
        else:
            checkpoint = self.checkpoints[self.step]
            if self.cached_input is None or self.step - self.cached_step != checkpoint["offset"]:
                raise ValueError(f"Step {self.step}: adapter offset does not match the most recent full step.")
            if values.get("controlnet_block_samples") is not None or values.get("controlnet_single_block_samples") is not None:
                raise NotImplementedError("ControlNet is not supported by the calibrated path.")
            joint_kwargs = values.get("joint_attention_kwargs")
            if joint_kwargs:
                raise NotImplementedError("Additional attention conditioning is not supported by the calibrated path.")
            transformer = self.transformer
            dtype = self.cached_input.dtype
            timestep = values["timestep"].to(dtype) * 1000
            guidance = values.get("guidance")
            temb = (
                transformer.time_text_embed(timestep, values["pooled_projections"])
                if guidance is None else transformer.time_text_embed(
                    timestep, guidance.to(dtype) * 1000, values["pooled_projections"]
                )
            )
            text_ids, image_ids = values["txt_ids"], values["img_ids"]
            if text_ids.ndim == 3:
                text_ids = text_ids[0]
            if image_ids.ndim == 3:
                image_ids = image_ids[0]
            rotary = transformer.pos_embed(torch.cat((text_ids, image_ids), dim=0))
            load_adapter_state(self.adapter_block, checkpoint["state_dict"])
            hidden = forward_single_block(
                self.adapter_block, self.cached_input, temb, rotary, text_length=self.text_length
            )
            hidden = hidden[:, self.text_length:]
            prediction = transformer.proj_out(transformer.norm_out(hidden, temb))
            output = Transformer2DModelOutput(sample=prediction) if values["return_dict"] else (prediction,)
            self.calibrated_steps.append(self.step)
        self.step += 1
        return output

    def close(self):
        self.transformer.forward = self.original_forward
        self.handle.remove()
