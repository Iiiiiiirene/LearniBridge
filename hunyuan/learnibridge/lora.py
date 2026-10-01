import math
from collections.abc import Iterable, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if rank < 0 or not math.isfinite(alpha) or not 0 <= dropout < 1:
            raise ValueError("Require rank >= 0, finite alpha, and 0 <= dropout < 1.")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank else 0.0
        self.enabled = True
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.base.requires_grad_(False)
        if rank:
            self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
            self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        if not self.rank or not self.enabled:
            return result
        update = self.dropout(inputs).to(self.lora_a.dtype) @ self.lora_a.t()
        update = update @ self.lora_b.t()
        return result + (update * self.scaling).to(result.dtype)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def find_linear_paths(module: nn.Module, prefix: str = "") -> list[str]:
    paths = []
    for name, child in module.named_children():
        child_path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            paths.append(child_path)
        elif not isinstance(child, LoRALinear):
            paths.extend(find_linear_paths(child, child_path))
    return paths


def inject_lora(module: nn.Module, targets: Sequence[str], rank=4, alpha=8.0, dropout=0.0):
    if len(set(targets)) != len(targets):
        raise ValueError("LoRA target paths must be unique.")
    replacements = []
    for path in targets:
        parent_path, _, name = path.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        base = getattr(parent, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} is not an nn.Linear (got {type(base).__name__}).")
        replacements.append((parent, name, LoRALinear(base, rank, alpha, dropout)))
    for parent, name, adapter in replacements:
        setattr(parent, name, adapter)
    return [adapter for _, _, adapter in replacements]


def freeze_except_lora(root: nn.Module) -> None:
    root.requires_grad_(False)
    for module in root.modules():
        if isinstance(module, LoRALinear):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)


def iter_lora_parameters(root: nn.Module) -> Iterable[nn.Parameter]:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield from (parameter for parameter in module.parameters(recurse=False) if parameter.requires_grad)


def set_lora_enabled(root: nn.Module, enabled: bool) -> None:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def adapter_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in root.state_dict().items()
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    }


def load_adapter_state(root: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {
        name: value for name, value in root.state_dict().items()
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    }
    if set(state) != set(expected):
        raise ValueError(f"Adapter keys differ: missing={sorted(set(expected) - set(state))}, "
                         f"unexpected={sorted(set(state) - set(expected))}")
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"Adapter shape mismatch for {name}: {value.shape} != {expected[name].shape}")
    root.load_state_dict(state, strict=False)
