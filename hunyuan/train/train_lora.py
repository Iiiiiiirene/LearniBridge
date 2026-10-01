import argparse
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
from learnibridge.lora import adapter_state_dict
from learnibridge.schedule import (
    add_interval_argument, resolve_training_interval, training_schedule_metadata, validate_training_features,
)

from hyvideo.modules import load_model, HUNYUAN_VIDEO_CONFIG
from hyvideo.config import parse_args as parse_hunyuan_args
from hyvideo.constants import PRECISION_TO_TYPE


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps an nn.Linear layer with a learnable low-rank update."""

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
            self.lora_a = nn.Parameter(base.weight.new_zeros(rank, base.in_features))
            self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
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
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def _locate_attr(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = module
    for name in parts[:-1]:
        parent = getattr(parent, name)
    return parent, parts[-1]


def inject_lora(
    module: nn.Module,
    targets: Sequence[str],
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> list[LoRALinear]:
    loras: list[LoRALinear] = []
    for path in targets:
        parent, name = _locate_attr(module, path)
        base = getattr(parent, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} is not an nn.Linear (got {type(base).__name__})")
        lora = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, name, lora)
        loras.append(lora)
    return loras


def freeze_except_lora(root: nn.Module) -> None:
    for param in root.parameters():
        param.requires_grad = False
    for module in root.modules():
        if isinstance(module, LoRALinear):
            for param in module.parameters(recurse=False):
                if param is not None:
                    param.requires_grad = True


def iter_lora_parameters(root: nn.Module) -> Iterable[nn.Parameter]:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            for param in module.parameters(recurse=False):
                if param is not None and param.requires_grad:
                    yield param


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HunyuanMultiRootDataset(Dataset):
    def __init__(
        self,
        roots: Sequence[Path],
        block_idx: int,
        prev_block_idx: int,
        steps: Sequence[int],
        offset: int,
        map_location: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        preload: bool = False,
    ) -> None:
        self.roots = [Path(r) for r in roots]
        self.block_idx = block_idx
        self.prev_block_idx = prev_block_idx
        self.steps = list(steps)
        self.offset = offset
        self.map_location = map_location
        self.dtype = dtype
        self.preload = preload

        self.cache: dict[int, list[dict[str, Any]]] = {}

        print(f"[Init] Found {len(self.roots)} data roots.")
        if preload:
            self._preload_all()

    # ------------------------------------------------------------
    # preload
    # ------------------------------------------------------------
    def _preload_all(self):
        print(f"[Preload] Preparing {len(self.steps)} steps...", flush=True)
        for i, step in enumerate(self.steps):
            batch_samples = []
            for root in self.roots:
                s = self._load_sample(root, step)
                if s is not None:
                    s["step"] = step
                    batch_samples.append(s)

            if batch_samples:
                self.cache[step] = batch_samples
            else:
                print(f"[Warning] Step {step} not found in any root.", flush=True)

            if (i + 1) % 50 == 0 or i == len(self.steps) - 1:
                print(f" Loaded {i + 1}/{len(self.steps)} steps", flush=True)

        print(f"[Preload Complete] Cached {len(self.cache)} steps.", flush=True)

    # ------------------------------------------------------------
    # loading helpers
    # ------------------------------------------------------------
    def _load_meta(self, path: Path) -> dict[str, Any] | None:
        if path.exists():
            data = torch.load(path, map_location=self.map_location)
            if isinstance(data, dict):
                return data
        return None

    def _load_sample(self, root: Path, step: int) -> dict[str, Any] | None:
        src_step = step - self.offset
        try:
            x_path = root / f"single_block{self.prev_block_idx}_step_{src_step}.pt"
            y_path = root / f"single_block{self.block_idx}_step_{step}.pt"
            meta_path = root / f"meta_step_{step}.pt"

            if not (x_path.exists() and y_path.exists() and meta_path.exists()):
                return None

            x_input = torch.load(x_path, map_location=self.map_location).to(self.dtype)
            target = torch.load(y_path, map_location=self.map_location).to(self.dtype)
            meta = self._load_meta(meta_path)
            if meta is None:
                return None

            return {
                "x_input": x_input,
                "target": target,
                "vec": meta["vec"].to(self.dtype),
                "txt_seq_len": meta["txt_seq_len"],
                "cu_seqlens_q": meta["cu_seqlens_q"],
                "cu_seqlens_kv": meta["cu_seqlens_kv"],
                "max_seqlen_q": meta["max_seqlen_q"],
                "max_seqlen_kv": meta["max_seqlen_kv"],
                "freqs_cos": meta["freqs_cos"].to(self.dtype) if meta["freqs_cos"] is not None else None,
                "freqs_sin": meta["freqs_sin"].to(self.dtype) if meta["freqs_sin"] is not None else None,
            }
        except Exception as e:
            print(f"[Error] {root} step {step}: {e}")
            return None

    # ------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------
    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        step = self.steps[idx]

        if self.preload and step in self.cache:
            return self.cache[step]

        batch_samples = []
        for root in self.roots:
            s = self._load_sample(root, step)
            if s is not None:
                s["step"] = step
                batch_samples.append(s)

        if not batch_samples:
            raise RuntimeError(f"No valid samples found for step {step}")

        return batch_samples

    def __len__(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def find_linear_paths(module: nn.Module, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            paths.append(child_prefix)
        else:
            paths.extend(find_linear_paths(child, child_prefix))
    return paths


def move_to(obj: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(obj, torch.Tensor):
        if torch.is_floating_point(obj):
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {k: move_to(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [move_to(v, device, dtype) for v in obj]
        return type(obj)(items)
    return obj


def autocast_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision {name}")


# ---------------------------------------------------------------------------
# Train (Multi-root batch version)
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    resolve_training_interval(args)
    validate_training_features(args, "hunyuan")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    amp_dtype = autocast_dtype(args.precision)
    amp_enabled = amp_dtype != torch.float32 and device.type == "cuda"

    # === Load model ===
    model_args = argparse.Namespace()
    model_args.model = args.model
    model_args.precision = args.precision
    model_args.latent_channels = args.latent_channels
    model_args.dit_weight = getattr(args, "dit_weight", None)
    model_args.use_fp8 = getattr(args, "use_fp8", False)
    model_args.model_resolution = getattr(args, "model_resolution", "720p")
    model_args.load_key = getattr(args, "load_key", "module")
    model_args.text_states_dim = getattr(args, "text_states_dim", 4096)
    model_args.text_states_dim_2 = getattr(args, "text_states_dim_2", 768)

    factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
    model = load_model(
        model_args,
        in_channels=model_args.latent_channels,
        out_channels=model_args.latent_channels,
        factor_kwargs=factor_kwargs,
    )

    if args.ckpt_dir:
        from hyvideo.inference import Inference
        if model_args.dit_weight is None:
            model_args.dit_weight = Path(args.ckpt_dir) / f"hunyuan-video-t2v-{model_args.model_resolution}"
        else:
            model_args.dit_weight = Path(model_args.dit_weight)
        model = Inference.load_state_dict(model_args, model, Path(args.ckpt_dir))

    if args.block_idx >= len(model.single_blocks):
        raise ValueError(f"block_idx {args.block_idx} is out of range")

    block = model.single_blocks[args.block_idx]

    available_paths = find_linear_paths(block)
    if args.target_modules == ["auto"]:
        target_paths = [p for p in available_paths if p.split(".")[-1] in {"linear1", "linear2"}]
        target_paths = target_paths or available_paths
    else:
        target_paths = args.target_modules

    print("LoRA target modules:", target_paths)

    inject_lora(block, target_paths, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    freeze_except_lora(block)
    block.to(device=device, dtype=dtype)

    # === Dataset ===
    print("Preloading dataset to GPU...")
    dataset = HunyuanMultiRootDataset(
        roots=[Path(p) for p in args.feature_roots],
        block_idx=args.block_idx,
        prev_block_idx=args.prev_block_idx,
        steps=args.steps,
        offset=args.offset,
        map_location="cuda",
        dtype=dtype,
        preload=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: b[0],
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(iter_lora_parameters(block), lr=args.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"Start training ({len(dataset)} steps × {len(args.feature_roots)} roots per step)...")

    # ===============================
    # Best checkpoint tracking
    # ===============================
    best_loss = float("inf")
    best_epoch = -1
    best_path = Path(args.save_path).with_suffix(".best.pt")

    # ===============================
    # Early stop config（原封不动）
    # ===============================
    epoch_window = 50
    pct_threshold = 0
    patience_windows = 3
    _prev_window_avg = None
    _stagnant_count = 0
    _early_stopped = False

    for epoch in range(args.epochs):
        block.train()
        total_loss = 0.0
        epoch_start = time.time()

        for samples in loader:
            optimizer.zero_grad(set_to_none=True)

            batch_loss = 0.0
            valid_count = 0

            for s in samples:
                freqs_cis = (
                    (s["freqs_cos"], s["freqs_sin"])
                    if s["freqs_cos"] is not None
                    else None
                )

                if amp_enabled:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        pred = block(
                            s["x_input"],
                            s["vec"],
                            s["txt_seq_len"],
                            s["cu_seqlens_q"],
                            s["cu_seqlens_kv"],
                            s["max_seqlen_q"],
                            s["max_seqlen_kv"],
                            freqs_cis,
                        )
                        loss = F.mse_loss(pred, s["target"]).float()
                else:
                    pred = block(
                        s["x_input"],
                        s["vec"],
                        s["txt_seq_len"],
                        s["cu_seqlens_q"],
                        s["cu_seqlens_kv"],
                        s["max_seqlen_q"],
                        s["max_seqlen_kv"],
                        freqs_cis,
                    )
                    loss = F.mse_loss(pred, s["target"]).float()

                if not torch.isfinite(loss):
                    print(f"[Warn] non-finite loss at epoch {epoch+1}, step {s['step']}. Skip.")
                    continue

                batch_loss += loss
                valid_count += 1

            if valid_count == 0:
                continue

            batch_loss /= valid_count
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(iter_lora_parameters(block)), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += batch_loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        epoch_time = time.time() - epoch_start

        # ===== 保存 best（最小 loss）=====
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

            best_state = adapter_state_dict(block)
            best_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "state_dict": best_state,
                    **training_schedule_metadata(args, "hunyuan"),
                    "block_idx": args.block_idx,
                    "epoch": best_epoch,
                    "best_loss": best_loss,
                    "base_frozen": True,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "target_modules": target_paths,
                    "steps": args.steps,
                    "offset": args.offset,
                    "dropout": args.dropout,
                },
                best_path,
            )
            torch.save(
                {
                    "state_dict": best_state,
                    **training_schedule_metadata(args, "hunyuan"),
                    "base_frozen": True,
                    "dropout": args.dropout,
                    "block_idx": args.block_idx,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "target_modules": target_paths,
                    "steps": args.steps,
                    "offset": args.offset,
                    "epoch": best_epoch,
                    "best_loss": best_loss,
                },
                args.save_path,
            )

        # ===== epoch 打印（原封不动）=====
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"mean_loss={avg_loss:.6f} "
                f"(avg over {len(args.feature_roots)} dirs) | "
                f"time={epoch_time:.2f}s"
            )
            print(f"[Best] epoch={best_epoch} loss={best_loss:.6f}")

        # ===== Early stop（原封不动）=====
        if (epoch + 1) % epoch_window == 0 and (epoch + 1) >= 400:
            if _prev_window_avg is not None:
                rel_change = abs(best_loss - _prev_window_avg) / max(_prev_window_avg, 1e-12)
                if (epoch + 1) % 50 == 0:
                    print(
                        f"[EarlyStop] window change={rel_change*100:.2f}% "
                        f"(threshold={pct_threshold*100:.1f}%)"
                    )
                if rel_change < pct_threshold:
                    _stagnant_count += 1
                    if (epoch + 1) % 50 == 0:
                        print(
                            f"[EarlyStop] stagnant windows: "
                            f"{_stagnant_count}/{patience_windows}"
                        )
                else:
                    _stagnant_count = 0

                if _stagnant_count >= patience_windows:
                    print("[EarlyStop] Loss stagnated across windows. Stopping early.")
                    _early_stopped = True
                    break

            _prev_window_avg = best_loss

    if _early_stopped:
        print("[EarlyStop] Training stopped before reaching max epochs.")

    print(f"Training finished. Best loss={best_loss:.6f} at epoch {best_epoch}")
    print(f"✅ Best LoRA saved to {best_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA training for the final single_block in HunyuanVideo.")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory containing the original HunyuanVideo weights.")
    parser.add_argument(
        "--feature-roots",
        type=str,
        nargs="+",
        required=True,
        help="List of feature directories (e.g. feature/0 feature/1 ...).",
    )
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="Training steps (self.cnt indices).")
    parser.add_argument("--block-idx", type=int, default=39, help="Index of the target single block (default: 39).")
    parser.add_argument("--prev-block-idx", type=int, default=38, help="Index that produced x_input.")
    parser.add_argument("--offset", type=int, help="Cache distance; derived from --N when supplied, otherwise 1.")
    add_interval_argument(parser)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--target-modules", nargs="+", default=["auto"], help="Specific linear module paths to wrap.")
    parser.add_argument("--save-path", type=str, default="hunyuan_block39_lora.pt")
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill", help="Model name from HUNYUAN_VIDEO_CONFIG")
    parser.add_argument("--latent-channels", type=int, default=16, help="Latent channels")
    parser.add_argument("--dit-weight", type=str, default=None, help="Path to DiT weight file or directory")
    parser.add_argument("--model-resolution", type=str, default="720p", help="Model resolution")
    parser.add_argument("--load-key", type=str, default="module", help="Key to load from checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
