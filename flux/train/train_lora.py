import argparse
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
from learnibridge.flux_compat import forward_single_block
from learnibridge.schedule import (
    add_interval_argument, resolve_training_interval, training_schedule_metadata, validate_training_features,
)
from learnibridge.lora import (
    LoRALinear,
    adapter_state_dict,
    freeze_except_lora,
    inject_lora,
    iter_lora_parameters,
)

try:
    from diffusers.models.transformers import FluxTransformer2DModel
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("diffusers>=0.29.0 is required to load FluxTransformer2DModel.") from exc


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FluxMultiRootDataset(Dataset):
    def __init__(
        self,
        roots: Sequence[Path],
        block_idx: int,
        prev_block_idx: int,
        steps: Sequence[int],
        offset: int,
        meta_offset: int = 0,
        map_location: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.roots = [Path(r) for r in roots]
        self.block_idx = block_idx
        self.prev_block_idx = prev_block_idx
        self.steps = list(steps)
        self.offset = offset
        self.meta_offset = meta_offset
        self.map_location = map_location
        self.dtype = dtype
        if not self.roots or not self.steps or offset < 1 or any(step < offset for step in self.steps):
            raise ValueError("Require feature roots, nonempty steps, and 1 <= offset <= each target step.")

    def _load_joint_kwargs(self, path: Path) -> dict[str, Any] | None:
        if path.exists():
            data = torch.load(path, map_location=self.map_location)
            if isinstance(data, dict):
                return data
        return None

    def _load_sample(self, root: Path, step: int) -> dict[str, Any] | None:
        src_step = step - self.offset
        meta_step = step - self.meta_offset
        try:
            x_path = root / f"singleblock_{self.prev_block_idx}_output_step_{src_step}.pt"
            y_path = root / f"singleblock_{self.block_idx}_output_step_{step}.pt"
            temb_path = root / f"temb_step_{meta_step}.pt"
            rot_path = root / f"image_rotary_emb_step_{meta_step}.pt"
            missing = [str(path) for path in (x_path, y_path, temb_path, rot_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing training features: {missing}")

            sample = {
                "x_input": torch.load(x_path, map_location=self.map_location).to(self.dtype),
                "target": torch.load(y_path, map_location=self.map_location).to(self.dtype),
                "temb": torch.load(temb_path, map_location=self.map_location).to(self.dtype),
                "image_rotary_emb": torch.load(rot_path, map_location=self.map_location),
                "joint_attention_kwargs": self._load_joint_kwargs(root / f"joint_attention_kwargs_step_{meta_step}.pt"),
                "text_length": (
                    torch.load(root / f"text_length_step_{meta_step}.pt", weights_only=True)
                    if (root / f"text_length_step_{meta_step}.pt").exists() else None
                ),
            }
            return sample
        except Exception as exc:
            raise RuntimeError(f"Cannot load features from {root} for step {step}") from exc

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        step = self.steps[idx]
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


def save_predictions(
    block: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
    block_idx: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    block.eval()
    with torch.no_grad():
        for sample in loader:
            step = sample["step"]
            x_input = move_to(sample["x_input"], device, dtype)
            temb = move_to(sample["temb"], device, dtype)
            image_rotary_emb = move_to(sample["image_rotary_emb"], device, dtype)
            joint_kwargs = move_to(sample["joint_attention_kwargs"], device, dtype)
            pred = block(
                hidden_states=x_input,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_kwargs,
            )
            torch.save(pred.cpu(), out_dir / f"pred_singleblock_{block_idx}_output_step_{step}.pt")


# ---------------------------------------------------------------------------
# Train (Multi-root batch version)
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    resolve_training_interval(args)
    validate_training_features(args, "flux")
    if args.epochs < 1 or args.rank < 1:
        raise ValueError("epochs and rank must be positive.")
    if getattr(args, "stack_blocks", None) and args.stack_blocks != [args.block_idx]:
        raise ValueError("The canonical path trains one final block; use the archived stacked experiment separately.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    amp_dtype = autocast_dtype(args.precision)
    amp_enabled = amp_dtype != torch.float32 and device.type == "cuda"

    # === Load model ===
    model = FluxTransformer2DModel.from_pretrained(
        args.ckpt_dir, subfolder="transformer", local_files_only=True
    )

    # 可选：堆叠多个相邻的 single_transformer_blocks 进行联合 LoRA 训练
    if args.block_idx != len(model.single_transformer_blocks) - 1 or args.prev_block_idx != args.block_idx - 1:
        raise ValueError("Canonical calibration requires the final block and its immediate predecessor.")
    block = model.single_transformer_blocks[args.block_idx]

    available_paths = find_linear_paths(block)
    if args.target_modules == ["auto"]:
        target_paths = available_paths
    else:
        target_paths = args.target_modules
        missing = [p for p in target_paths if p not in available_paths]
        if missing:
            raise ValueError(f"Requested target modules not found: {missing}")

    print("LoRA target modules:", target_paths)
    inject_lora(block, target_paths, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    freeze_except_lora(block)
    block.to(device=device, dtype=dtype)
    del model

    # === Dataset ===
    dataset = FluxMultiRootDataset(
        roots=[Path(p) for p in args.feature_roots],
        block_idx=args.block_idx,
        prev_block_idx=args.prev_block_idx,
        steps=args.steps,
        offset=args.offset,
        dtype=dtype,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

    optimizer = torch.optim.AdamW(iter_lora_parameters(block), lr=args.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"Start training ({len(dataset)} steps × {len(args.feature_roots)} roots per step)...")

    import time

    # ===== 早停配置 =====
    epoch_window      = 100        # 每多少个 epoch 评估一次
    pct_threshold     = args.early_stop_threshold
    patience_windows  = 2          # 连续多少个窗口停滞则早停
    _prev_window_avg  = None
    _stagnant_count   = 0
    _early_stopped    = False

    for epoch in range(args.epochs):
        block.train()
        total_loss  = 0.0
        epoch_start = time.time()

        for samples in loader:  # 每个 batch 是一个 list（来自多个目录）
            optimizer.zero_grad(set_to_none=True)

            batch_loss  = 0.0
            valid_count = 0

            for sample in samples:
                x_input = move_to(sample["x_input"], device, dtype)
                target  = move_to(sample["target"], device, dtype)
                temb    = move_to(sample["temb"], device, dtype)
                image_rotary_emb = move_to(sample["image_rotary_emb"], device, dtype)
                joint_kwargs      = move_to(sample["joint_attention_kwargs"], device, dtype)

                if amp_enabled:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        pred = forward_single_block(
                            block,
                            hidden_states=x_input,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_kwargs,
                            text_length=sample["text_length"],
                        )
                        loss = F.mse_loss(pred.float(), target.float())
                else:
                    pred = forward_single_block(
                        block,
                        hidden_states=x_input,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_kwargs,
                        text_length=sample["text_length"],
                    )
                    loss = F.mse_loss(pred.float(), target.float())

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, step {sample['step']}.")

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

        avg_loss  = total_loss / max(len(loader), 1)
        epoch_time = time.time() - epoch_start

        # ===== 仅每 100 个 epoch 打印一次 =====
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1}/{args.epochs}] mean_loss={avg_loss:.6f} "
                  f"(avg over {len(args.feature_roots)} dirs) | time={epoch_time:.2f}s")

        if (epoch + 1) % epoch_window == 0:
            if _prev_window_avg is not None:
                rel_change = abs(avg_loss - _prev_window_avg) / max(_prev_window_avg, 1e-12)
                if (epoch + 1) % 100 == 0:
                    print(f"[EarlyStop] window change={rel_change*100:.2f}% "
                          f"(threshold={pct_threshold*100:.1f}%)")
                if rel_change < pct_threshold:
                    _stagnant_count += 1
                    if (epoch + 1) % 100 == 0:
                        print(f"[EarlyStop] stagnant windows: {_stagnant_count}/{patience_windows}")
                else:
                    _stagnant_count = 0

                if _stagnant_count >= patience_windows:
                    print("[EarlyStop] Loss stagnated across windows. Stopping early.")
                    _early_stopped = True
                    break

            _prev_window_avg = avg_loss

    if _early_stopped:
        print("[EarlyStop] Training stopped before reaching max epochs.")


    # === Save ===
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        **training_schedule_metadata(args, "flux"),
        "format_version": 1,
        "state_dict": adapter_state_dict(block),
        "block_idx": args.block_idx,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": target_paths,
        "steps": args.steps,
        "offset": args.offset,
        "seed": args.seed,
        "final_loss": avg_loss,
    }
    torch.save(payload, save_path)
    print(f"✅ Saved LoRA adapter to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA training for the final single_transformer_block in Flux.")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing the original Flux weights.")
    parser.add_argument(
        "--feature-roots",
        type=str,
        nargs="+",
        required=True,
        help="List of feature directories (e.g. feature/0 feature/1 ...).",
    )
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="Training steps (self.cnt indices).")
    parser.add_argument("--block-idx", type=int, default=37, help="Index of the target single block (default: 37).")
    parser.add_argument("--prev-block-idx", type=int, default=36, help="Index that produced x_input.")
    parser.add_argument("--offset", type=int, help="Cache distance; derived from --N when supplied, otherwise 1.")
    add_interval_argument(parser)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--early-stop-threshold", type=float, default=0.0, help="Relative loss-change threshold; zero disables early stopping.")
    parser.add_argument("--target-modules", nargs="+", default=["auto"], help="Specific linear module paths to wrap.")
    parser.add_argument("--save-path", type=str, default="flux_block37_lora.pt")
    parser.add_argument(
        "--stack-blocks",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of consecutive block indices to stack for joint LoRA (e.g., 36 37).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
