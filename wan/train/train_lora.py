import argparse
import time
import logging
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from learnibridge.lora import adapter_state_dict, freeze_except_lora, inject_lora, iter_lora_parameters

logging.getLogger("torch._inductor").setLevel(logging.ERROR)

from learnibridge.wan_data import FeatureBlockDataset
from learnibridge.schedule import (
    add_interval_argument, resolve_training_interval, training_schedule_metadata, validate_training_features,
)


def forward_block_once(block, meta: dict[str, torch.Tensor], x_in: torch.Tensor) -> torch.Tensor:
    device = block.modulation.device
    dtype  = next(block.parameters()).dtype 

    x        = x_in.to(device=device, dtype=dtype)
    e        = meta["e"].to(device=device, dtype=dtype)
    context  = meta["context"].to(device=device, dtype=dtype)
    freqs = meta["freqs"].to(device=device)

    seq_lens   = meta["seq_lens"].to(device=device, dtype=torch.long)
    grid_sizes = meta["grid_sizes"].to(device=device, dtype=torch.long)
    context_lens = meta.get("context_lens")
    if context_lens is not None and not torch.is_tensor(context_lens):
        context_lens = torch.as_tensor(context_lens)
    if context_lens is not None:
        context_lens = context_lens.to(device=device, dtype=torch.long)

    return block(
        x,
        e=e,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_lens=context_lens,
    )

def train(args):
    resolve_training_interval(args, offset_name="n", legacy_offset=None)
    validate_training_features(args, "wan")
    from wan.modules.model import WanModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WanModel.from_pretrained(args.ckpt_dir).to(device).eval()
    block = model.blocks[args.block_idx]

    targets = [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.0", "ffn.2",
    ]
    inject_lora(block, targets, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    freeze_except_lora(block)
    block.to(device=device, dtype=torch.float32)

    dataset = FeatureBlockDataset(
        [Path(p) for p in args.feature_root],  # 多个路径
        args.block_idx,
        args.layer_idx,
        args.n,
        args.steps,
        device=device,
        preload=True
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(iter_lora_parameters(block), lr=args.lr, weight_decay=0.0)
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda')
    
    # ===== 在循环前定义一些变量 =====
    epoch_window = 100
    pct_threshold = 0.06
    prev_avg_loss = None
    stagnant_count = 0
    
    ####含有计时器
    for epoch in range(args.epochs):
        block.train()
        total_loss = 0.0
        epoch_start = time.time()

        for step_idx, sample in enumerate(loader):
            step_start = time.time()
            data_load_time = step_start - epoch_start if 'epoch_start' in locals() else 0.0
            epoch_start = step_start

            torch.cuda.synchronize()
            zero_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            zero_time = time.time() - zero_start

            torch.cuda.synchronize()
            transfer_start = time.time()
            x_input = sample["x_input"].to(device, non_blocking=True)
            target = sample["target"].to(device, non_blocking=True)
            torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start

            torch.cuda.synchronize()
            fwd_start = time.time()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred = forward_block_once(block, sample["meta"], x_input)
                loss = F.mse_loss(pred, target)
            torch.cuda.synchronize()
            forward_time = time.time() - fwd_start

            torch.cuda.synchronize()
            bwd_start = time.time()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            backward_time = time.time() - bwd_start

            torch.cuda.synchronize()
            opt_start = time.time()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            optimizer_time = time.time() - opt_start

            total_time = (
                data_load_time
                + zero_time
                + transfer_time
                + forward_time
                + backward_time
                + optimizer_time
            )

            total_loss += loss.item()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===", flush=True)
                print(
                    f"Loss: {loss.item():.6f} | "
                    f"Data: {data_load_time:.4f}s | "
                    f"Zero: {zero_time:.4f}s | "
                    f"Transfer: {transfer_time:.4f}s | "
                    f"Fwd: {forward_time:.4f}s | "
                    f"Bwd: {backward_time:.4f}s | "
                    f"Opt: {optimizer_time:.4f}s | "
                    f"Total: {total_time:.4f}s",
                    flush=True
                )
            epoch_start = time.time()

        avg_epoch_loss = total_loss / max(len(loader), 1)
        # ===== Check loss change every 100 epochs =====
        if (epoch + 1) % epoch_window == 0:
            # Compare current 100-epoch average loss with the previous window
            if prev_avg_loss is not None:
                rel_change = abs(avg_epoch_loss - prev_avg_loss) / max(prev_avg_loss, 1e-12)
                print(f"[Check] Loss change: {rel_change*100:.2f}% (threshold {pct_threshold*100:.1f}%)", flush=True)

                # Count how many consecutive times the change is below threshold
                if rel_change < pct_threshold:
                    stagnant_count += 1
                    print(f"→ Consecutive low-change count: {stagnant_count}/3", flush=True)
                else:
                    stagnant_count = 0  # reset counter if loss change is large enough

                # If the loss change stays small for 3 consecutive windows, stop training early
                if stagnant_count >= 3:
                    print(
                        f"\n[Early Stop] Loss change < {pct_threshold*100:.1f}% "
                        f"for 3 consecutive {epoch_window}-epoch windows. Stopping training.",
                        flush=True
                    )
                    break

            # Store current window average loss for next comparison
            prev_avg_loss = avg_epoch_loss

            
    # ####不含计时器
    # for epoch in range(args.epochs):
    #     block.train()
    #     total_loss = 0.0

    #     for step_idx, sample in enumerate(loader):
    #         optimizer.zero_grad(set_to_none=True)

    #         x_input = sample["x_input"].to(device, non_blocking=True)
    #         target = sample["target"].to(device, non_blocking=True)

    #         with torch.amp.autocast('cuda', dtype=torch.float16):
    #             pred = forward_block_once(block, sample["meta"], x_input)
    #             loss = F.mse_loss(pred, target)

    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()

    #         total_loss += loss.item()

    #     avg_loss = total_loss / len(loader)
    #     if (epoch + 1) % 100 == 0 or epoch == 0:
    #         print(f"Epoch {epoch + 1}/{args.epochs} | loss: {avg_loss:.6f}", flush=True)
            
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **training_schedule_metadata(args, "wan"),
        "state_dict": adapter_state_dict(block), "block_idx": args.block_idx,
        "base_frozen": True, "rank": args.rank, "alpha": args.alpha,
        "dropout": args.dropout, "target_modules": targets,
        "steps": args.steps, "offset": args.n, "layer_idx": args.layer_idx,
    }, save_path)
    print(f"Saved LoRA block adapter to {save_path}")

    block.eval()
    with torch.no_grad():
        for sample in loader:
            step = sample["step"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = forward_block_once(block, sample["meta"], sample["x_input"])
            target = sample["target"].to(device=pred.device, dtype=pred.dtype)
            mse = F.mse_loss(pred, target).item()
            print(f"Step {step}: MSE = {mse:.6f}")
            pred_path = save_path.parent / f"pred_block_{args.block_idx}_{args.layer_idx}_output_step_{step}.pt"
            torch.save(pred.cpu(), pred_path)

def parse_args():
    from wan.configs import WAN_CONFIGS
    parser = argparse.ArgumentParser(description="Train LoRA adapters against cached block outputs.")
    parser.add_argument("--task", default="t2v-1.3B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--feature-root", nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--block-idx", type=int, required=True)
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--n", type=int, help="Legacy cache offset; distinct from the periodic interval --N.")
    add_interval_argument(parser)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-path", default="./lora_block.pt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
