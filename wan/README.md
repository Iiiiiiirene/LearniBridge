# Wan2.1-T2V-1.3B

Use a separate environment with CUDA-enabled PyTorch and a matching torchvision build. Follow the upstream Wan2.1 environment instructions, then install `requirements/wan.txt`. Install a FlashAttention build compatible with your PyTorch/CUDA environment when using the upstream FlashAttention implementation.

```bash
export WAN_SOURCE=/path/to/Wan2.1
export PYTHONPATH="$WAN_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_DIR=/path/to/Wan2.1-T2V-1.3B
export N=5
```

The checkpoint directory must include the original diffusion model, T5 encoder, VAE, and configuration files.

## 1. Cache

Run from the LearniBridge root:

```bash
python run.py wan cache --task t2v-1.3B --ckpt_dir "$CKPT_DIR" \
  --prompt-file wan/prompts/prompts.txt --max-prompts 11 \
  --feature-dir cache/wan --cache-blocks 28 29 \
  --size '832*480' --frame_num 81 --sample_steps 50 \
  --base_seed 42 --sample_shift 5 --sample_guide_scale 5 \
  --offload_model True --t5_cpu
```

Features are stored in numbered prompt directories, with separate conditional/unconditional CFG metadata. `--cache-blocks 28 29` saves the two blocks needed by the final-block trainer without skipping model computation. Omitting it retains the original all-block export.

## 2. Train

```bash
python scripts/train_schedule.py --model wan --ckpt-dir "$CKPT_DIR" \
  --N "$N" --num-steps 50 \
  --feature-roots cache/wan/{0..10} \
  --output-dir "runs/wan/N${N}/adapters" --gpus 0
```

`--N` derives targets and cache offsets automatically. `N` counts sampling steps, not the two CFG forward calls within each step. A 50-step run with `N=5` trains 80 adapters: one per intermediate step and CFG branch. Training and inference must share `N` and the total step count. Use `--dry-run` to preview or `--epochs 1` for a connectivity check.

Without `--N`, the original runner defaults are:

- Targets: `10 14 19 25 31 37 42 47`
- Offsets: `2 3 4 5 5 5 5 4`
- Rank/alpha: `32 / 64`; learning rate: `1e-4`; epochs: `5000`

Both CFG branches are trained and saved separately. Each requested feature root is used.

## 3. Infer

For a single video, load the adapters you just trained and use the first original test prompt:

```bash
mkdir -p "runs/wan/N${N}/videos"
python run.py wan infer \
  --N "$N" \
  --task t2v-1.3B --ckpt_dir "$CKPT_DIR" \
  --lora_dir "runs/wan/N${N}/adapters" \
  --prompt "$(head -n 1 wan/prompts/wan_sample_50.txt)" \
  --lora_rank 32 --lora_alpha 64 \
  --size '832*480' --frame_num 81 --sample_steps 50 \
  --base_seed 42 --sample_shift 5 --sample_guide_scale 5 \
  --offload_model True --t5_cpu \
  --save_file "runs/wan/N${N}/videos/example.mp4"
```

The output is `runs/wan/N${N}/videos/example.mp4`. Replace the `--prompt` value for another example and use a new `--save_file` path for repeat runs. `--N` generates the same schedule as training and validates both CFG adapter branches. It cannot be combined with explicit replacement steps/ranges or `--use_pred_first`.

To process the original batch-inference list:

```bash
export LORA_DIR="$PWD/runs/wan/N${N}/adapters"
export OUTPUT_DIR="$PWD/runs/wan/N${N}/videos"
export NUM_STEPS=50
CUDA_VISIBLE_DEVICES=0 bash wan/infer/batch_infer.sh online
```

The batch script forwards the exported `N` and `NUM_STEPS` and reads `wan/prompts/wan_sample_50.txt` by default. For legacy adapters, unset `N` and supply their adapter directory to retain the original batch replacement list and cached-residual fallback. The low-level trainer's legacy lowercase `--n` is a cache offset, not the uppercase interval `--N`.

For a full-compute baseline, set a different `OUTPUT_DIR` and run `bash wan/infer/batch_infer.sh baseline`. Keep the seed, prompt, size, frame count, and sampling settings the same.
