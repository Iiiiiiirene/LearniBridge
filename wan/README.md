# Wan2.1-T2V-1.3B

Run from the repository root. Use a separate environment with CUDA-enabled PyTorch, matching torchvision, and the upstream Wan2.1 dependencies, including compatible FlashAttention when required.

```bash
python -m pip install -r wan/requirements.txt
export WAN_SOURCE=/path/to/Wan2.1
export PYTHONPATH="$WAN_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_DIR=/path/to/Wan2.1-T2V-1.3B
export N=5
```

Prepare complete model weights, T5 encoder, VAE, and configuration files.

## 1. Cache

```bash
python wan/run.py cache --task t2v-1.3B --ckpt_dir "$CKPT_DIR" \
  --prompt-file wan/prompts/prompts.txt --max-prompts 11 \
  --feature-dir cache/wan --cache-blocks 28 29 \
  --size '832*480' --frame_num 81 --sample_steps 50 \
  --base_seed 42 --sample_shift 5 --sample_guide_scale 5 \
  --offload_model True --t5_cpu
```

## 2. Train

```bash
python wan/run.py train --ckpt-dir "$CKPT_DIR" \
  --N "$N" --num-steps 50 --feature-roots cache/wan/{0..10} \
  --output-dir "runs/wan/N${N}/adapters" --gpus 0
```

Both CFG branches are trained separately. Add `--dry-run` to preview or `--epochs 1` for a connectivity check. The default is 5000 epochs. Use a new output directory for each run.

## 3. Infer

```bash
mkdir -p "runs/wan/N${N}/videos"
python wan/run.py infer --N "$N" \
  --task t2v-1.3B --ckpt_dir "$CKPT_DIR" \
  --lora_dir "runs/wan/N${N}/adapters" \
  --prompt "$(head -n 1 wan/prompts/wan_sample_50.txt)" \
  --lora_rank 32 --lora_alpha 64 \
  --size '832*480' --frame_num 81 --sample_steps 50 \
  --base_seed 42 --sample_shift 5 --sample_guide_scale 5 \
  --offload_model True --t5_cpu \
  --save_file "runs/wan/N${N}/videos/example.mp4"
```

## Batch and baseline

```bash
export LORA_DIR="$PWD/runs/wan/N${N}/adapters"
export OUTPUT_DIR="$PWD/runs/wan/N${N}/videos"
export NUM_STEPS=50
CUDA_VISIBLE_DEVICES=0 bash wan/infer/batch_infer.sh online
```

For a full-compute baseline, choose a different `OUTPUT_DIR` and replace `online` with `baseline`. The default prompt list is `wan/prompts/wan_sample_50.txt`; set `PROMPT_FILE` to an absolute path to use another list.
