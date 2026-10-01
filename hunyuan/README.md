# HunyuanVideo

Run from the repository root. Use a separate Python 3.10 environment with CUDA-enabled PyTorch and the upstream HunyuanVideo dependencies.

```bash
python -m pip install -r hunyuan/requirements.txt
export HUNYUAN_SOURCE=/path/to/HunyuanVideo
export PYTHONPATH="$HUNYUAN_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_BASE=/path/to/HunyuanVideo-weights
export DIT_WEIGHT="$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
export N=5
```

Prepare the complete model, VAE, and text encoders. Set `MODEL_BASE` before starting Python.

## 1. Cache

```bash
python hunyuan/run.py cache \
  --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
  --prompt-file hunyuan/prompts/prompt.txt --max-prompts 6 \
  --feature-dir cache/hunyuan --cache-blocks 38 39 \
  --video-size 544 960 --video-length 49 --infer-steps 50 \
  --seed 42 --embedded-cfg-scale 6.0 --flow-shift 7.0 \
  --flow-reverse --use-cpu-offload --save-path runs/hunyuan/cache-preview
```

## 2. Train

```bash
python hunyuan/run.py train --ckpt-dir "$MODEL_BASE" \
  --N "$N" --num-steps 50 --feature-roots cache/hunyuan/{0..5} \
  --output-dir "runs/hunyuan/N${N}/adapters" --gpus 0
```

Add `--dry-run` to preview or `--epochs 1` for a connectivity check. The default is 5000 epochs. Use a new output directory for each run.

## 3. Infer

```bash
python hunyuan/run.py infer --N "$N" \
  --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
  --lora-dir "runs/hunyuan/N${N}/adapters" \
  --prompt "$(head -n 1 hunyuan/prompts/sample_50.txt)" \
  --block-idx 39 --target-modules auto --lora-rank 32 --lora-alpha 64 \
  --video-size 544 960 --video-length 49 --infer-steps 50 \
  --seed 42 --embedded-cfg-scale 6.0 --flow-shift 7.0 \
  --flow-reverse --use-cpu-offload --save-path "runs/hunyuan/N${N}/videos"
```

## Batch and baseline

```bash
export LORA_DIR="$PWD/runs/hunyuan/N${N}/adapters"
export OUTPUT_DIR="$PWD/runs/hunyuan/N${N}/videos"
export NUM_STEPS=50
GPU=0 bash hunyuan/infer/batch_infer.sh online
```

For a full-compute baseline, choose a different `OUTPUT_DIR` and replace `online` with `baseline`. The default prompt list is `hunyuan/prompts/sample_50.txt`; set `PROMPT_FILE` to an absolute path to use another list.
