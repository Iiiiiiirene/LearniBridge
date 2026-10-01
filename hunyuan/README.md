# HunyuanVideo

Use a separate Python 3.10 environment. Install CUDA-enabled PyTorch, the dependencies required by the upstream HunyuanVideo implementation, and `requirements/hunyuan.txt`.

Make the upstream implementation available without copying it into this repository:

```bash
export HUNYUAN_SOURCE=/path/to/HunyuanVideo
export PYTHONPATH="$HUNYUAN_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_BASE=/path/to/HunyuanVideo-weights
export DIT_WEIGHT="$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
export N=5
```

The model root must contain the DiT, VAE, and prepared text encoders required by upstream HunyuanVideo. Set `MODEL_BASE` before starting Python because upstream code reads it during import.

## 1. Cache

Run from the LearniBridge root:

```bash
python run.py hunyuan cache \
  --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
  --prompt-file hunyuan/prompts/prompt.txt --max-prompts 6 \
  --feature-dir cache/hunyuan --cache-blocks 38 39 \
  --video-size 544 960 --video-length 49 --infer-steps 50 \
  --seed 42 --embedded-cfg-scale 6.0 --flow-shift 7.0 \
  --flow-reverse --use-cpu-offload --save-path runs/hunyuan/cache-preview
```

The cache uses one numbered directory per prompt. `--cache-blocks` only selects which computed features are saved; all model blocks still run during caching. Omitting it retains the original extraction block selection.

## 2. Train

```bash
python scripts/train_schedule.py --model hunyuan --ckpt-dir "$MODEL_BASE" \
  --N "$N" --num-steps 50 \
  --feature-roots cache/hunyuan/{0..5} \
  --output-dir "runs/hunyuan/N${N}/adapters" --gpus 0
```

`--N` derives targets and offsets from full steps `0, N, 2N, ...`. The default epoch count and learning rate remain unchanged. Use the same `N` and total step count for inference. Add `--dry-run` to preview or `--epochs 1` for a connectivity check.

Without `--N`, the original runner defaults are:

- Targets: `2 10 20 30 39 45`
- Offsets: `1 7 9 9 8 5`
- Rank/alpha: `32 / 64`; learning rate: `3e-3`; epochs: `5000`

## 3. Infer

For a single video, load the adapters you just trained and use the first original test prompt:

```bash
python run.py hunyuan infer \
  --N "$N" \
  --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
  --lora-dir "runs/hunyuan/N${N}/adapters" \
  --prompt "$(head -n 1 hunyuan/prompts/sample_50.txt)" \
  --block-idx 39 --target-modules auto --lora-rank 32 --lora-alpha 64 \
  --video-size 544 960 --video-length 49 --infer-steps 50 \
  --seed 42 --embedded-cfg-scale 6.0 --flow-shift 7.0 \
  --flow-reverse --use-cpu-offload \
  --save-path "runs/hunyuan/N${N}/videos"
```

The output is an MP4 under `runs/hunyuan/N${N}/videos`. Replace the `--prompt` value for another example. `--N` generates the same schedule as training and rejects missing or mismatched adapters; no explicit replacement list is needed.

To process the original batch-inference list:

```bash
export LORA_DIR="$PWD/runs/hunyuan/N${N}/adapters"
export OUTPUT_DIR="$PWD/runs/hunyuan/N${N}/videos"
export NUM_STEPS=50
GPU=0 bash hunyuan/infer/batch_infer.sh online
```

The batch script reads `hunyuan/prompts/sample_50.txt` by default. Set `PROMPT_FILE` to an absolute path for another list because the batch script changes its working directory.

The batch script forwards the exported `N` and `NUM_STEPS`. For legacy adapters, unset `N` and supply the corresponding legacy adapter directory to retain the original batch replacement list. The legacy Python default list differs from that batch list.

For the original full-compute baseline, run `bash hunyuan/infer/batch_infer.sh baseline`. Use a separate output directory when comparing outputs.
