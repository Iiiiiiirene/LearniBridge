# FLUX.1-dev

Run from the repository root, in an environment with CUDA-enabled PyTorch:

```bash
python -m pip install -r flux/requirements.txt
export FLUX_MODEL=/path/to/FLUX.1-dev
export N=5
```

Use complete local Diffusers-format model weights, including the transformer, VAE, text encoders, and tokenizers.

## 1. Cache

```bash
python flux/run.py cache --model-path "$FLUX_MODEL" \
  --prompt-file flux/prompts/train.txt --max-prompts 10 \
  --num-steps 50 --height 1024 --width 1024 --precision fp16 \
  --output-dir cache/flux
```

## 2. Train

```bash
python flux/run.py train --ckpt-dir "$FLUX_MODEL" \
  --N "$N" --num-steps 50 --feature-roots cache/flux/{0..9} \
  --output-dir "runs/flux/N${N}/adapters" --gpus 0
```

Add `--dry-run` to preview the training jobs or `--epochs 1` for a connectivity check. The default is 800 epochs. Use a new output directory for each run.

## 3. Infer

```bash
python flux/run.py infer --model-path "$FLUX_MODEL" \
  --N "$N" --adapter-dir "runs/flux/N${N}/adapters" \
  --prompt "Three-quarters front view of a blue 1977 Porsche 911 coming around a curve in a mountain road" \
  --num-steps 50 --seed 42 --output-dir "runs/flux/N${N}/images"
```

For batch generation, replace `--prompt` with `--prompt-file flux/prompts/test.txt`.

## Baseline

```bash
python flux/run.py baseline --model-path "$FLUX_MODEL" \
  --prompt-file flux/prompts/test.txt --max-prompts 25 \
  --num-steps 50 --precision fp16 --output-dir runs/flux/baseline
```

Keep the prompt, seed, size, step count, and precision identical when comparing outputs. Training and inference must use matching adapters and settings.
