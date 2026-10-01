# FLUX.1-dev

Run commands from the repository root. Install `requirements/flux.txt` in an environment with CUDA-enabled PyTorch.

## Model weights

Set `FLUX_MODEL` to a complete local Diffusers-format FLUX.1-dev pipeline, including `model_index.json`, transformer, VAE, text encoders, and tokenizers. Do not substitute a quantized checkpoint for an original-precision experiment.

```bash
export FLUX_MODEL=/path/to/FLUX.1-dev
export N=5
```

## 1. Cache

```bash
python run.py flux cache --model-path "$FLUX_MODEL" \
  --prompt-file flux/prompts/train.txt --max-prompts 10 \
  --num-steps 50 --height 1024 --width 1024 --precision fp16 \
  --output-dir cache/flux
```

Each numbered prompt directory contains final-block inputs and outputs, timestep conditioning, rotary embeddings, and text-token lengths. Use a new or empty output directory.

## 2. Train

```bash
python scripts/train_schedule.py --model flux --ckpt-dir "$FLUX_MODEL" \
  --N "$N" --num-steps 50 --feature-roots cache/flux/{0..9} \
  --output-dir "runs/flux/N${N}/adapters" --gpus 0
```

`--N` derives every target step and its offset from the nearest full-compute step. For `N=5`, full steps are `0, 5, 10, ...` and target step 9 uses source step 5, not step 4. Training and inference must share `N` and the 50-step trajectory. Add `--dry-run` to preview, or `--epochs 1` for a connectivity check.

The original learning rate and epoch defaults are retained. Without `--N`, the original runner uses:

- Targets: `7 14 21 28 35 42 49`
- Offsets: `6 6 6 6 6 6 6`
- Rank/alpha: `32 / 64`; learning rate: `1e-3`; epochs: `800`

The canonical trainer uses target-step conditioning. The scheduler calls the included `flux/train/train_lora.py`; no external training-script filename is required.

## 3. Infer

```bash
python run.py flux infer --model-path "$FLUX_MODEL" \
  --N "$N" --adapter-dir "runs/flux/N${N}/adapters" \
  --prompt-file flux/prompts/test.txt \
  --num-steps 50 --seed 42 --output-dir "runs/flux/N${N}/images"
```

Use `--prompt` for a single image. `--N` requires matching adapters for all intermediate steps and checks checkpoint schedule metadata. `N=1` needs no adapter directory. Without `--N`, `--replace-steps` selects a legacy manual list and missing adapters use the original cached-residual branch; do not combine both options.

For an unaccelerated comparison:

```bash
python run.py flux baseline --model-path "$FLUX_MODEL" \
  --prompt-file flux/prompts/test.txt --max-prompts 25 \
  --num-steps 50 --precision fp16 --output-dir runs/flux/baseline
```

Keep the prompt, seed, size, step count, and precision identical when comparing outputs.
