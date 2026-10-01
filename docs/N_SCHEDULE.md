# Periodic caching with N

## Definition

The paper's inference description defines N as the interval between full-compute sampling steps. This implementation uses zero-based sampling indices:

- Full computation: `step % N == 0`.
- Calibration: `step % N != 0`.
- Cached source: `source = step - step % N`.
- Training offset: `offset = step - source`, between 1 and N-1.

Thus N=5 uses full steps 0, 5, 10, ...; targets 6, 7, 8, 9 all use source 5 with offsets 1, 2, 3, 4. N is not the offset, the number of skipped steps is N-1 between full steps, and sampling indices are not raw diffusion timestep values.

The final partial interval is handled without inventing another sampling step. With 12 total steps and N=5, the final full step is 10 and target 11 uses source 10. N=1 always uses full computation; training launches no adapter jobs and inference needs no adapters.

Wan uses the same sampling indices for both CFG branches. Two forward calls at a sampling step do not double N. Each branch retains its own cache and adapter file.

## Training and inference

Use `--N` (alias `--skip-interval`) for all three backends. The high-level training scheduler also takes `--num-steps`, defaulting to 50:

```bash
python scripts/train_schedule.py --model flux --N 5 --num-steps 50 \
  --ckpt-dir "$FLUX_MODEL" --feature-roots cache/flux/{0..9} \
  --output-dir runs/flux/N5/adapters --gpus 0 --dry-run
```

Remove `--dry-run` to train. Add `--epochs 1` only for a connectivity check. The scheduler creates a separate job for each calibration target and, for Wan, each CFG branch. Original backend learning-rate and epoch defaults are retained.

Inference uses the same N and total step count:

| Backend | Interval option | Total-step option | Adapter option |
| --- | --- | --- | --- |
| FLUX | `--N 5` | `--num-steps 50` | `--adapter-dir` |
| HunyuanVideo | `--N 5` | `--infer-steps 50` | `--lora-dir` |
| Wan | `--N 5` | `--sample_steps 50` | `--lora_dir` |

See the root README for complete three-step examples. Video batch scripts accept exported `N` and `NUM_STEPS`; unset `N` to use their legacy lists.

For low-level `run.py MODEL train`, continue to choose `--steps`; adding `--N` derives the required offset. A single low-level job can include multiple targets only when their offsets are equal. Use the scheduler for targets with different offsets.

## Cache and checkpoint checks

- Pre-calibration still runs the complete original trajectory. Changing only N can reuse a complete cache with the same total step count and generation settings.
- Caches collected only for an old hand-selected target list may not cover the new pairs. Training checks that every required source, target, and conditioning file exists for every supplied prompt.
- When `metadata.json` is present, its recorded total step count must match training. Legacy caches without that metadata emit a warning because their sampling trajectory cannot be independently certified.
- Training writes `schedule.json`; each new adapter also records the model, N, total steps, target steps, offsets, and source steps.
- Inference validates every required adapter before model loading and checks the actual cached source at runtime. Wrong N, wrong total steps, wrong backend/block, missing adapters, or mixed Wan CFG branches are errors.
- Keep outputs in separate directories for each N. Loading an existing adapter filename does not prove that its training pair matches the new schedule.

## Compatibility and validation scope

The new periodic mode calibrates **every** intermediate step. For 50 steps and N=5 this means 10 full steps, 40 calibration steps, and 40 adapters for FLUX/HunyuanVideo or 80 for Wan. This is not a relabeling of the old manually selected LoRA targets.

Omitting `--N` retains the previous manual training arrays, explicit inference lists, and cached-residual fallback for steps without adapters. Manual step/offset lists cannot be mixed with `--N`. In Wan, lowercase `--n` remains the legacy training offset and must not be confused with uppercase `--N`.

Old checkpoints without schedule metadata remain usable through the legacy path; they are rejected by periodic mode rather than silently treated as N-compatible. Train new periodic adapters to use the new mode.

`scripts/validate_schedule.py` checks schedule generation, edge cases, training CLI pairing, checkpoint compatibility, missing data, CFG isolation, and backward compatibility without loading models. These checks do not establish image/video quality, convergence, or the paper's measured speedups. The earlier one-epoch GPU report predates this N implementation and is not validation of the new mode.
