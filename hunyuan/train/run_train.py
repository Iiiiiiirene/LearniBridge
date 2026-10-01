"""Explicit per-step scheduling; at most one training process per selected GPU."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learnibridge.schedule import CacheSchedule, adapter_filename, add_interval_argument, validate_interval_features

TRAINING_DEFAULTS = {
    "flux": {
        "steps": [7, 14, 21, 28, 35, 42, 49],
        "offsets": [6, 6, 6, 6, 6, 6, 6],
        "epochs": 800, "lr": 1e-3,
    },
    "hunyuan": {
        "steps": [2, 10, 20, 30, 39, 45],
        "offsets": [1, 7, 9, 9, 8, 5],
        "epochs": 5000, "lr": 3e-3,
    },
    "wan": {
        "steps": [10, 14, 19, 25, 31, 37, 42, 47],
        "offsets": [2, 3, 4, 5, 5, 5, 5, 4],
        "epochs": 5000, "lr": 1e-4,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(model="hunyuan")
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--feature-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--offsets", type=int, nargs="+")
    add_interval_argument(parser)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    schedule = None
    if args.skip_interval is not None:
        if args.steps is not None or args.offsets is not None:
            parser.error("--N generates targets and offsets; do not also supply --steps/--offsets.")
        try:
            schedule = CacheSchedule(args.num_steps, args.skip_interval)
        except ValueError as error:
            parser.error(str(error))
        args.steps = schedule.replace_steps
        args.offsets = [schedule.offset(step) for step in args.steps]
    elif args.steps is None and args.offsets is None:
        args.steps = TRAINING_DEFAULTS[args.model]["steps"]
        args.offsets = TRAINING_DEFAULTS[args.model]["offsets"]
    elif args.steps is None or args.offsets is None:
        parser.error("Supply both --steps and --offsets, or omit both to use the original arrays.")
    if args.epochs is None:
        args.epochs = TRAINING_DEFAULTS[args.model]["epochs"]
    if args.lr is None:
        args.lr = TRAINING_DEFAULTS[args.model]["lr"]
    if len(args.steps) != len(args.offsets) or len(set(args.steps)) != len(args.steps):
        parser.error("steps/offsets must have equal length and steps must be unique.")
    if any(offset < 1 or step < offset for step, offset in zip(args.steps, args.offsets)):
        parser.error("Every target step must satisfy 1 <= offset <= step.")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("Each GPU may appear only once.")
    if min(args.rank, args.epochs) < 1:
        parser.error("rank and epochs must be positive.")
    checkpoint = args.ckpt_dir.resolve()
    roots = [str(path.resolve()) for path in args.feature_roots]
    output = args.output_dir.resolve()
    jobs = []
    for step, offset in zip(args.steps, args.offsets):
        for layer in ([0, 1] if args.model == "wan" else [None]):
            block = {"flux": 37, "hunyuan": 39, "wan": 29}[args.model]
            filename = adapter_filename(args.model, step, layer)
            command = [
                sys.executable, str(ROOT / "run.py"), "train-step",
                "--ckpt-dir", str(checkpoint), "--steps", str(step), "--block-idx", str(block),
                "--rank", str(args.rank), "--alpha", str(args.alpha),
                "--epochs", str(args.epochs), "--lr", str(args.lr),
                "--save-path", str(output / filename),
            ]
            if layer is None:
                command += ["--feature-roots", *roots, "--offset", str(offset),
                            "--prev-block-idx", str(block - 1), "--precision", "bf16"]
            else:
                command += ["--feature-root", *roots, "--n", str(offset), "--layer-idx", str(layer)]
            if args.model == "hunyuan":
                command += [
                    "--dit-weight", str(checkpoint / "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"),
                ]
            if schedule is not None:
                command += ["--N", str(schedule.interval), "--num-steps", str(schedule.num_steps)]
            jobs.append((command, output / filename, output / f"{filename}.log"))
    if args.dry_run:
        if schedule is not None:
            print(json.dumps(schedule.manifest(args.model), indent=2))
        for index, (command, _, _) in enumerate(jobs):
            print(f"CUDA_VISIBLE_DEVICES={args.gpus[index % len(args.gpus)]} {shlex.join(command)}")
        return 0
    if not jobs:
        print("All requested sampling steps use full computation; no adapters need training.")
        return 0
    if not checkpoint.is_dir() or any(not Path(path).is_dir() for path in roots):
        parser.error("Checkpoint and all feature roots must be existing local directories.")
    if schedule is not None:
        try:
            validate_interval_features(roots, schedule, args.model)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    if (output / "schedule.json").exists() or any(destination.exists() or log.exists() for _, destination, log in jobs):
        parser.error("Refusing to overwrite existing adapters/logs; use a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    if schedule is not None:
        (output / "schedule.json").write_text(json.dumps(schedule.manifest(args.model), indent=2) + "\n")
    for start in range(0, len(jobs), len(args.gpus)):
        running = []
        for gpu, (command, _, log) in zip(args.gpus, jobs[start:start + len(args.gpus)]):
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
            with log.open("w") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            running.append((process, log))
            print(f"Started GPU {gpu}: {log}", flush=True)
        failed = []
        for process, log in running:
            returncode = process.wait()
            if returncode:
                failed.append((returncode, log))
        if failed:
            print(f"Training failed; no further jobs submitted: {failed}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
