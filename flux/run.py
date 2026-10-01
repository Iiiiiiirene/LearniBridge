#!/usr/bin/env python3
"""Run this model's cache, training, or inference entry point."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENTRIES = {
    "cache": "feature/extract_features.py",
    "extract": "feature/extract_features.py",
    "train": "train/run_train.py",
    "train-step": "train/train_lora.py",
    "baseline": "infer/generate.py",
    "infer": "infer/online_gen.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Relative data paths use your current working directory.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("action", choices=ENTRIES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    module_root = ROOT
    environment = os.environ.copy()
    source_paths = [str(module_root)]
    if environment.get("PYTHONPATH"):
        source_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    environment["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, str(module_root / ENTRIES[args.action]), *arguments]
    if args.dry_run:
        print(f"cwd: {Path.cwd()}\ncommand: {shlex.join(command)}")
        return 0
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
