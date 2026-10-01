"""CPU-only regression checks for periodic scheduling; no model dependencies."""

import argparse
import ast
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learnibridge.schedule import (
    BLOCK_INDICES, CacheSchedule, adapter_filename, add_interval_argument,
    resolve_inference_schedule, resolve_training_interval, training_schedule_metadata,
    validate_cached_source, validate_checkpoint_schedule, validate_interval_adapters,
    validate_interval_features,
)


def checkpoint(schedule, model, step, branch=None):
    return {
        "cache_schedule": schedule.identity(model), "steps": [step],
        "offset": schedule.offset(step), "source_steps": [schedule.source_step(step)],
        "block_idx": BLOCK_INDICES[model], "layer_idx": branch,
        "state_dict": {}, "rank": 32, "alpha": 64,
    }


class ScheduleTests(unittest.TestCase):
    def test_periodic_pairs(self):
        schedule = CacheSchedule(12, 5)
        self.assertEqual(schedule.full_steps, [0, 5, 10])
        self.assertEqual(schedule.replace_steps, [1, 2, 3, 4, 6, 7, 8, 9, 11])
        self.assertEqual([schedule.offset(step) for step in schedule.replace_steps], [1, 2, 3, 4, 1, 2, 3, 4, 1])
        self.assertEqual(schedule.source_step(11), 10)

    def test_full_compute_and_short_trajectories(self):
        self.assertEqual(CacheSchedule(50, 1).full_steps, list(range(50)))
        self.assertEqual(CacheSchedule(50, 1).replace_steps, [])
        self.assertEqual(CacheSchedule(1, 5).replace_steps, [])
        self.assertEqual(CacheSchedule(3, 5).replace_steps, [1, 2])

    def test_schedule_invariants(self):
        for count in range(1, 24):
            for interval in range(1, 12):
                schedule = CacheSchedule(count, interval)
                self.assertEqual(sorted(schedule.full_steps + schedule.replace_steps), list(range(count)))
                self.assertFalse(set(schedule.full_steps) & set(schedule.replace_steps))
                last_full = None
                for step in range(count):
                    if step in schedule.full_steps:
                        last_full = step
                    else:
                        validate_cached_source(schedule, step, last_full)
                        self.assertLess(last_full, step)
                        self.assertEqual(step - last_full, schedule.offset(step))

    def test_invalid_values(self):
        for count, interval in [(0, 5), (5, 0), (-1, 2), (5, -2), (5, 1.5), (True, 2)]:
            with self.assertRaises(ValueError):
                CacheSchedule(count, interval)
        for step in [-1, 5, 1.5]:
            with self.assertRaises(ValueError):
                CacheSchedule(5, 2).source_step(step)

    def test_inference_excludes_manual_steps(self):
        self.assertIsNone(resolve_inference_schedule(None, 50, [2, 3]))
        self.assertEqual(resolve_inference_schedule(5, 50).full_steps, list(range(0, 50, 5)))
        for manual in [[], [2, 3]]:
            with self.assertRaises(ValueError):
                resolve_inference_schedule(5, 50, manual)

    def test_training_derives_offset_not_interval(self):
        for offset_name in ["offset", "n"]:
            args = SimpleNamespace(skip_interval=5, num_steps=12, steps=[4, 9], **{offset_name: None})
            resolve_training_interval(args, offset_name)
            self.assertEqual(getattr(args, offset_name), 4)
            self.assertEqual(training_schedule_metadata(args, "flux")["source_steps"], [0, 5])
        args = SimpleNamespace(skip_interval=None, steps=[4], offset=None)
        resolve_training_interval(args)
        self.assertEqual(args.offset, 1)
        self.assertEqual(training_schedule_metadata(args, "flux"), {})

    def test_training_rejects_wrong_offsets_and_full_steps(self):
        for steps, offset in [([5], None), ([1, 2], None), ([4], 5), ([12], None), ([], None)]:
            with self.assertRaises(ValueError):
                resolve_training_interval(SimpleNamespace(skip_interval=5, num_steps=12, steps=steps, offset=offset))
        with self.assertRaises(ValueError):
            resolve_training_interval(SimpleNamespace(skip_interval=None, n=None), "n", None)

    def test_checkpoint_compatibility(self):
        schedule = CacheSchedule(12, 5)
        payload = checkpoint(schedule, "flux", 9)
        validate_checkpoint_schedule(payload, schedule, "flux", 9)
        for field, bad in [
            ("cache_schedule", CacheSchedule(12, 4).identity("flux")),
            ("cache_schedule", CacheSchedule(50, 5).identity("flux")),
            ("cache_schedule", schedule.identity("wan")),
            ("offset", 5), ("source_steps", [4]), ("steps", [8]), ("block_idx", 38),
        ]:
            modified = {**payload, field: bad}
            with self.assertRaises(ValueError):
                validate_checkpoint_schedule(modified, schedule, "flux", 9)
        legacy = {key: value for key, value in payload.items() if key != "cache_schedule"}
        with self.assertRaises(ValueError):
            validate_checkpoint_schedule(legacy, schedule, "flux", 9)
        with self.assertRaises(ValueError):
            validate_checkpoint_schedule([], schedule, "flux", 9)

    def test_wan_cfg_branches(self):
        schedule = CacheSchedule(6, 5)
        self.assertNotEqual(adapter_filename("wan", 4, 0), adapter_filename("wan", 4, 1))
        for branch in [0, 1]:
            payload = checkpoint(schedule, "wan", 4, branch)
            validate_checkpoint_schedule(payload, schedule, "wan", 4, branch)
            with self.assertRaises(ValueError):
                validate_checkpoint_schedule(payload, schedule, "wan", 4, 1 - branch)

    def test_directory_preflight(self):
        schedule = CacheSchedule(6, 3)
        for model in BLOCK_INDICES:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for step in schedule.replace_steps:
                    for branch in ([0, 1] if model == "wan" else [None]):
                        (directory / adapter_filename(model, step, branch)).write_text(
                            json.dumps(checkpoint(schedule, model, step, branch)),
                        )
                manifest = directory / "schedule.json"
                manifest.write_text(json.dumps(schedule.manifest(model)))
                loader = lambda path: json.loads(path.read_text())
                validate_interval_adapters(directory, schedule, model, loader)
                with self.assertRaises(ValueError):
                    validate_interval_adapters(directory, schedule, model, loader, rank=16)
                with self.assertRaises(ValueError):
                    validate_interval_adapters(directory, schedule, model, loader, alpha=32)
                manifest.write_text(json.dumps(CacheSchedule(6, 2).manifest(model)))
                with self.assertRaises(ValueError):
                    validate_interval_adapters(directory, schedule, model, loader)
                manifest.unlink()
                validate_interval_adapters(directory, schedule, model, loader)
                next(directory.glob("*.pt")).unlink()
                with self.assertRaises(FileNotFoundError):
                    validate_interval_adapters(directory, schedule, model, loader)
        validate_interval_adapters(None, CacheSchedule(50, 1), "flux")
        with self.assertRaises(ValueError):
            validate_interval_adapters(None, schedule, "flux")

    def test_stale_cache_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_cached_source(CacheSchedule(12, 5), 9, 4)
        with self.assertRaises(ValueError):
            validate_cached_source(CacheSchedule(12, 5), 1, None)
        validate_cached_source(None, 9, 4)

    def test_required_cache_pairs_and_step_count(self):
        schedule = CacheSchedule(6, 3)
        for model in BLOCK_INDICES:
            with tempfile.TemporaryDirectory() as temporary:
                roots = [Path(temporary) / "0", Path(temporary) / "1"]
                for root in roots:
                    root.mkdir()
                    (root / "metadata.json").write_text(json.dumps({"steps": 6}))
                    for step in [1, 2, 4, 5]:
                        source = 0 if step < 3 else 3
                        if model == "flux":
                            names = [f"singleblock_36_output_step_{source}.pt", f"singleblock_37_output_step_{step}.pt",
                                     f"temb_step_{step}.pt", f"image_rotary_emb_step_{step}.pt"]
                        elif model == "hunyuan":
                            names = [f"single_block38_step_{source}.pt", f"single_block39_step_{step}.pt", f"meta_step_{step}.pt"]
                        else:
                            names = [f"block_{block}_{branch}_output_step_{source if block == 28 else step}.pt"
                                     for block in [28, 29] for branch in [0, 1]]
                            names += [f"meta_step_{step}_{branch}.pt" for branch in [0, 1]]
                        for name in names:
                            (root / name).touch()
                validate_interval_features(roots, schedule, model)
                victim = next(roots[1].glob("*.pt"))
                victim.unlink()
                with self.assertRaises(FileNotFoundError):
                    validate_interval_features(roots, schedule, model)
                victim.touch()
                (roots[0] / "metadata.json").write_text(json.dumps({"steps": 50}))
                with self.assertRaises(ValueError):
                    validate_interval_features(roots, schedule, model)
                (roots[0] / "metadata.json").unlink()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    validate_interval_features(roots, schedule, model)
                self.assertEqual(len(caught), 1)

    def test_scheduler_cli_all_backends(self):
        for model in BLOCK_INDICES:
            command = [
                sys.executable, "-B", str(ROOT / "scripts/train_schedule.py"), "--model", model,
                "--ckpt-dir", "unused-model", "--feature-roots", "unused-cache",
                "--output-dir", "unused-output", "--N", "5", "--num-steps", "12",
                "--epochs", "1", "--dry-run",
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            manifest, _ = json.JSONDecoder().raw_decode(result.stdout)
            self.assertEqual(manifest, CacheSchedule(12, 5).manifest(model))
            jobs = [shlex.split(line)[1:] for line in result.stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
            self.assertEqual(len(jobs), 18 if model == "wan" else 9)
            for job in jobs:
                step = int(job[job.index("--steps") + 1])
                offset_flag = "--n" if model == "wan" else "--offset"
                self.assertEqual(int(job[job.index(offset_flag) + 1]), step % 5)
                self.assertEqual(job[job.index("--N") + 1], "5")
                self.assertEqual(job[job.index("--num-steps") + 1], "12")
            invalid = subprocess.run(command + ["--steps", "4", "--offsets", "3"], capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)

    def test_legacy_scheduler_and_n_one(self):
        for model, count in [("flux", 7), ("hunyuan", 6), ("wan", 16)]:
            command = [
                sys.executable, "-B", str(ROOT / "scripts/train_schedule.py"), "--model", model,
                "--ckpt-dir", "unused-model", "--feature-roots", "unused-cache",
                "--output-dir", "unused-output", "--dry-run",
            ]
            legacy = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(sum(line.startswith("CUDA_VISIBLE_DEVICES=") for line in legacy.stdout.splitlines()), count)
            full = subprocess.run(command + ["--N", "1"], capture_output=True, text=True, check=True)
            self.assertNotIn("CUDA_VISIBLE_DEVICES=", full.stdout)

    def test_n_argument_aliases(self):
        parser = argparse.ArgumentParser()
        add_interval_argument(parser)
        self.assertEqual(parser.parse_args(["--N", "5"]).skip_interval, 5)
        self.assertEqual(parser.parse_args(["--skip-interval", "5"]).skip_interval, 5)

    def test_training_entrypoint_parsers(self):
        for model in ["flux", "hunyuan", "wan"]:
            tree = ast.parse((ROOT / model / "train/train_lora.py").read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
            function = copy.deepcopy(function)
            function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
            namespace = {"argparse": argparse, "add_interval_argument": add_interval_argument,
                         "WAN_CONFIGS": {"t2v-1.3B": None}}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "<trainer-parser>", "exec"), namespace)
            previous = sys.argv
            try:
                sys.argv = ["trainer", "--ckpt-dir", "unused",
                            "--feature-root" if model == "wan" else "--feature-roots", "unused",
                            "--steps", "4", "--N", "5", "--num-steps", "12"]
                if model == "wan":
                    sys.argv += ["--block-idx", "29", "--layer-idx", "0"]
                parsed = namespace["parse_args"]()
            finally:
                sys.argv = previous
            offset_name = "n" if model == "wan" else "offset"
            resolve_training_interval(parsed, offset_name)
            self.assertEqual(getattr(parsed, offset_name), 4)

    def test_inference_parser_accepts_n_and_keeps_upstream_flags(self):
        for model, filename in [("flux", "online_gen.py"), ("hunyuan", "online_lora.py")]:
            tree = ast.parse((ROOT / model / "infer" / filename).read_text())
            function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
            body = function.body if function is not None else next(
                node.body for node in tree.body if isinstance(node, ast.If) and "__name__" in ast.dump(node.test)
            )
            statements = []
            for statement in body:
                if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr in ("parse_args", "parse_known_args") for node in ast.walk(statement)):
                    break
                statements.append(copy.deepcopy(statement))
            namespace = {
                "argparse": argparse, "add_interval_argument": add_interval_argument,
                "MODEL_PATH": "unused", "PROMPT_FILE": "unused", "OUTPUT_DIR": "unused",
                "NUM_INFERENCE_STEPS": 50, "SEED": 42,
            }
            exec(compile(ast.Module(body=statements, type_ignores=[]), "<inference-parser>", "exec"), namespace)
            parser = namespace["parser"]
            parsed, unknown = parser.parse_known_args(["--N", "5", "--infer-steps", "12"] if model == "hunyuan" else ["--N", "5"])
            self.assertEqual(parsed.skip_interval, 5)
            self.assertIsNone(parsed.replace_steps)
            self.assertEqual(unknown, ["--infer-steps", "12"] if model == "hunyuan" else [])

    def test_wan_inference_parser(self):
        tree = ast.parse((ROOT / "wan/infer/online_gen.py").read_text())
        functions = [copy.deepcopy(node) for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in ("_parse_args", "_validate_args")]
        namespace = {
            "argparse": argparse, "add_interval_argument": add_interval_argument,
            "resolve_inference_schedule": resolve_inference_schedule,
            "WAN_CONFIGS": {"t2v-1.3B": None, "t2v-14B": None},
            "SIZE_CONFIGS": {"832*480": None},
            "EXAMPLE_PROMPT": {"t2v-1.3B": None},
            "SUPPORTED_SIZES": {"t2v-1.3B": ["832*480"]},
            "str2bool": lambda value: value.lower() == "true",
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<wan-parser>", "exec"), namespace)
        base = ["infer", "--task", "t2v-1.3B", "--size", "832*480", "--ckpt_dir", "unused", "--base_seed", "42"]
        previous = sys.argv
        try:
            sys.argv = base + ["--N", "5"]
            parsed = namespace["_parse_args"]()
            self.assertEqual((parsed.skip_interval, parsed.sample_steps, parsed.lora_rank, parsed.lora_alpha), (5, 50, 32, 64))
            sys.argv = base
            legacy = namespace["_parse_args"]()
            self.assertEqual((legacy.lora_rank, legacy.lora_alpha), (4, 8))
            for options in [["--replace_steps", "4"], ["--replace_start", "0"], ["--use_pred_first"]]:
                sys.argv = base + ["--N", "5"] + options
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    namespace["_parse_args"]()
        finally:
            sys.argv = previous

    def test_video_batch_forwards_n_without_legacy_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prompt = directory / "prompts.txt"
            prompt.write_text("A test prompt.\n")
            capture = directory / "arguments.txt"
            for executable in ["python", "python3"]:
                stub = directory / executable
                stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CAPTURE_FILE"\n')
                stub.chmod(0o755)
            for model in ["hunyuan", "wan"]:
                for mode, interval in [("online", None), ("online", "5"), ("baseline", "5")]:
                    environment = {
                        **os.environ, "PATH": str(directory) + os.pathsep + os.environ.get("PATH", ""),
                        "PROMPT_FILE": str(prompt), "CAPTURE_FILE": str(capture),
                        "MODEL_BASE": str(directory), "CKPT_DIR": str(directory), "LORA_DIR": str(directory),
                        "OUTPUT_DIR": str(directory / "videos"), "NUM_STEPS": "12",
                    }
                    environment.pop("N", None)
                    if interval is not None:
                        environment["N"] = interval
                    capture.write_text("")
                    subprocess.run(
                        ["bash", str(ROOT / model / "infer/batch_infer.sh"), mode],
                        env=environment, check=True, capture_output=True, text=True,
                    )
                    tokens = capture.read_text().splitlines()
                    total_flag = "--infer-steps" if model == "hunyuan" else "--sample_steps"
                    manual_flag = "--replace-steps" if model == "hunyuan" else "--replace_steps"
                    self.assertEqual(tokens[tokens.index(total_flag) + 1], "12")
                    if interval is not None and mode == "online":
                        self.assertEqual(tokens[tokens.index("--N") + 1], "5")
                        self.assertNotIn(manual_flag, tokens)
                    else:
                        self.assertNotIn("--N", tokens)
                        if mode == "online":
                            self.assertIn(manual_flag, tokens)


if __name__ == "__main__":
    unittest.main()
