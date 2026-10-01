"""Shared periodic cache schedule using zero-based sampling-step indices."""

from dataclasses import dataclass
import json
from pathlib import Path
import warnings


BLOCK_INDICES = {"flux": 37, "hunyuan": 39, "wan": 29}


@dataclass(frozen=True)
class CacheSchedule:
    num_steps: int
    interval: int

    def __post_init__(self):
        for name, value in (("num_steps", self.num_steps), ("N", self.interval)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def full_steps(self):
        return list(range(0, self.num_steps, self.interval))

    @property
    def replace_steps(self):
        return [step for step in range(self.num_steps) if step % self.interval]

    def source_step(self, step):
        if type(step) is not int or not 0 <= step < self.num_steps:
            raise ValueError(f"Step {step} is outside [0, {self.num_steps}).")
        return step - step % self.interval

    def offset(self, step):
        return step - self.source_step(step)

    def identity(self, model):
        if model not in BLOCK_INDICES:
            raise ValueError(f"Unknown backend: {model}")
        return {
            "version": 1, "model": model, "N": self.interval,
            "num_steps": self.num_steps, "indexing": "zero_based_sampling_steps",
        }

    def manifest(self, model):
        return {
            **self.identity(model),
            "full_steps": self.full_steps,
            "replace_steps": self.replace_steps,
            "training_pairs": [
                {"step": step, "source_step": self.source_step(step), "offset": self.offset(step)}
                for step in self.replace_steps
            ],
        }


def add_interval_argument(parser):
    parser.add_argument(
        "--N", "--skip-interval", dest="skip_interval", type=int,
        help="Full computation every N sampling steps; intermediate steps use trained calibration.",
    )


def resolve_inference_schedule(interval, num_steps, manual_steps=None):
    if interval is None:
        return None
    if manual_steps is not None:
        raise ValueError("Do not combine --N with a manual replacement-step list.")
    return CacheSchedule(num_steps, interval)


def resolve_training_interval(args, offset_name="offset", legacy_offset=1):
    interval = getattr(args, "skip_interval", None)
    explicit_offset = getattr(args, offset_name)
    if interval is None:
        if explicit_offset is None:
            if legacy_offset is None:
                raise ValueError("Supply --n for legacy training or --N for periodic training.")
            setattr(args, offset_name, legacy_offset)
        return
    schedule = CacheSchedule(args.num_steps, interval)
    offsets = {schedule.offset(step) for step in args.steps}
    if not args.steps or 0 in offsets:
        raise ValueError("--N training targets must be skipped steps, not full-compute steps.")
    if len(offsets) != 1:
        raise ValueError("These targets need different offsets; use this model's run.py train --N entry.")
    offset = offsets.pop()
    if explicit_offset is not None and explicit_offset != offset:
        raise ValueError(f"Supplied offset {explicit_offset} conflicts with --N; expected {offset}.")
    setattr(args, offset_name, offset)


def training_schedule_metadata(args, model):
    interval = getattr(args, "skip_interval", None)
    if interval is None:
        return {}
    schedule = CacheSchedule(args.num_steps, interval)
    return {
        "cache_schedule": schedule.identity(model),
        "source_steps": [schedule.source_step(step) for step in args.steps],
    }


def validate_interval_features(roots, schedule, model, steps=None, branches=None):
    steps = schedule.replace_steps if steps is None else steps
    branches = (0, 1) if branches is None else branches
    missing = []
    unverified = []
    block = BLOCK_INDICES[model]
    for root in map(Path, roots):
        metadata_path = root / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("steps") != schedule.num_steps:
                raise ValueError(f"{root}: cache step count differs from --num-steps={schedule.num_steps}.")
        else:
            unverified.append(str(root))
        for step in steps:
            source = schedule.source_step(step)
            if model == "flux":
                names = [
                    f"singleblock_{block - 1}_output_step_{source}.pt",
                    f"singleblock_{block}_output_step_{step}.pt",
                    f"temb_step_{step}.pt", f"image_rotary_emb_step_{step}.pt",
                ]
            elif model == "hunyuan":
                names = [
                    f"single_block{block - 1}_step_{source}.pt",
                    f"single_block{block}_step_{step}.pt", f"meta_step_{step}.pt",
                ]
            else:
                names = []
                for branch in branches:
                    names.extend([
                        f"block_{block - 1}_{branch}_output_step_{source}.pt",
                        f"block_{block}_{branch}_output_step_{step}.pt",
                    ])
                    branch_meta = f"meta_step_{step}_{branch}.pt"
                    names.append(branch_meta if (root / branch_meta).is_file() else f"meta_step_{step}.pt")
            missing.extend(str(root / name) for name in names if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing --N training features ({len(missing)}): {missing[:5]}. "
            "Extract a complete trajectory with the same total step count.",
        )
    if unverified:
        warnings.warn(
            f"{len(unverified)} legacy cache roots lack metadata.json; verify that they use "
            f"the same {schedule.num_steps}-step sampling trajectory before training.",
            stacklevel=2,
        )


def validate_training_features(args, model):
    if getattr(args, "skip_interval", None) is None:
        return
    if args.block_idx != BLOCK_INDICES[model]:
        raise ValueError(f"--N mode calibrates the original final block {BLOCK_INDICES[model]} for {model}.")
    if model == "wan" and args.task != "t2v-1.3B":
        raise ValueError("--N mode supports the Wan2.1-T2V-1.3B backend.")
    roots = args.feature_root if model == "wan" else args.feature_roots
    validate_interval_features(
        roots, CacheSchedule(args.num_steps, args.skip_interval), model, args.steps,
        [args.layer_idx] if model == "wan" else None,
    )


def adapter_filename(model, step, branch=None):
    block = BLOCK_INDICES[model]
    if model == "wan":
        if branch not in (0, 1):
            raise ValueError("Wan requires CFG branch 0 or 1.")
        return f"block{block}_step{step}_{branch}.pt"
    if branch is not None:
        raise ValueError("Only Wan uses separate CFG adapter files.")
    prefix = "blocks" if model == "flux" else "block"
    return f"{prefix}{block}_step{step}.pt"


def validate_checkpoint_schedule(payload, schedule, model, step, branch=None):
    if not isinstance(payload, dict):
        raise ValueError(f"Step {step}: expected an adapter checkpoint dictionary.")
    if payload.get("cache_schedule") != schedule.identity(model):
        raise ValueError(f"Step {step}: adapter belongs to a different or legacy schedule; retrain for this --N.")
    targets = payload.get("steps")
    if not isinstance(targets, list) or step not in targets:
        raise ValueError(f"Step {step}: adapter training targets do not match.")
    if payload.get("offset") != schedule.offset(step) or not schedule.offset(step):
        raise ValueError(f"Step {step}: adapter cache offset does not match the periodic schedule.")
    if payload.get("source_steps") != [schedule.source_step(target) for target in targets]:
        raise ValueError(f"Step {step}: adapter source-step metadata does not match.")
    if payload.get("block_idx") != BLOCK_INDICES[model]:
        raise ValueError(f"Step {step}: adapter belongs to the wrong block.")
    if model == "wan" and payload.get("layer_idx") != branch:
        raise ValueError(f"Step {step}: adapter belongs to the wrong CFG branch.")


def validate_interval_adapters(directory, schedule, model, loader=None, rank=None, alpha=None):
    if not schedule.replace_steps:
        return
    if not directory:
        raise ValueError("--N > 1 requires the matching trained adapter directory.")
    directory = Path(directory)
    expected = [
        (step, branch, directory / adapter_filename(model, step, branch))
        for step in schedule.replace_steps
        for branch in ((0, 1) if model == "wan" else (None,))
    ]
    missing = [str(path) for _, _, path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing --N adapters ({len(missing)}): {missing[:5]}")
    manifest_path = directory / "schedule.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != schedule.manifest(model):
        raise ValueError("Adapter directory schedule.json does not match the requested model, N, and step count.")
    if loader is None:
        import torch

        def loader(path):
            return torch.load(path, map_location="cpu", weights_only=True)

    for step, branch, path in expected:
        payload = loader(path)
        validate_checkpoint_schedule(payload, schedule, model, step, branch)
        if rank is not None and payload.get("rank") != rank:
            raise ValueError(f"Step {step}: requested LoRA rank differs from the trained adapter.")
        if alpha is not None and payload.get("alpha") != alpha:
            raise ValueError(f"Step {step}: requested LoRA alpha differs from the trained adapter.")


def validate_cached_source(schedule, step, source_step):
    if schedule is not None and source_step != schedule.source_step(step):
        raise ValueError(
            f"Step {step}: cached source {source_step} differs from scheduled source {schedule.source_step(step)}."
        )
