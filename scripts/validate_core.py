"""Fast CPU checks for the reusable calibration utilities; no model download."""

import copy
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learnibridge.checkpoints import load_video_adapter
from learnibridge.cli_compat import parse_native_arguments
from learnibridge.lora import adapter_state_dict, freeze_except_lora, inject_lora, iter_lora_parameters, set_lora_enabled
from learnibridge.wan_data import FeatureBlockDataset
from learnibridge.schedule import (
    BLOCK_INDICES, CacheSchedule, adapter_filename, training_schedule_metadata, validate_interval_adapters,
)


def main():
    native_parser = argparse.ArgumentParser()
    native_parser.add_argument("--value", type=int)
    previous_arguments = sys.argv
    def namespace_only(namespace=None):
        return native_parser.parse_args(namespace=namespace)
    assert parse_native_arguments(namespace_only, ["--value", "7"]).value == 7
    assert sys.argv is previous_arguments
    assert parse_native_arguments(native_parser.parse_args, ["--value", "9"]).value == 9
    torch.manual_seed(42)
    torch.set_num_threads(1)
    baseline = nn.Sequential(nn.Linear(16, 16)).eval()
    adapted = copy.deepcopy(baseline)
    inject_lora(adapted, ["0"], rank=4, alpha=8)
    freeze_except_lora(adapted)
    trainable = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    assert trainable == ["0.lora_a", "0.lora_b"]
    inputs = torch.randn(2, 8, 16)
    with torch.no_grad():
        expected = baseline(inputs)
        torch.testing.assert_close(adapted(inputs), expected, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(iter_lora_parameters(adapted), lr=0.02, weight_decay=0)
    for iteration in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(adapted(inputs), expected + 0.2)
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(adapted[0].base.weight, baseline[0].weight, rtol=0, atol=0)
    assert adapted[0].base.weight.grad is None
    state = adapter_state_dict(adapted)
    state["0.base.weight"] = torch.zeros_like(baseline[0].weight)
    payload = {"state_dict": state, "rank": 4, "alpha": 8, "base_frozen": True,
               "target_modules": ["0"], "steps": [1], "offset": 1}
    restored = copy.deepcopy(baseline)
    load_video_adapter(restored, payload)
    torch.testing.assert_close(restored[0].base.weight, baseline[0].weight, rtol=0, atol=0)
    torch.testing.assert_close(restored(inputs), adapted(inputs), rtol=0, atol=0)
    set_lora_enabled(restored, False)
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    rejected = False
    try:
        load_video_adapter(copy.deepcopy(baseline), {"state_dict": state})
    except ValueError:
        rejected = True
    assert rejected
    with tempfile.TemporaryDirectory() as temporary:
        roots = []
        for index in range(2):
            root = Path(temporary) / str(index)
            root.mkdir()
            torch.save(torch.tensor([index], dtype=torch.float32), root / "block_0_0_output_step_0.pt")
            torch.save(torch.tensor([index + 1], dtype=torch.float32), root / "block_1_0_output_step_1.pt")
            torch.save({"e": torch.ones(1)}, root / "meta_step_1_0.pt")
            roots.append(root)
        dataset = FeatureBlockDataset(roots, 1, 0, 1, [1], device="cpu")
        assert len(dataset) == 2
        assert {sample["x_input"].item() for sample in dataset} == {0.0, 1.0}
    with tempfile.TemporaryDirectory() as temporary:
        schedule = CacheSchedule(6, 3)
        for model in BLOCK_INDICES:
            directory = Path(temporary) / model
            directory.mkdir()
            (directory / "schedule.json").write_text(json.dumps(schedule.manifest(model)))
            for step in schedule.replace_steps:
                arguments = SimpleNamespace(skip_interval=3, num_steps=6, steps=[step])
                for branch in ([0, 1] if model == "wan" else [None]):
                    torch.save({
                        **training_schedule_metadata(arguments, model),
                        "state_dict": adapter_state_dict(adapted),
                        "block_idx": BLOCK_INDICES[model], "steps": [step],
                        "offset": schedule.offset(step), "layer_idx": branch,
                        "rank": 4, "alpha": 8,
                    }, directory / adapter_filename(model, step, branch))
            validate_interval_adapters(directory, schedule, model, rank=4, alpha=8)
    print(json.dumps({"passed": True, "device": "cpu", "base_frozen": True,
                      "base_checkpoint_injection_ignored": True, "checkpoint_roundtrip": True,
                      "missing_metadata_rejected": True, "all_feature_roots_used": True,
                      "native_cli_compatibility": True, "periodic_checkpoint_serialization": True}))


if __name__ == "__main__":
    main()
