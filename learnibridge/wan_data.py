from pathlib import Path

import torch
from torch.utils.data import Dataset


class FeatureBlockDataset(Dataset):
    def __init__(self, roots, block_idx, layer_idx, n, steps, device="cuda", preload=True):
        self.roots = [Path(root) for root in roots]
        self.steps = list(steps)
        if not self.roots or not self.steps or n < 1 or any(step < n for step in self.steps):
            raise ValueError("Require nonempty roots/steps and 1 <= offset <= target step.")
        self.block_idx = block_idx
        self.layer_idx = layer_idx
        self.n = n
        self.device = torch.device(device)
        self.samples = [(root, step) for step in self.steps for root in self.roots]
        self.cache = {}
        if preload:
            self.cache = {index: self._load(root, step) for index, (root, step) in enumerate(self.samples)}

    def _load(self, root, step):
        source_path = root / f"block_{self.block_idx - 1}_{self.layer_idx}_output_step_{step - self.n}.pt"
        target_path = root / f"block_{self.block_idx}_{self.layer_idx}_output_step_{step}.pt"
        metadata_path = root / f"meta_step_{step}_{self.layer_idx}.pt"
        if not metadata_path.is_file():
            metadata_path = root / f"meta_step_{step}.pt"
        missing = [str(path) for path in (source_path, target_path, metadata_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing calibration data: {missing}")
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        metadata = {
            name: value.to(self.device) if torch.is_tensor(value) else value
            for name, value in metadata.items()
        }
        return {
            "step": step, "root": str(root),
            "x_input": torch.load(source_path, map_location=self.device, weights_only=True),
            "target": torch.load(target_path, map_location=self.device, weights_only=True),
            "meta": metadata,
        }

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        return self._load(*self.samples[index])

    def __len__(self):
        return len(self.samples)
