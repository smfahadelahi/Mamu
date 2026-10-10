"""Random-window dataset over uint16 .bin shards (memory-mapped, no RAM copy)."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenShards(Dataset):
    """Each item is a random window of seq_len + 1 tokens (input and shifted target)."""

    def __init__(self, shard_dir: str, split: str, seq_len: int, samples: int, seed: int = 0):
        paths = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.bin")))
        if not paths:
            raise FileNotFoundError(f"no {split} shards in {shard_dir}")
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.lengths = np.array([len(s) for s in self.shards])
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.total = int(self.lengths.sum())
        self.seq_len = seq_len
        self.samples = samples
        self.seed = seed
        if self.total <= seq_len + 1:
            raise ValueError("not enough tokens for one window")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        pos = int(rng.integers(0, self.total - self.seq_len - 1))
        shard = int(np.searchsorted(self.offsets, pos, side="right") - 1)
        start = pos - int(self.offsets[shard])
        s = self.shards[shard]
        # Windows that would run past the end of a shard are shifted back inside it.
        if start + self.seq_len + 1 > len(s):
            start = max(0, len(s) - self.seq_len - 1)
        chunk = np.asarray(s[start:start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk)
