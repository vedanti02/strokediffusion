"""Data loader for stroke-parameter .pt files.

Default canonical storage is /home/vkshirsa/strokediffusion_outputs/ (previously
`baseline/output_pts`).

Provides:
- StrokePtDataset: simple torch.utils.data.Dataset that loads .pt files produced by
  `baseline/test.py` (shape (20,65) -> view(-1,13) -> (100,13)).
- get_dataloader: convenience wrapper that returns a DataLoader used by `train.py`.

Design goals:
- Fail-fast with clear error messages when no files found or shapes mismatch.
- Be defensive about value ranges (assume baseline saves values in [0,1]; convert to [-1,1]
  because training pipeline expects strokes in [-1,1]).
"""
from __future__ import annotations

import os
import glob
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class StrokePtDataset(Dataset):
    """Dataset that loads stroke-parameter .pt files saved by the Learning-to-Paint baseline.

    Each file is expected to contain a tensor of shape (T, 65) where T (default 20)
    should be reshaped to (T*5, 13) == (100, 13) before being returned.

    The dataset will:
    - torch.load(each_file)
    - view(-1, 13)
    - optionally truncate/pad to (num_strokes, stroke_dim)
    - convert to float32 and map from [0,1] -> [-1,1] if values appear in [0,1]
    """

    def __init__(
        self,
        root: str,
        num_strokes: int = 100,
        stroke_dim: int = 13,
        extensions: tuple = (".pt",),
    ):
        self.root = os.path.expanduser(root)
        self.num_strokes = num_strokes
        self.stroke_dim = stroke_dim
        pattern = os.path.join(self.root, "*")
        files = sorted([p for p in glob.glob(pattern) if os.path.splitext(p)[1] in extensions])
        if not files:
            raise FileNotFoundError(
                f"No stroke .pt files found in '{self.root}'. Expected files like 'xxx.pt'."
            )
        self.files: List[str] = files

    def __len__(self):
        return len(self.files)

    def _load_one(self, path: str) -> torch.Tensor:
        x = torch.load(path)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.ndim == 2 and x.size(1) == 65:
            x = x.view(-1, 13)
        elif x.ndim == 1 and x.numel() == self.num_strokes * self.stroke_dim:
            x = x.view(self.num_strokes, self.stroke_dim)
        elif x.ndim == 2 and x.size(1) == self.stroke_dim and x.size(0) == self.num_strokes:
            # already in expected shape
            pass
        else:
            # try to coerce if possible, otherwise raise
            try:
                x = x.view(-1, self.stroke_dim)
            except Exception:
                raise ValueError(
                    f"Unexpected stroke file shape for '{path}': {tuple(x.shape)}. "
                    f"Expected (T,65) or ({self.num_strokes},{self.stroke_dim})."
                )

        # Ensure correct number of strokes: truncate or pad with zeros
        if x.size(0) >= self.num_strokes:
            x = x[: self.num_strokes]
        else:
            pad = torch.zeros(self.num_strokes - x.size(0), self.stroke_dim, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        x = x.to(torch.float32)

        # If values appear to be in [0,1], convert to [-1,1] (training expects [-1,1])
        if x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 2.0 - 1.0

        return x

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        return self._load_one(path)


def get_dataloader(
    data_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    num_strokes: int = 100,
    stroke_dim: int = 13,
) -> DataLoader:
    """Create a DataLoader for stroke .pt files.

    Args:
        data_path: directory containing .pt files (defaults to `/home/vkshirsa/strokediffusion_outputs`)
    """
    if data_path is None:
        # use absolute canonical path to avoid importing src.config at module import time
        data_path = os.path.expanduser("/home/vkshirsa/strokediffusion_outputs")

    dataset = StrokePtDataset(data_path, num_strokes=num_strokes, stroke_dim=stroke_dim)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


__all__ = ["StrokePtDataset", "get_dataloader"]


if __name__ == "__main__":
    # quick local smoke test
    ds = StrokePtDataset("/home/vkshirsa/strokediffusion_outputs")
    print(f"Found {len(ds)} files; sample shape: {ds[0].shape}")
