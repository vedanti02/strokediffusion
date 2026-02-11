"""Data loader for paired (stroke .pt, target image) training data.

Provides:
- StrokeImageDataset:    loads (strokes, target_image) pairs
- get_paired_dataloader: returns DataLoader yielding (strokes, images)

Design goals:
- Fail-fast with clear error messages when no files found or shapes mismatch.
- Defensive about value ranges: convert [0,1] → [-1,1] for training.
"""
from __future__ import annotations

import os
import glob
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class StrokeImageDataset(Dataset):
    """Dataset returning (strokes, target_image) pairs.

    Pairs each stroke .pt file with its source image.  The .pt filename
    is *{img_name}.pt* (e.g. ``000001.pt``) and the corresponding image is
    ``{img_dir}/{img_name}.jpg`` (or ``.png``).

    Returns:
        strokes:      (num_strokes, stroke_dim) float32, values in [-1, 1]
        target_image: (3, canvas_size, canvas_size) float32, values in [0, 1]
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        stroke_dir: str,
        image_dir: str,
        num_strokes: int = 100,
        stroke_dim: int = 13,
        canvas_size: int = 128,
    ):
        self.stroke_dir = os.path.expanduser(stroke_dir)
        self.image_dir = os.path.expanduser(image_dir)
        self.num_strokes = num_strokes
        self.stroke_dim = stroke_dim
        self.canvas_size = canvas_size

        self.transform = transforms.Compose([
            transforms.Resize((canvas_size, canvas_size)),
            transforms.ToTensor(),  # -> [0, 1]
        ])

        # Find .pt files that have a matching image
        pt_files = sorted(glob.glob(os.path.join(self.stroke_dir, "*.pt")))
        self.pairs: List[Tuple[str, str]] = []
        for pt_path in pt_files:
            stem = os.path.splitext(os.path.basename(pt_path))[0]
            img_path = self._find_image(stem)
            if img_path is not None:
                self.pairs.append((pt_path, img_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matching (stroke, image) pairs found.\n"
                f"  stroke_dir: {self.stroke_dir}\n"
                f"  image_dir:  {self.image_dir}\n"
                f"Ensure .pt filenames match image filenames (e.g. 000001.pt ↔ 000001.jpg)."
            )

    def _find_image(self, stem: str) -> Optional[str]:
        """Look for an image file matching the given stem in *image_dir*."""
        for ext in self.IMG_EXTS:
            candidate = os.path.join(self.image_dir, stem + ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_strokes(self, path: str) -> torch.Tensor:
        """Load and normalise strokes from a .pt file."""
        x = torch.load(path, weights_only=True)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.ndim == 2 and x.size(1) == 65:
            x = x.view(-1, 13)
        else:
            try:
                x = x.view(-1, self.stroke_dim)
            except Exception:
                raise ValueError(
                    f"Unexpected shape for '{path}': {tuple(x.shape)}. "
                    f"Expected (T,65) or (N,{self.stroke_dim})."
                )
        if x.size(0) >= self.num_strokes:
            x = x[: self.num_strokes]
        else:
            x = torch.cat([x, torch.zeros(self.num_strokes - x.size(0), self.stroke_dim)], 0)

        x = x.to(torch.float32)
        # [0,1] → [-1,1]
        if x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 2.0 - 1.0
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pt_path, img_path = self.pairs[idx]
        strokes = self._load_strokes(pt_path)
        image = self.transform(Image.open(img_path).convert("RGB"))
        return strokes, image


def get_paired_dataloader(
    stroke_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    num_strokes: int = 100,
    stroke_dim: int = 13,
    canvas_size: int = 128,
) -> DataLoader:
    """Create a DataLoader returning ``(strokes, target_image)`` pairs.

    Args:
        stroke_dir: dir with .pt stroke files  (default from config)
        image_dir:  dir with source face images (default from config)
    """
    if stroke_dir is None:
        stroke_dir = os.path.expanduser("/home/vkshirsa/strokediffusion_outputs")
    if image_dir is None:
        # defer import to avoid circular dependency at module init
        from src.config import IMAGE_PATH
        image_dir = IMAGE_PATH

    dataset = StrokeImageDataset(
        stroke_dir, image_dir,
        num_strokes=num_strokes,
        stroke_dim=stroke_dim,
        canvas_size=canvas_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


__all__ = ["StrokeImageDataset", "get_paired_dataloader"]


if __name__ == "__main__":
    from src.config import DATA_PATH, IMAGE_PATH
    ds = StrokeImageDataset(DATA_PATH, IMAGE_PATH)
    print(f"Found {len(ds)} pairs; strokes: {ds[0][0].shape}, image: {ds[0][1].shape}")
