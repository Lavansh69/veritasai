"""
VeritasAI – Dataset & Preprocessing
PyTorch Dataset supporting FaceForensics++, DFDC, Celeb-DF folder structures.

Supports two folder layouts:
  1. FLAT – root_dir/real/ + root_dir/fake/  (auto-split by ratio)
  2. PRE-SPLIT – root_dir/{train,val,test}/{real,fake}/  (use existing splits)
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """Binary classification dataset: real (0) vs fake (1).

    Folder layout A (flat – will be split automatically):
        root_dir/
            real/       ← authentic images
            fake/       ← deepfake images

    Folder layout B (pre-split – used as-is):
        root_dir/
            train/
                real/
                fake/
            val/
                real/
                fake/
            test/
                real/
                fake/
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment: bool = True,
    ):
        self.image_size = image_size
        self.split = split

        root = Path(root_dir)

        # ── Detect layout ─────────────────────────────────────────
        split_dir = root / split          # e.g. root/train
        flat_real = root / "real"
        flat_fake = root / "fake"

        if split_dir.exists() and (split_dir / "real").exists():
            # Layout B: pre-split folders
            real_dir = split_dir / "real"
            fake_dir = split_dir / "fake"
            real_files = sorted(self._glob_images(real_dir))
            fake_files = sorted(self._glob_images(fake_dir))
            # Also check for capitalised variants (e.g. "Real", "Fake")
            if not real_files:
                real_dir_cap = split_dir / "Real"
                real_files = sorted(self._glob_images(real_dir_cap))
            if not fake_files:
                fake_dir_cap = split_dir / "Fake"
                fake_files = sorted(self._glob_images(fake_dir_cap))

            self.samples = (
                [(str(f), 0) for f in real_files]
                + [(str(f), 1) for f in fake_files]
            )
            random.seed(42)
            random.shuffle(self.samples)

        elif flat_real.exists() or flat_fake.exists():
            # Layout A: flat real/ + fake/ → split by ratio
            real_files = sorted(self._glob_images(flat_real))
            fake_files = sorted(self._glob_images(flat_fake))

            samples = (
                [(str(f), 0) for f in real_files]
                + [(str(f), 1) for f in fake_files]
            )
            random.seed(42)
            random.shuffle(samples)

            n = len(samples)
            t = int(n * train_ratio)
            v = int(n * (train_ratio + val_ratio))

            if split == "train":
                self.samples = samples[:t]
            elif split == "val":
                self.samples = samples[t:v]
            else:
                self.samples = samples[v:]
        else:
            # Nothing matched – try to be helpful
            children = [d.name for d in root.iterdir()] if root.exists() else []
            raise FileNotFoundError(
                f"Cannot find images under '{root_dir}'.\n"
                f"Expected either '{root}/real/' + '{root}/fake/' (flat) or "
                f"'{root}/{split}/real/' + '{root}/{split}/fake/' (pre-split).\n"
                f"Found children: {children}"
            )

        # ── Transforms ────────────────────────────────────────────
        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def _glob_images(directory: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        files = []
        if directory.exists():
            for f in directory.rglob("*"):
                if f.suffix.lower() in exts:
                    files.append(f)
        return files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a blank image on error
            img = Image.new("RGB", (self.image_size, self.image_size))

        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32)
