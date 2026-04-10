"""
VeritasAI – Advanced Dataset with Albumentations
Backwards-compatible: the original dataset.py is NOT modified.

This provides an AdvancedDeepfakeDataset class with aggressive,
real-world augmentations designed to make the model robust against
social-media compression, noise, and other common transformations.

Usage in train_v2.py:
    from advanced_dataset import AdvancedDeepfakeDataset
    train_ds = AdvancedDeepfakeDataset(data_dir, image_size=384, split="train")
"""

import os
import random
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AdvancedDeepfakeDataset(Dataset):
    """Binary classification dataset with aggressive augmentations.

    Supports the same two folder layouts as the original DeepfakeDataset:
      Layout A (flat):       root_dir/real/ + root_dir/fake/
      Layout B (pre-split):  root_dir/{train,val,test}/{real,fake}/
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        image_size: int = 384,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment: bool = True,
    ):
        self.image_size = image_size
        self.split = split

        root = Path(root_dir)

        # ── Detect layout (identical logic to original dataset.py) ──
        split_dir = root / split
        flat_real = root / "real"
        flat_fake = root / "fake"

        if split_dir.exists() and (split_dir / "real").exists():
            real_files = sorted(self._glob_images(split_dir / "real"))
            fake_files = sorted(self._glob_images(split_dir / "fake"))
            if not real_files:
                real_files = sorted(self._glob_images(split_dir / "Real"))
            if not fake_files:
                fake_files = sorted(self._glob_images(split_dir / "Fake"))

            self.samples = (
                [(str(f), 0) for f in real_files]
                + [(str(f), 1) for f in fake_files]
            )
            random.seed(42)
            random.shuffle(self.samples)

        elif flat_real.exists() or flat_fake.exists():
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
            children = [d.name for d in root.iterdir()] if root.exists() else []
            raise FileNotFoundError(
                f"Cannot find images under '{root_dir}'.\n"
                f"Expected either '{root}/real/' + '{root}/fake/' (flat) or "
                f"'{root}/{split}/real/' + '{root}/{split}/fake/' (pre-split).\n"
                f"Found children: {children}"
            )

        # ── Albumentations Transforms ──────────────────────────────
        if augment and split == "train":
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1,
                    rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5,
                ),

                # ── Real-world degradation augmentations ───────────
                A.OneOf([
                    A.ImageCompression(
                        quality_lower=20, quality_upper=70,
                        compression_type=A.ImageCompression.ImageCompressionType.JPEG,
                        p=1.0,
                    ),
                    A.Downscale(
                        scale_min=0.25, scale_max=0.5,
                        interpolation=cv2.INTER_LINEAR, p=1.0,
                    ),
                ], p=0.5),

                # ── Blur variants ──────────────────────────────────
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.3),

                # ── Noise injection ────────────────────────────────
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.3),

                # ── Color / lighting jitter ────────────────────────
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=20,
                        val_shift_limit=20, p=1.0,
                    ),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ], p=0.4),

                # ── Cutout / Random Erasing ────────────────────────
                A.CoarseDropout(
                    max_holes=8, max_height=image_size // 8,
                    max_width=image_size // 8,
                    min_holes=1, min_height=image_size // 16,
                    min_width=image_size // 16,
                    fill_value=0, p=0.3,
                ),

                # ── Normalize to ImageNet stats + convert ──────────
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
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
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not read {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        augmented = self.transform(image=img)
        tensor = augmented["image"]
        return tensor, torch.tensor(label, dtype=torch.float32)
