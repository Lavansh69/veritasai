"""
VeritasAI – Audio Deepfake Dataset
PyTorch Dataset for loading audio files and converting them to Mel spectrograms.
Completely separate from the image/video dataset files.

Expected directory structure:
    dataset_root/
    ├── real/
    │   ├── audio_001.wav
    │   ├── audio_002.mp3
    │   └── ...
    └── fake/
        ├── deepfake_001.wav
        ├── deepfake_002.mp3
        └── ...
"""

import os
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class AudioDeepfakeDataset(Dataset):
    """Loads audio files from real/ and fake/ subdirectories,
    converts to Mel spectrograms, and returns (spectrogram, label) pairs.

    Args:
        root_dir:    Path to dataset root (must contain real/ and fake/ subdirs)
        sample_rate: Target sample rate for loading audio
        duration:    Duration in seconds to pad/truncate to
        n_mels:      Number of Mel bands
        augment:     Whether to apply data augmentation (noise, shift, etc.)
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        duration: int = 5,
        n_mels: int = 128,
        augment: bool = False,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.augment = augment
        self.target_length = sample_rate * duration

        self.samples: list[tuple[str, int]] = []  # (filepath, label)

        root = Path(root_dir)
        for label_name, label_value in [("real", 0), ("fake", 1)]:
            label_dir = root / label_name
            if not label_dir.exists():
                print(f"Warning: {label_dir} not found — skipping")
                continue
            for fpath in sorted(label_dir.iterdir()):
                if fpath.suffix.lower() in AUDIO_EXTENSIONS:
                    self.samples.append((str(fpath), label_value))

        print(f"AudioDeepfakeDataset: loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_process(self, audio_path: str) -> np.ndarray:
        """Load audio → pad/truncate → Mel spectrogram → normalize."""
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Pad or truncate
        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)), mode="constant")
        else:
            if self.augment:
                # Random crop during training
                max_start = len(y) - self.target_length
                start = np.random.randint(0, max_start + 1)
                y = y[start : start + self.target_length]
            else:
                y = y[: self.target_length]

        # Data augmentation
        if self.augment:
            y = self._augment(y)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_mels=self.n_mels, fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        return mel_db.astype(np.float32)

    def _augment(self, y: np.ndarray) -> np.ndarray:
        """Apply simple audio augmentations."""
        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise

        # Random volume change
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            y = y * gain

        # Random time shift (small)
        if np.random.random() < 0.2:
            shift = np.random.randint(-1600, 1600)  # ±0.1s at 16kHz
            y = np.roll(y, shift)

        return y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filepath, label = self.samples[idx]

        try:
            mel = self._load_and_process(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}, returning zeros")
            mel = np.zeros((self.n_mels, self.target_length // 512 + 1), dtype=np.float32)

        # Shape: (1, n_mels, time_frames) — 1 channel
        spectrogram = torch.from_numpy(mel).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return spectrogram, label_tensor


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_dataset.py <dataset_root>")
        print("  dataset_root should contain real/ and fake/ subdirectories with audio files")
        sys.exit(1)

    ds = AudioDeepfakeDataset(sys.argv[1], augment=True)
    if len(ds) > 0:
        spec, lab = ds[0]
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Label: {lab.item()}")
    else:
        print("No samples found!")
