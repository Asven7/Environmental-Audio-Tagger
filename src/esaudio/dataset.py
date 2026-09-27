from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import (
    apply_global_time_shift,
    load_audio,
    mix_two_sources,
    normalize_rms,
    pad_or_crop,
)
from .config import window_samples
from .manifests import parse_json_list


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


class DeterministicAugmenter:
    """Small deterministic train-time augmentation keyed by sample and epoch."""

    def __init__(self, config: dict) -> None:
        aug = config.get("augmentation", {})
        self.enabled = bool(aug.get("enabled", False))
        self.gain_db = float(aug.get("gain_db", 3.0))
        self.max_shift_seconds = float(aug.get("max_shift_seconds", 0.1))
        self.noise_probability = float(aug.get("noise_probability", 0.25))
        self.noise_snr_db = float(aug.get("noise_snr_db", 25.0))
        self.sample_rate = int(config["project"]["sample_rate"])
        self.base_seed = int(config["training"].get("data_seed", 1234))

    def __call__(self, waveform: np.ndarray, sample_id: str, epoch: int) -> np.ndarray:
        if not self.enabled:
            return waveform
        rng = np.random.default_rng(_stable_seed(self.base_seed, sample_id, epoch))
        output = np.asarray(waveform, dtype=np.float32).copy()

        gain_db = float(rng.uniform(-self.gain_db, self.gain_db))
        output *= float(10.0 ** (gain_db / 20.0))

        max_shift = round(self.max_shift_seconds * self.sample_rate)
        if max_shift > 0:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            output = apply_global_time_shift(output, shift)

        if rng.random() < self.noise_probability:
            signal_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2) + 1e-8))
            if signal_rms > 1e-6:
                noise_rms = signal_rms / (10.0 ** (self.noise_snr_db / 20.0))
                noise = rng.normal(0.0, noise_rms, size=output.shape).astype(np.float32)
                output = output + noise

        peak = float(np.max(np.abs(output))) if output.size else 0.0
        if peak > 0.99:
            output = output * (0.99 / peak)
        return output.astype(np.float32, copy=False)


class AudioManifestDataset(Dataset):
    """Load single or synthetic mixed audio samples from a CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        audio_root: str | Path,
        config: dict,
        split: str,
        include_ood: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root)
        self.config = config
        self.split = split
        self.frame = pd.read_csv(self.manifest_path)
        if "split" in self.frame:
            self.frame = self.frame[self.frame["split"].astype(str) == split].reset_index(drop=True)
        if not include_ood and "is_ood" in self.frame:
            self.frame = self.frame[~self.frame["is_ood"].astype(bool)].reset_index(drop=True)
        self.sample_rate = int(config["project"]["sample_rate"])
        self.length = window_samples(config)
        self.num_classes = len(config["project"]["target_classes"])
        data_cfg = config.get("data", {})
        self.train_crop_strategy = str(data_cfg.get("train_crop_strategy", "energy"))
        self.eval_crop_strategy = str(data_cfg.get("eval_crop_strategy", "center"))
        self.target_dbfs = float(data_cfg.get("target_rms_dbfs", -20.0))
        self.augmenter = DeterministicAugmenter(config)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def _load_window(self, relative_path: str) -> np.ndarray:
        waveform = load_audio(self.audio_root / relative_path, self.sample_rate)
        strategy = self.train_crop_strategy if self.split == "train" else self.eval_crop_strategy
        return pad_or_crop(waveform, self.length, strategy=strategy)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | list[str]]:
        row = self.frame.iloc[int(index)]
        sample_type = str(row["sample_type"])
        first = self._load_window(str(row["source_a"]))
        if sample_type == "single":
            waveform = normalize_rms(first, target_dbfs=self.target_dbfs)
        elif sample_type == "mix":
            source_b = str(row["source_b"])
            if not source_b or source_b == "nan":
                raise ValueError(f"Mixture sample {row['sample_id']} is missing source_b")
            second = self._load_window(source_b)
            waveform = mix_two_sources(
                first,
                second,
                relative_db=float(row["relative_db"]),
                overlap_ratio=float(row["overlap_ratio"]),
                target_dbfs=self.target_dbfs,
            )
        else:
            raise ValueError(f"Unsupported sample_type={sample_type}")

        if self.split == "train":
            waveform = self.augmenter(waveform, str(row["sample_id"]), self.epoch)

        target = np.zeros(self.num_classes, dtype=np.float32)
        for label_index in parse_json_list(row["label_indices"]):
            target[int(label_index)] = 1.0

        return {
            "waveform": torch.from_numpy(waveform.astype(np.float32, copy=False)),
            "target": torch.from_numpy(target),
            "sample_id": str(row["sample_id"]),
            "sample_type": sample_type,
            "relative_db": float(row["relative_db"]) if pd.notna(row["relative_db"]) else float("nan"),
            "overlap_ratio": float(row["overlap_ratio"]) if pd.notna(row["overlap_ratio"]) else 0.0,
            "label_names_json": str(row["label_names"]),
        }

    def target_matrix(self) -> np.ndarray:
        result = np.zeros((len(self.frame), self.num_classes), dtype=np.float32)
        for row_idx, value in enumerate(self.frame["label_indices"].tolist()):
            for label_index in parse_json_list(value):
                result[row_idx, int(label_index)] = 1.0
        return result
