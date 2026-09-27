from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import chirp

from .audio import save_audio
from .manifests import MANIFEST_COLUMNS, generate_mixture_rows, write_manifests


def _tone_pattern(class_index: int, duration: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Generate distinct synthetic signatures for engineering smoke tests only."""
    n = round(duration * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    base_freqs = [180, 260, 360, 480, 620, 790, 980, 1250, 1500, 1900]
    freq = base_freqs[class_index % len(base_freqs)]
    kind = class_index % 5
    if kind == 0:
        signal = np.sin(2 * np.pi * freq * t)
    elif kind == 1:
        signal = 0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * (freq * 1.5) * t)
    elif kind == 2:
        envelope = (np.sin(2 * np.pi * 3.0 * t) > 0).astype(np.float32)
        signal = envelope * np.sin(2 * np.pi * freq * t)
    elif kind == 3:
        signal = chirp(t, f0=freq * 0.6, f1=freq * 1.4, t1=duration, method="linear")
    else:
        carrier = np.sin(2 * np.pi * freq * t)
        signal = carrier * (0.5 + 0.5 * np.sin(2 * np.pi * 5.0 * t))
    noise = rng.normal(0.0, 0.015, size=n)
    signal = 0.25 * np.asarray(signal) + noise
    return signal.astype(np.float32)


def generate_demo_dataset(root: str | Path, config: dict, samples_per_class_per_split: dict[str, int] | None = None) -> dict:
    root = Path(root)
    audio_dir = root / "audio"
    manifests_dir = root / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = int(config["project"]["sample_rate"])
    duration = float(config["project"]["window_seconds"])
    known_classes = list(config["project"]["target_classes"])
    heldout_classes = list(config["project"]["heldout_classes"])
    counts = samples_per_class_per_split or {"train": 8, "val": 3, "test": 3}
    rng = np.random.default_rng(int(config["training"].get("data_seed", 1234)))

    known_rows: list[dict] = []
    ood_rows: list[dict] = []
    all_classes = known_classes + heldout_classes
    class_to_index = {name: i for i, name in enumerate(known_classes)}
    for class_index, class_name in enumerate(all_classes):
        is_ood = class_name in heldout_classes
        for split in ("train", "val", "test"):
            if is_ood and split == "train":
                continue
            for sample_index in range(int(counts[split])):
                local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                waveform = _tone_pattern(class_index, duration, sample_rate, local_rng)
                # Add small deterministic phase/level variation.
                waveform *= float(local_rng.uniform(0.8, 1.2))
                relative = Path("audio") / split / class_name / f"{sample_index:03d}.wav"
                save_audio(root / relative, waveform, sample_rate)
                row = {
                    "sample_id": f"demo-{split}-{class_name}-{sample_index:03d}",
                    "split": split,
                    "sample_type": "single",
                    "source_a": relative.as_posix(),
                    "source_b": "",
                    "label_indices": json.dumps([] if is_ood else [class_to_index[class_name]]),
                    "label_names": json.dumps([class_name]),
                    "relative_db": np.nan,
                    "overlap_ratio": 0.0,
                    "is_ood": bool(is_ood),
                }
                (ood_rows if is_ood else known_rows).append(row)

    known = pd.DataFrame(known_rows, columns=MANIFEST_COLUMNS)
    ood = pd.DataFrame(ood_rows, columns=MANIFEST_COLUMNS)
    paths = write_manifests(known, ood, manifests_dir, config)
    return {key: str(value) for key, value in paths.items()}
