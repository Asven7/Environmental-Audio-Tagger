from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audio import load_audio, normalize_rms, pad_or_crop
from .inference import AudioTagger


def benchmark_manifest(
    tagger: AudioTagger,
    manifest_path: str | Path,
    audio_root: str | Path,
    max_samples: int = 100,
    warmup: int = 5,
) -> dict:
    frame = pd.read_csv(manifest_path)
    singles = frame[frame["sample_type"].astype(str) == "single"].head(max_samples)
    if singles.empty:
        raise ValueError("Runtime benchmark requires at least one single sample")

    waveforms: list[np.ndarray] = []
    for row in singles.itertuples(index=False):
        waveform = load_audio(Path(audio_root) / row.source_a, tagger.sample_rate)
        waveform = pad_or_crop(waveform, tagger.window_samples, strategy="center")
        waveforms.append(waveform)

    for waveform in waveforms[: min(warmup, len(waveforms))]:
        tagger.predict_waveform(waveform)

    records = [tagger.predict_waveform(w).as_dict() for w in waveforms]
    feature = np.asarray([r["feature_ms"] for r in records], dtype=float)
    inference = np.asarray([r["inference_ms"] for r in records], dtype=float)
    total = np.asarray([r["total_ms"] for r in records], dtype=float)
    return {
        "n_samples": int(len(records)),
        "feature_ms_mean": float(feature.mean()),
        "inference_ms_mean": float(inference.mean()),
        "total_ms_mean": float(total.mean()),
        "total_ms_median": float(np.median(total)),
        "total_ms_p95": float(np.percentile(total, 95)),
        "stream_hop_ms": None,
    }
