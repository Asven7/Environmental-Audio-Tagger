from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


EPS = 1e-8


def load_audio(path: str | Path, target_sample_rate: int) -> np.ndarray:
    """Load an audio file as mono float32 and resample when necessary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_sample_rate:
        g = math.gcd(int(sample_rate), int(target_sample_rate))
        waveform = resample_poly(
            waveform,
            up=int(target_sample_rate) // g,
            down=int(sample_rate) // g,
        ).astype(np.float32, copy=False)
    return np.asarray(waveform, dtype=np.float32)


def save_audio(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(waveform, dtype=np.float32), sample_rate)


def rms(waveform: np.ndarray) -> float:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)) + EPS))


def rms_dbfs(waveform: np.ndarray) -> float:
    value = rms(waveform)
    if value <= EPS:
        return -120.0
    return float(20.0 * np.log10(value))


def normalize_rms(
    waveform: np.ndarray,
    target_dbfs: float = -20.0,
    max_gain_db: float = 12.0,
    silence_dbfs: float = -60.0,
) -> np.ndarray:
    """RMS-normalize with bounded gain; silence is left unchanged.

    The same operation is used for training and inference to reduce train/serve skew.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    current_dbfs = rms_dbfs(waveform)
    if current_dbfs <= silence_dbfs:
        return waveform.copy()

    gain_db = min(target_dbfs - current_dbfs, max_gain_db)
    gain = float(10.0 ** (gain_db / 20.0))
    normalized = waveform * gain
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return normalized.astype(np.float32, copy=False)


def pad_or_crop(
    waveform: np.ndarray,
    length: int,
    strategy: Literal["center", "start", "energy"] = "center",
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if length <= 0:
        raise ValueError("length must be positive")
    if waveform.size == length:
        return waveform.copy()
    if waveform.size < length:
        result = np.zeros(length, dtype=np.float32)
        result[: waveform.size] = waveform
        return result

    if strategy == "start":
        start = 0
    elif strategy == "center":
        start = (waveform.size - length) // 2
    elif strategy == "energy":
        # Search candidate windows with a reasonably dense stride while keeping cost low.
        stride = max(1, length // 8)
        best_start = 0
        best_energy = -1.0
        final_start = waveform.size - length
        candidates = list(range(0, final_start + 1, stride))
        if candidates[-1] != final_start:
            candidates.append(final_start)
        for candidate in candidates:
            chunk = waveform[candidate : candidate + length]
            energy = float(np.mean(np.square(chunk, dtype=np.float64)))
            if energy > best_energy:
                best_energy = energy
                best_start = candidate
        start = best_start
    else:
        raise ValueError(f"Unsupported crop strategy: {strategy}")
    return waveform[start : start + length].astype(np.float32, copy=True)


def apply_global_time_shift(waveform: np.ndarray, shift_samples: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if shift_samples == 0:
        return waveform.copy()
    result = np.zeros_like(waveform)
    if shift_samples > 0:
        result[shift_samples:] = waveform[:-shift_samples]
    else:
        amount = abs(shift_samples)
        result[:-amount] = waveform[amount:]
    return result


def mix_two_sources(
    first: np.ndarray,
    second: np.ndarray,
    relative_db: float,
    overlap_ratio: float,
    target_dbfs: float = -20.0,
) -> np.ndarray:
    """Create a fixed-length two-source mixture.

    The first source occupies the full window. The second source is shifted so that
    `overlap_ratio` of its duration remains inside the output window. Positive
    relative_db means the first source is louder than the second source.
    """
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("Both sources must have identical shape")
    if not 0 < overlap_ratio <= 1:
        raise ValueError("overlap_ratio must be in (0, 1]")

    # Mixture synthesis needs a controlled relative level, so allow a wider
    # normalization gain range than live inference. Silence is still protected
    # by normalize_rms' silence gate.
    a = normalize_rms(first, target_dbfs=target_dbfs, max_gain_db=40.0)
    b = normalize_rms(second, target_dbfs=target_dbfs, max_gain_db=40.0)

    # Symmetric gain assignment keeps the mixture centered around the target level.
    a_gain = float(10.0 ** ((relative_db / 2.0) / 20.0))
    b_gain = float(10.0 ** ((-relative_db / 2.0) / 20.0))
    a = a * a_gain
    b = b * b_gain

    length = first.size
    shift = int(round(length * (1.0 - overlap_ratio)))
    shifted_b = np.zeros(length, dtype=np.float32)
    shifted_b[shift:] = b[: length - shift]

    mixed = a + shifted_b
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    mixed = normalize_rms(mixed, target_dbfs=target_dbfs)
    return mixed.astype(np.float32, copy=False)
