from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


_FLOAT_EPS = np.finfo(np.float32).eps


def _as_mono_float32(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform)
    if array.ndim == 2:
        # SoundFile uses [samples, channels]. Averaging is deterministic and
        # prevents the amplitude doubling that summing channels would cause.
        array = array.mean(axis=1)
    elif array.ndim != 1:
        raise ValueError(f"Audio waveform must be 1-D mono or 2-D [samples, channels], got {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Audio waveform is empty")
    if not np.isfinite(array).all():
        raise ValueError("Audio waveform contains NaN or infinite values")
    return np.ascontiguousarray(array)


def resample_audio(waveform: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Resample a mono waveform using polyphase filtering."""
    waveform = _as_mono_float32(waveform)
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive integers")
    if source_sample_rate == target_sample_rate:
        return waveform.copy()

    divisor = math.gcd(int(source_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // divisor
    down = int(source_sample_rate) // divisor
    result = resample_poly(waveform, up, down)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Resampling produced non-finite values")
    return np.ascontiguousarray(result)


def load_audio(path: str | Path, target_sample_rate: int) -> np.ndarray:
    """Load an audio file, downmix to mono, and resample to the target rate."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    try:
        audio, source_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Could not read audio file '{path}': {exc}") from exc

    waveform = _as_mono_float32(audio)
    return resample_audio(waveform, int(source_sample_rate), int(target_sample_rate))


def rms(waveform: np.ndarray) -> float:
    waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform))))


def rms_dbfs(waveform: np.ndarray, floor_dbfs: float = -120.0) -> float:
    """Return RMS level in dBFS, with a finite floor for silence."""
    value = rms(waveform)
    if value <= 10 ** (floor_dbfs / 20.0):
        return float(floor_dbfs)
    return float(20.0 * np.log10(value))


def normalize_rms(
    waveform: np.ndarray,
    target_dbfs: float = -20.0,
    max_gain_db: float = 20.0,
    silence_threshold_dbfs: float = -60.0,
) -> np.ndarray:
    """Apply bounded RMS normalization without amplifying silence aggressively.

    A hard gain limit protects quiet/noisy clips from extreme amplification.
    Truly silent or very low-energy windows are returned unchanged.
    """
    waveform = _as_mono_float32(waveform)
    if max_gain_db < 0:
        raise ValueError("max_gain_db must be non-negative")
    current_dbfs = rms_dbfs(waveform)
    if current_dbfs <= silence_threshold_dbfs:
        return waveform.copy()

    gain_db = float(target_dbfs) - current_dbfs
    gain_db = float(np.clip(gain_db, -max_gain_db, max_gain_db))
    gain = 10.0 ** (gain_db / 20.0)
    normalized = waveform * np.float32(gain)
    return np.asarray(normalized, dtype=np.float32)


def apply_global_time_shift(waveform: np.ndarray, shift_samples: int) -> np.ndarray:
    """Shift a mono waveform without wrap-around while preserving its length.

    Positive values delay the signal and zero-fill the beginning. Negative values
    advance the signal and zero-fill the end. This behavior is appropriate for
    train-time augmentation because samples that move outside the window are
    intentionally discarded rather than wrapped to the opposite edge.
    """
    waveform = _as_mono_float32(waveform)
    if not isinstance(shift_samples, (int, np.integer)):
        raise TypeError("shift_samples must be an integer")

    shift = int(shift_samples)
    if shift == 0:
        return waveform.copy()

    result = np.zeros_like(waveform)
    n_samples = waveform.size
    if abs(shift) >= n_samples:
        return result

    if shift > 0:
        result[shift:] = waveform[: n_samples - shift]
    else:
        advance = -shift
        result[: n_samples - advance] = waveform[advance:]
    return result


def _energy_crop_start(waveform: np.ndarray, target_samples: int) -> int:
    squared = np.square(waveform.astype(np.float64, copy=False))
    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    energies = prefix[target_samples:] - prefix[:-target_samples]
    return int(np.argmax(energies))


def pad_or_crop(waveform: np.ndarray, target_samples: int, strategy: str = "center") -> np.ndarray:
    """Return exactly ``target_samples`` samples using a deterministic strategy.

    Short signals are right-padded with zeros. Long signals may be cropped from
    the start, center, or the maximum-energy region.
    """
    waveform = _as_mono_float32(waveform)
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")

    n_samples = waveform.size
    if n_samples == target_samples:
        return waveform.copy()
    if n_samples < target_samples:
        return np.pad(waveform, (0, target_samples - n_samples), mode="constant").astype(np.float32)

    if strategy == "start":
        start = 0
    elif strategy == "center":
        start = (n_samples - target_samples) // 2
    elif strategy == "energy":
        start = _energy_crop_start(waveform, target_samples)
    else:
        raise ValueError("strategy must be one of: 'start', 'center', 'energy'")
    return np.ascontiguousarray(waveform[start : start + target_samples], dtype=np.float32)


def _scale_to_peak_limit(waveform: np.ndarray, peak_limit: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak <= peak_limit or peak == 0.0:
        return np.asarray(waveform, dtype=np.float32)
    return np.asarray(waveform * (peak_limit / peak), dtype=np.float32)


def mix_two_sources(
    first: np.ndarray,
    second: np.ndarray,
    relative_db: float = 0.0,
    overlap_ratio: float = 1.0,
    target_dbfs: float = -20.0,
) -> np.ndarray:
    """Create a controlled two-source mixture of equal-length waveforms.

    ``relative_db`` is defined as first-source level minus second-source level.
    Therefore ``+6 dB`` makes the first source approximately 6 dB stronger.
    ``overlap_ratio`` controls how much of the second source is placed inside the
    output window: 1.0 means full overlap and 0.5 means the second source starts
    halfway through the window.
    """
    first = _as_mono_float32(first)
    second = _as_mono_float32(second)
    if first.shape != second.shape:
        raise ValueError("Sources must have the same shape before mixing")
    if not 0 < float(overlap_ratio) <= 1:
        raise ValueError("overlap_ratio must be in (0, 1]")
    if not np.isfinite(float(relative_db)):
        raise ValueError("relative_db must be finite")

    # Controlled mixture synthesis benefits from a wider normalization range
    # than live inference. The silence guard still prevents extreme amplification.
    first_norm = normalize_rms(first, target_dbfs=target_dbfs, max_gain_db=40.0)
    second_norm = normalize_rms(second, target_dbfs=target_dbfs, max_gain_db=40.0)

    second_gain = 10.0 ** (-float(relative_db) / 20.0)
    second_norm = second_norm * np.float32(second_gain)

    n_samples = first_norm.size
    overlap_samples = max(1, min(n_samples, round(n_samples * float(overlap_ratio))))
    second_start = n_samples - overlap_samples

    mixed = first_norm.copy()
    mixed[second_start:] += second_norm[:overlap_samples]
    return _scale_to_peak_limit(mixed)


def save_audio(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Write a mono float waveform as a WAV-compatible audio file."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    waveform = _as_mono_float32(waveform)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, int(sample_rate))
