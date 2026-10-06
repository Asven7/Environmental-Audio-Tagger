from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .audio import load_audio


class PredictionLike(Protocol):
    scores: dict[str, float]
    active_labels: list[str]
    status: str


class AudioTaggerLike(Protocol):
    sample_rate: int
    window_seconds: float
    hop_seconds: float
    class_names: list[str]

    def predict_waveform(self, waveform: np.ndarray) -> PredictionLike:
        ...



@dataclass
class StreamWindow:
    """Backward-compatible window payload used by the legacy microphone CLI."""

    index: int
    start_sample: int
    end_sample: int
    waveform: np.ndarray


class StreamingWindowBuffer:
    """Backward-compatible sample-count streaming buffer.

    This preserves the pre-Phase-13 public API used by
    ``scripts/live_microphone.py``. New code should prefer
    :class:`RollingWindowBuffer`, which also carries sample-rate-aware timing.
    """

    def __init__(self, window_samples: int, hop_samples: int) -> None:
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window_samples and hop_samples must be positive")
        if hop_samples > window_samples:
            raise ValueError("hop_samples must not exceed window_samples")
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self._buffer = np.empty(0, dtype=np.float32)
        self._samples_seen = 0
        self._next_emit_at = self.window_samples
        self._window_index = 0

    def push(self, chunk: np.ndarray) -> list[StreamWindow]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []

        self._buffer = np.concatenate([self._buffer, chunk])
        self._samples_seen += int(chunk.size)
        emitted: list[StreamWindow] = []

        while self._samples_seen >= self._next_emit_at:
            end_sample = self._next_emit_at
            start_sample = end_sample - self.window_samples
            buffer_absolute_start = self._samples_seen - self._buffer.size
            local_start = start_sample - buffer_absolute_start
            local_end = end_sample - buffer_absolute_start
            if local_start < 0 or local_end > self._buffer.size:
                raise RuntimeError("Streaming buffer bookkeeping error")

            waveform = self._buffer[local_start:local_end].copy()
            emitted.append(
                StreamWindow(
                    index=self._window_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    waveform=waveform,
                )
            )
            self._window_index += 1
            self._next_emit_at += self.hop_samples

        earliest_needed = max(
            0,
            self._next_emit_at - self.window_samples,
        )
        buffer_absolute_start = self._samples_seen - self._buffer.size
        trim = max(0, earliest_needed - buffer_absolute_start)
        if trim > 0:
            self._buffer = self._buffer[trim:]
        return emitted

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._samples_seen = 0
        self._next_emit_at = self.window_samples
        self._window_index = 0

@dataclass(frozen=True)
class WindowSlice:
    index: int
    start_sample: int
    end_sample: int
    valid_end_sample: int
    sample_rate: int
    waveform: np.ndarray

    @property
    def start_seconds(self) -> float:
        return self.start_sample / float(self.sample_rate)

    @property
    def end_seconds(self) -> float:
        return self.end_sample / float(self.sample_rate)

    @property
    def valid_end_seconds(self) -> float:
        return self.valid_end_sample / float(self.sample_rate)

    @property
    def padded(self) -> bool:
        return self.valid_end_sample < self.end_sample


@dataclass(frozen=True)
class WindowPrediction:
    index: int
    start_seconds: float
    end_seconds: float
    valid_end_seconds: float
    padded: bool
    scores: dict[str, float]
    active_labels: list[str]
    status: str
    display_status: str
    processing_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def display_status(status: str, active_labels: list[str]) -> str:
    if active_labels:
        return "Known class(es) detected"
    if status == "no_confident_known_class":
        return "No confident known class"
    return str(status)


def score_rows(
    class_names: Iterable[str],
    scores: dict[str, float],
    active_labels: Iterable[str],
    thresholds: Iterable[float] | None = None,
) -> list[dict]:
    names = list(class_names)
    active = set(active_labels)
    threshold_values = list(thresholds) if thresholds is not None else None
    if threshold_values is not None and len(threshold_values) != len(names):
        raise ValueError("Threshold count must match class count")

    rows = []
    for index, name in enumerate(names):
        if name not in scores:
            raise ValueError(f"Missing score for class {name!r}")
        row = {
            "class": name,
            "score": float(scores[name]),
            "active": name in active,
        }
        if threshold_values is not None:
            row["threshold"] = float(threshold_values[index])
        rows.append(row)
    return rows


def iter_waveform_windows(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> Iterable[WindowSlice]:
    """Yield sequential fixed windows; right-pad the final partial window."""

    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    window_samples = round(float(window_seconds) * sample_rate)
    hop_samples = round(float(hop_seconds) * sample_rate)
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected mono waveform [T], got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Waveform contains non-finite values")
    if array.size == 0:
        return

    index = 0
    start = 0
    total = int(array.size)
    while start < total:
        nominal_end = start + window_samples
        valid_end = min(nominal_end, total)
        chunk = array[start:valid_end]
        if chunk.size < window_samples:
            chunk = np.pad(
                chunk,
                (0, window_samples - chunk.size),
                mode="constant",
            )
        yield WindowSlice(
            index=index,
            start_sample=start,
            end_sample=nominal_end,
            valid_end_sample=valid_end,
            sample_rate=sample_rate,
            waveform=chunk.astype(np.float32, copy=False),
        )
        index += 1
        start += hop_samples


class RollingWindowBuffer:
    """Continuous fixed-rate buffer that emits one window per hop.

    The first window is emitted only after ``window_seconds`` of audio have
    arrived. Thereafter windows are emitted every ``hop_seconds``. Arbitrary
    input chunk sizes are supported, including chunks spanning multiple hops.
    """

    def __init__(
        self,
        sample_rate: int,
        window_seconds: float,
        hop_seconds: float,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window_samples = round(float(window_seconds) * self.sample_rate)
        self.hop_samples = round(float(hop_seconds) * self.sample_rate)

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.window_samples <= 0 or self.hop_samples <= 0:
            raise ValueError("window/hop lengths must be positive")
        if self.hop_samples > self.window_samples:
            raise ValueError("hop must not exceed window length")

        self.reset()

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_sample = 0
        self._total_samples = 0
        self._next_emit_sample = self.window_samples
        self._emitted_windows = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def emitted_windows(self) -> int:
        return self._emitted_windows

    @property
    def ready(self) -> bool:
        return self._total_samples >= self.window_samples

    @property
    def buffered_seconds(self) -> float:
        return min(self._total_samples, self.window_samples) / float(
            self.sample_rate
        )

    def push(self, samples: np.ndarray) -> list[WindowSlice]:
        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim == 2:
            # sounddevice-style [frames, channels] input. Downmix by mean,
            # matching the project's deterministic file downmix behavior.
            if chunk.shape[1] == 0:
                raise ValueError("Audio chunk has zero channels")
            chunk = chunk.mean(axis=1, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(
                f"Expected audio chunk [T] or [T,C], got shape {chunk.shape}"
            )
        if chunk.size == 0:
            return []
        if not np.isfinite(chunk).all():
            raise ValueError("Audio chunk contains non-finite values")

        self._buffer = np.concatenate(
            [self._buffer, chunk.astype(np.float32, copy=False)]
        )
        self._total_samples += int(chunk.size)

        emitted: list[WindowSlice] = []
        while self._next_emit_sample <= self._total_samples:
            start = self._next_emit_sample - self.window_samples
            local_start = start - self._buffer_start_sample
            local_end = self._next_emit_sample - self._buffer_start_sample
            if local_start < 0 or local_end > self._buffer.size:
                raise RuntimeError("Rolling buffer internal indexing error")

            waveform = self._buffer[local_start:local_end].copy()
            emitted.append(
                WindowSlice(
                    index=self._emitted_windows,
                    start_sample=start,
                    end_sample=self._next_emit_sample,
                    valid_end_sample=self._next_emit_sample,
                    sample_rate=self.sample_rate,
                    waveform=waveform,
                )
            )
            self._emitted_windows += 1
            self._next_emit_sample += self.hop_samples

        # Keep only samples that can be part of the next not-yet-emitted
        # window. This bounds memory independently of stream duration.
        earliest_needed = max(
            0,
            self._next_emit_sample - self.window_samples,
        )
        trim = earliest_needed - self._buffer_start_sample
        if trim > 0:
            self._buffer = self._buffer[trim:].copy()
            self._buffer_start_sample = earliest_needed

        return emitted


class StreamingInferenceEngine:
    def __init__(self, tagger: AudioTaggerLike) -> None:
        self.tagger = tagger
        self.buffer = RollingWindowBuffer(
            tagger.sample_rate,
            tagger.window_seconds,
            tagger.hop_seconds,
        )

    def reset(self) -> None:
        self.buffer.reset()

    def push(self, samples: np.ndarray) -> list[WindowPrediction]:
        predictions: list[WindowPrediction] = []
        for window in self.buffer.push(samples):
            predictions.append(self._predict_window(window))
        return predictions

    def _predict_window(self, window: WindowSlice) -> WindowPrediction:
        start = time.perf_counter()
        result = self.tagger.predict_waveform(window.waveform)
        processing_ms = (time.perf_counter() - start) * 1000.0
        return WindowPrediction(
            index=window.index,
            start_seconds=window.start_seconds,
            end_seconds=window.end_seconds,
            valid_end_seconds=window.valid_end_seconds,
            padded=window.padded,
            scores={str(k): float(v) for k, v in result.scores.items()},
            active_labels=list(result.active_labels),
            status=str(result.status),
            display_status=display_status(
                str(result.status),
                list(result.active_labels),
            ),
            processing_ms=float(processing_ms),
        )


def analyze_waveform(
    tagger: AudioTaggerLike,
    waveform: np.ndarray,
) -> list[WindowPrediction]:
    results: list[WindowPrediction] = []
    for window in iter_waveform_windows(
        waveform,
        sample_rate=tagger.sample_rate,
        window_seconds=tagger.window_seconds,
        hop_seconds=tagger.hop_seconds,
    ):
        start = time.perf_counter()
        prediction = tagger.predict_waveform(window.waveform)
        processing_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            WindowPrediction(
                index=window.index,
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                valid_end_seconds=window.valid_end_seconds,
                padded=window.padded,
                scores={
                    str(k): float(v)
                    for k, v in prediction.scores.items()
                },
                active_labels=list(prediction.active_labels),
                status=str(prediction.status),
                display_status=display_status(
                    str(prediction.status),
                    list(prediction.active_labels),
                ),
                processing_ms=float(processing_ms),
            )
        )
    return results


def analyze_file(
    tagger: AudioTaggerLike,
    path: str | Path,
) -> list[WindowPrediction]:
    """Analyze an entire file in sequential overlapping windows."""

    waveform = load_audio(path, tagger.sample_rate)
    return analyze_waveform(tagger, waveform)
