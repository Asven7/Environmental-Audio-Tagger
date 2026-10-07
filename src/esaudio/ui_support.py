from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Iterable

import numpy as np
from scipy.signal import resample_poly

from .streaming import (
    RollingWindowBuffer,
    WindowPrediction,
    display_status,
    score_rows,
)


@dataclass
class MicrophoneStreamState:
    buffer: RollingWindowBuffer
    source_sample_rate: int | None = None
    source_samples_seen: int = 0
    target_samples_produced: int = 0
    history: list[list] = field(default_factory=list)
    is_recording: bool = True
    latest_status: str = ""
    latest_active: str = ""
    latest_processing_ms: float = 0.0
    latest_scores: list[list] = field(default_factory=list)


def new_microphone_state(
    tagger,
    *,
    recording: bool = True,
) -> MicrophoneStreamState:
    state = MicrophoneStreamState(
        buffer=RollingWindowBuffer(
            int(tagger.sample_rate),
            float(tagger.window_seconds),
            float(tagger.hop_seconds),
        ),
        is_recording=bool(recording),
    )
    state.latest_status = (
        f"Recording started. Waiting for the first complete "
        f"{float(tagger.window_seconds):.1f} s window."
        if recording
        else "Ready. Press record to start."
    )
    return state


def microphone_display_tuple(
    state: MicrophoneStreamState,
) -> tuple[
    MicrophoneStreamState,
    str,
    str,
    float,
    list[list],
    list[list],
]:
    return (
        state,
        state.latest_status,
        state.latest_active,
        state.latest_processing_ms,
        list(state.latest_scores),
        list(state.history),
    )


def mark_microphone_stopped(
    tagger,
    state: MicrophoneStreamState | None,
) -> tuple[
    MicrophoneStreamState,
    str,
    str,
    float,
    list[list],
    list[list],
]:
    if state is None:
        state = new_microphone_state(tagger, recording=False)
    state.is_recording = False
    state.latest_status = "Recording stopped. Results preserved."
    return microphone_display_tuple(state)


def clear_microphone_state(
    tagger,
) -> tuple[
    MicrophoneStreamState,
    str,
    str,
    float,
    list[list],
    list[list],
]:
    state = new_microphone_state(tagger, recording=False)
    state.latest_status = "Results cleared. Press record to start again."
    state.latest_active = ""
    state.latest_processing_ms = 0.0
    state.latest_scores = []
    state.history = []
    return microphone_display_tuple(state)


def audio_array_to_float32(data: np.ndarray) -> np.ndarray:
    """Convert Gradio/sound-card-style audio to mono float32 in [-1, 1]."""

    array = np.asarray(data)
    if array.ndim == 2:
        if array.shape[1] == 0:
            raise ValueError("Audio input has zero channels")
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32)
        else:
            array = array.astype(np.float32, copy=False)
        array = array.mean(axis=1)
    elif array.ndim == 1:
        array = array.astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"Expected audio [T] or [T,C], got shape {array.shape}"
        )

    original_dtype = np.asarray(data).dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        if scale <= 0:
            raise ValueError("Invalid integer audio dtype")
        array = array / scale

    if not np.isfinite(array).all():
        raise ValueError("Audio input contains non-finite values")
    return np.clip(array, -1.0, 1.0).astype(np.float32, copy=False)


def _fit_length(array: np.ndarray, target_length: int) -> np.ndarray:
    target_length = int(target_length)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if array.size == target_length:
        return array.astype(np.float32, copy=False)
    if array.size > target_length:
        return array[:target_length].astype(np.float32, copy=False)
    if target_length == 0:
        return np.empty(0, dtype=np.float32)
    return np.pad(
        array,
        (0, target_length - array.size),
        mode="constant",
    ).astype(np.float32, copy=False)


def convert_microphone_chunk(
    audio: tuple[int, np.ndarray],
    state: MicrophoneStreamState,
    *,
    target_sample_rate: int,
) -> np.ndarray:
    """Convert one browser microphone chunk to the model sample rate."""

    if audio is None:
        return np.empty(0, dtype=np.float32)
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        raise ValueError("Microphone audio must be (sample_rate, numpy_array)")

    source_rate = int(audio[0])
    target_rate = int(target_sample_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")

    chunk = audio_array_to_float32(np.asarray(audio[1]))
    if chunk.size == 0:
        return chunk

    if state.source_sample_rate is None:
        state.source_sample_rate = source_rate
    elif state.source_sample_rate != source_rate:
        raise ValueError(
            "Microphone sample rate changed during one recording session; "
            "restart the recording"
        )

    source_after = state.source_samples_seen + int(chunk.size)
    target_after = round(source_after * target_rate / source_rate)
    target_needed = target_after - state.target_samples_produced

    if source_rate == target_rate:
        converted = chunk
    else:
        common = gcd(source_rate, target_rate)
        converted = resample_poly(
            chunk,
            target_rate // common,
            source_rate // common,
        ).astype(np.float32, copy=False)

    converted = _fit_length(converted, target_needed)
    state.source_samples_seen = source_after
    state.target_samples_produced += int(converted.size)
    return converted


def prediction_score_rows(tagger, prediction: WindowPrediction) -> list[list]:
    rows = score_rows(
        tagger.class_names,
        prediction.scores,
        prediction.active_labels,
        tagger.thresholds,
    )
    return [
        [
            row["class"],
            round(float(row["score"]), 4),
            round(float(row["threshold"]), 4),
            bool(row["active"]),
        ]
        for row in rows
    ]


def history_row(prediction: WindowPrediction) -> list:
    return [
        int(prediction.index),
        round(float(prediction.start_seconds), 3),
        round(float(prediction.end_seconds), 3),
        ", ".join(prediction.active_labels) if prediction.active_labels else "—",
        prediction.display_status,
        round(float(prediction.processing_ms), 3),
    ]


def append_history(
    state: MicrophoneStreamState,
    predictions: Iterable[WindowPrediction],
    *,
    max_rows: int | None = None,
) -> None:
    """Append live-window rows.

    By default, preserve the complete recording-session history until the user
    explicitly clears the UI or starts a new recording. ``max_rows`` remains
    available for callers that intentionally want a bounded view.
    """

    for prediction in predictions:
        state.history.append(history_row(prediction))

    if max_rows is not None and max_rows > 0 and len(state.history) > max_rows:
        state.history[:] = state.history[-max_rows:]


def process_microphone_chunk(
    tagger,
    audio: tuple[int, np.ndarray] | None,
    state: MicrophoneStreamState | None,
) -> tuple[
    MicrophoneStreamState,
    str,
    str,
    float,
    list[list],
    list[list],
]:
    """Process one Gradio microphone chunk and preserve final display state."""

    if state is None:
        state = new_microphone_state(tagger, recording=True)

    # A late stream event can arrive after stop_recording. Ignore it so that
    # "Recording stopped" and the last visible results are not overwritten.
    if not state.is_recording:
        return microphone_display_tuple(state)

    # Some Gradio/browser combinations emit a final None chunk when recording
    # stops. Preserve the last visible result instead of clearing the tables.
    if audio is None:
        return microphone_display_tuple(state)

    converted = convert_microphone_chunk(
        audio,
        state,
        target_sample_rate=int(tagger.sample_rate),
    )

    predictions: list[WindowPrediction] = []
    for window in state.buffer.push(converted):
        import time

        start = time.perf_counter()
        result = tagger.predict_waveform(window.waveform)
        processing_ms = (time.perf_counter() - start) * 1000.0
        predictions.append(
            WindowPrediction(
                index=window.index,
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                valid_end_seconds=window.valid_end_seconds,
                padded=window.padded,
                scores={
                    str(name): float(score)
                    for name, score in result.scores.items()
                },
                active_labels=list(result.active_labels),
                status=str(result.status),
                display_status=display_status(
                    str(result.status),
                    list(result.active_labels),
                ),
                processing_ms=float(processing_ms),
            )
        )

    append_history(state, predictions)

    if not predictions:
        buffered = state.buffer.buffered_seconds
        state.latest_status = (
            f"Buffering microphone audio: {buffered:.2f} / "
            f"{float(tagger.window_seconds):.2f} s"
        )
        return microphone_display_tuple(state)

    latest = predictions[-1]
    state.latest_status = latest.display_status
    state.latest_active = (
        ", ".join(latest.active_labels) if latest.active_labels else "—"
    )
    state.latest_processing_ms = round(float(latest.processing_ms), 3)
    state.latest_scores = prediction_score_rows(tagger, latest)
    return microphone_display_tuple(state)


def file_results_rows(tagger, predictions: Iterable[WindowPrediction]) -> list[list]:
    rows: list[list] = []
    for prediction in predictions:
        rows.append(
            [
                int(prediction.index),
                round(float(prediction.start_seconds), 3),
                round(float(prediction.end_seconds), 3),
                round(float(prediction.valid_end_seconds), 3),
                bool(prediction.padded),
                ", ".join(prediction.active_labels)
                if prediction.active_labels
                else "—",
                prediction.display_status,
                round(float(prediction.processing_ms), 3),
                *[
                    round(float(prediction.scores[name]), 4)
                    for name in tagger.class_names
                ],
            ]
        )
    return rows


def file_results_headers(tagger) -> list[str]:
    return [
        "window",
        "start_s",
        "end_s",
        "valid_end_s",
        "padded",
        "active_labels",
        "status",
        "processing_ms",
        *[f"score:{name}" for name in tagger.class_names],
    ]
