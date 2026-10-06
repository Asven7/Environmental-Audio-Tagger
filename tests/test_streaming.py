from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from esaudio.streaming import (
    RollingWindowBuffer,
    StreamingInferenceEngine,
    StreamingWindowBuffer,
    analyze_waveform,
    display_status,
    iter_waveform_windows,
    score_rows,
)


@dataclass
class _FakePrediction:
    scores: dict[str, float]
    active_labels: list[str]
    status: str


class _FakeTagger:
    sample_rate = 10
    window_seconds = 2.0
    hop_seconds = 1.0
    class_names = ["a", "b"]
    thresholds = np.asarray([0.5, 0.5], dtype=np.float32)

    def __init__(self):
        self.calls = []

    def predict_waveform(self, waveform):
        array = np.asarray(waveform, dtype=np.float32)
        self.calls.append(array.copy())
        mean = float(np.mean(array))
        scores = {"a": mean, "b": 1.0 - mean}
        active = [name for name, score in scores.items() if score >= 0.5]
        return _FakePrediction(
            scores=scores,
            active_labels=active,
            status=(
                "known_labels_detected"
                if active
                else "no_confident_known_class"
            ),
        )


def test_streaming_window_buffer_overlap():
    """Preserve the pre-Phase-13 public API used by live_microphone.py."""

    buffer = StreamingWindowBuffer(window_samples=8, hop_samples=4)
    out = []
    out.extend(buffer.push(np.arange(0, 3, dtype=np.float32)))
    assert out == []
    out.extend(buffer.push(np.arange(3, 10, dtype=np.float32)))
    assert len(out) == 1
    np.testing.assert_array_equal(
        out[0].waveform,
        np.arange(0, 8, dtype=np.float32),
    )
    out.extend(buffer.push(np.arange(10, 14, dtype=np.float32)))
    assert len(out) == 2
    np.testing.assert_array_equal(
        out[1].waveform,
        np.arange(4, 12, dtype=np.float32),
    )


def test_file_windows_follow_two_second_window_one_second_hop_and_pad_final():
    waveform = np.arange(25, dtype=np.float32)
    windows = list(
        iter_waveform_windows(
            waveform,
            sample_rate=10,
            window_seconds=2.0,
            hop_seconds=1.0,
        )
    )

    assert len(windows) == 3
    assert [(w.start_sample, w.end_sample) for w in windows] == [
        (0, 20),
        (10, 30),
        (20, 40),
    ]
    assert windows[0].padded is False
    assert windows[1].padded is True
    assert windows[2].padded is True
    np.testing.assert_array_equal(windows[0].waveform, np.arange(20))
    np.testing.assert_array_equal(windows[1].waveform[:15], np.arange(10, 25))
    np.testing.assert_array_equal(windows[1].waveform[15:], np.zeros(5))


def test_rolling_buffer_waits_for_first_complete_window():
    buffer = RollingWindowBuffer(10, 2.0, 1.0)
    assert buffer.push(np.arange(19, dtype=np.float32)) == []
    assert buffer.ready is False

    emitted = buffer.push(np.asarray([19], dtype=np.float32))
    assert buffer.ready is True
    assert len(emitted) == 1
    assert emitted[0].start_seconds == pytest.approx(0.0)
    assert emitted[0].end_seconds == pytest.approx(2.0)
    np.testing.assert_array_equal(emitted[0].waveform, np.arange(20))


def test_rolling_buffer_emits_exact_overlapping_windows():
    buffer = RollingWindowBuffer(10, 2.0, 1.0)

    first = buffer.push(np.arange(20, dtype=np.float32))
    second = buffer.push(np.arange(20, 30, dtype=np.float32))
    third = buffer.push(np.arange(30, 40, dtype=np.float32))

    assert len(first) == len(second) == len(third) == 1
    np.testing.assert_array_equal(first[0].waveform, np.arange(0, 20))
    np.testing.assert_array_equal(second[0].waveform, np.arange(10, 30))
    np.testing.assert_array_equal(third[0].waveform, np.arange(20, 40))


def test_rolling_buffer_handles_chunk_spanning_multiple_hops():
    buffer = RollingWindowBuffer(10, 2.0, 1.0)
    emitted = buffer.push(np.arange(45, dtype=np.float32))

    assert len(emitted) == 3
    assert [w.start_sample for w in emitted] == [0, 10, 20]
    assert [w.end_sample for w in emitted] == [20, 30, 40]


def test_rolling_buffer_downmixes_stereo_and_reset_clears_state():
    buffer = RollingWindowBuffer(10, 2.0, 1.0)
    stereo = np.stack(
        [
            np.arange(20, dtype=np.float32),
            np.arange(20, dtype=np.float32) + 2,
        ],
        axis=1,
    )
    emitted = buffer.push(stereo)
    assert len(emitted) == 1
    np.testing.assert_allclose(emitted[0].waveform, np.arange(20) + 1)

    buffer.reset()
    assert buffer.total_samples == 0
    assert buffer.emitted_windows == 0
    assert buffer.ready is False


def test_streaming_engine_calls_tagger_once_per_emitted_hop():
    tagger = _FakeTagger()
    engine = StreamingInferenceEngine(tagger)

    assert engine.push(np.zeros(10, dtype=np.float32)) == []
    first = engine.push(np.zeros(10, dtype=np.float32))
    second = engine.push(np.ones(10, dtype=np.float32))

    assert len(first) == 1
    assert len(second) == 1
    assert len(tagger.calls) == 2
    assert first[0].start_seconds == pytest.approx(0.0)
    assert second[0].start_seconds == pytest.approx(1.0)
    assert first[0].processing_ms >= 0
    assert second[0].processing_ms >= 0


def test_analyze_waveform_returns_timestamped_predictions_for_all_windows():
    tagger = _FakeTagger()
    waveform = np.linspace(0.0, 1.0, 25, dtype=np.float32)

    results = analyze_waveform(tagger, waveform)

    assert len(results) == 3
    assert [result.start_seconds for result in results] == [0.0, 1.0, 2.0]
    assert [result.end_seconds for result in results] == [2.0, 3.0, 4.0]
    assert results[-1].padded is True
    assert len(tagger.calls) == 3


def test_display_payload_uses_transparent_rejection_wording_and_class_order():
    assert (
        display_status("no_confident_known_class", [])
        == "No confident known class"
    )
    rows = score_rows(
        ["a", "b"],
        {"b": 0.2, "a": 0.8},
        ["a"],
        [0.5, 0.6],
    )
    assert [row["class"] for row in rows] == ["a", "b"]
    assert rows[0] == {
        "class": "a",
        "score": pytest.approx(0.8),
        "active": True,
        "threshold": pytest.approx(0.5),
    }
    assert rows[1]["active"] is False
