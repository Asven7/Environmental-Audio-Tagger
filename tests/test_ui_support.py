from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from esaudio.ui_support import (
    audio_array_to_float32,
    clear_microphone_state,
    convert_microphone_chunk,
    file_results_headers,
    file_results_rows,
    mark_microphone_stopped,
    new_microphone_state,
    process_microphone_chunk,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _Prediction:
    scores: dict[str, float]
    active_labels: list[str]
    status: str


class _FakeTagger:
    sample_rate = 10
    window_seconds = 2.0
    hop_seconds = 1.0
    class_names = ["a", "b"]
    thresholds = np.asarray([0.5, 0.6], dtype=np.float32)

    def __init__(self):
        self.calls = 0

    def predict_waveform(self, waveform):
        self.calls += 1
        mean = float(np.mean(waveform))
        scores = {"a": mean, "b": 1.0 - mean}
        active = [
            name
            for name, threshold in zip(self.class_names, self.thresholds)
            if scores[name] >= threshold
        ]
        return _Prediction(
            scores=scores,
            active_labels=active,
            status=(
                "known_labels_detected"
                if active
                else "no_confident_known_class"
            ),
        )


def test_audio_array_to_float32_normalizes_int16_and_downmixes():
    stereo = np.asarray(
        [
            [32767, -32768],
            [16384, 16384],
        ],
        dtype=np.int16,
    )
    mono = audio_array_to_float32(stereo)
    assert mono.dtype == np.float32
    assert mono.shape == (2,)
    assert abs(float(mono[0])) < 1e-3
    assert mono[1] == pytest.approx(0.5, abs=2e-4)


def test_microphone_chunk_resampling_uses_cumulative_target_length():
    tagger = _FakeTagger()
    state = new_microphone_state(tagger)

    first = convert_microphone_chunk(
        (15, np.ones(7, dtype=np.int16) * 1000),
        state,
        target_sample_rate=10,
    )
    second = convert_microphone_chunk(
        (15, np.ones(8, dtype=np.int16) * 1000),
        state,
        target_sample_rate=10,
    )

    assert first.size + second.size == 10
    assert state.source_samples_seen == 15
    assert state.target_samples_produced == 10


def test_microphone_stream_buffers_until_first_complete_window():
    tagger = _FakeTagger()
    state = new_microphone_state(tagger)

    audio = (10, np.zeros(10, dtype=np.int16))
    state, status, active, processing, scores, history = (
        process_microphone_chunk(tagger, audio, state)
    )

    assert "Buffering" in status
    assert active == ""
    assert processing == 0.0
    assert scores == []
    assert history == []
    assert tagger.calls == 0


def test_microphone_stream_emits_at_two_seconds_then_each_hop():
    tagger = _FakeTagger()
    state = new_microphone_state(tagger)

    first = (10, np.zeros(10, dtype=np.int16))
    second = (10, np.zeros(10, dtype=np.int16))
    third = (10, np.ones(10, dtype=np.int16) * 32767)

    state, *_ = process_microphone_chunk(tagger, first, state)
    state, status2, _, _, score_rows2, history2 = process_microphone_chunk(
        tagger,
        second,
        state,
    )
    state, status3, _, _, score_rows3, history3 = process_microphone_chunk(
        tagger,
        third,
        state,
    )

    assert tagger.calls == 2
    assert len(history2) == 1
    assert len(history3) == 2
    assert history2[0][1:3] == [0.0, 2.0]
    assert history3[-1][1:3] == [1.0, 3.0]
    assert len(score_rows2) == len(tagger.class_names)
    assert len(score_rows3) == len(tagger.class_names)
    assert status2
    assert status3


def test_microphone_stream_rejects_mid_session_sample_rate_change():
    tagger = _FakeTagger()
    state = new_microphone_state(tagger)
    convert_microphone_chunk(
        (10, np.zeros(5, dtype=np.int16)),
        state,
        target_sample_rate=10,
    )
    with pytest.raises(ValueError, match="sample rate changed"):
        convert_microphone_chunk(
            (20, np.zeros(5, dtype=np.int16)),
            state,
            target_sample_rate=10,
        )


def test_file_table_keeps_class_order_and_frozen_score_columns():
    tagger = _FakeTagger()
    from esaudio.streaming import WindowPrediction

    prediction = WindowPrediction(
        index=0,
        start_seconds=0.0,
        end_seconds=2.0,
        valid_end_seconds=2.0,
        padded=False,
        scores={"b": 0.2, "a": 0.8},
        active_labels=["a"],
        status="known_labels_detected",
        display_status="Known class(es) detected",
        processing_ms=3.2,
    )

    headers = file_results_headers(tagger)
    rows = file_results_rows(tagger, [prediction])

    assert headers[-2:] == ["score:a", "score:b"]
    assert rows[0][-2:] == [0.8, 0.2]


def test_ui_and_microphone_scripts_do_not_touch_frozen_test_or_retune():
    for relative in ("scripts/run_ui.py", "scripts/live_microphone.py"):
        source = (ROOT / relative).read_text(encoding="utf-8").lower()
        forbidden = (
            "known_test",
            "ood_test",
            "test_evaluation",
            "tune_per_class_thresholds",
            "save_thresholds",
        )
        for token in forbidden:
            assert token not in source


def test_live_microphone_uses_new_streaming_engine_not_legacy_buffer():
    source = (ROOT / "scripts/live_microphone.py").read_text(
        encoding="utf-8"
    )
    assert "StreamingInferenceEngine" in source
    assert "StreamingWindowBuffer" not in source



def test_complete_microphone_history_is_preserved_beyond_thirty_windows():
    tagger = _FakeTagger()
    state = new_microphone_state(tagger)

    # 40 seconds at 10 Hz with a 2 s window / 1 s hop produces 39 windows.
    audio = (10, np.zeros(400, dtype=np.int16))
    state, _, _, _, _, history = process_microphone_chunk(
        tagger,
        audio,
        state,
    )

    assert len(history) == 39
    assert history[0][0:3] == [0, 0.0, 2.0]
    assert history[-1][0:3] == [38, 38.0, 40.0]

    state, status, _, _, _, stopped_history = mark_microphone_stopped(
        tagger,
        state,
    )
    assert status == "Recording stopped. Results preserved."
    assert len(stopped_history) == 39

def test_stop_preserves_latest_scores_and_history():
    tagger = _FakeTagger()
    state = new_microphone_state(tagger)
    state, *_ = process_microphone_chunk(
        tagger,
        (10, np.zeros(20, dtype=np.int16)),
        state,
    )

    previous_scores = list(state.latest_scores)
    previous_history = list(state.history)
    (
        state,
        status,
        _,
        _,
        scores,
        history,
    ) = mark_microphone_stopped(tagger, state)

    assert status == "Recording stopped. Results preserved."
    assert state.is_recording is False
    assert scores == previous_scores
    assert history == previous_history
    assert scores
    assert history


def test_late_none_chunk_after_stop_does_not_clear_or_overwrite_status():
    tagger = _FakeTagger()
    state = new_microphone_state(tagger)
    state, *_ = process_microphone_chunk(
        tagger,
        (10, np.zeros(20, dtype=np.int16)),
        state,
    )
    state, *_ = mark_microphone_stopped(tagger, state)

    before_scores = list(state.latest_scores)
    before_history = list(state.history)
    state, status, _, _, scores, history = process_microphone_chunk(
        tagger,
        None,
        state,
    )

    assert status == "Recording stopped. Results preserved."
    assert scores == before_scores
    assert history == before_history


def test_clear_is_explicit_and_separate_from_stop():
    tagger = _FakeTagger()
    state, status, active, processing, scores, history = (
        clear_microphone_state(tagger)
    )

    assert state.is_recording is False
    assert status == "Results cleared. Press record to start again."
    assert active == ""
    assert processing == 0.0
    assert scores == []
    assert history == []
