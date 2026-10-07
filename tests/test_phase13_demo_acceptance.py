from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "verify_phase13_demo.py"
    spec = importlib.util.spec_from_file_location(
        "verify_phase13_demo_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Tagger:
    class_names = ["a", "b"]
    hop_seconds = 1.0
    window_seconds = 2.0


def _prediction(index: int):
    return SimpleNamespace(
        index=index,
        start_seconds=float(index),
        end_seconds=float(index + 2),
        scores={"a": 0.25, "b": 0.75},
        processing_ms=4.0,
    )


def test_validate_predictions_accepts_expected_window_grid():
    module = _load_module()
    result = module._validate_predictions(
        _Tagger(),
        [_prediction(0), _prediction(1), _prediction(2)],
    )

    assert result["n_windows"] == 3
    assert result["first_start_seconds"] == 0.0
    assert result["last_end_seconds"] == 4.0
    assert result["observed_file_smoke_below_hop"] is True


def test_validate_predictions_rejects_broken_timestamp_grid():
    module = _load_module()
    bad = _prediction(1)
    bad.start_seconds = 1.5

    with pytest.raises(RuntimeError, match="start mismatch"):
        module._validate_predictions(
            _Tagger(),
            [_prediction(0), bad],
        )


def test_validate_predictions_rejects_invalid_score():
    module = _load_module()
    bad = _prediction(0)
    bad.scores["b"] = 1.5

    with pytest.raises(RuntimeError, match="Invalid score"):
        module._validate_predictions(_Tagger(), [bad])


def test_phase13c_script_does_not_touch_frozen_evaluation_protocol():
    source = (
        ROOT / "scripts" / "verify_phase13_demo.py"
    ).read_text(encoding="utf-8").lower()

    forbidden = (
        "tune_per_class_thresholds",
        "save_thresholds",
        "run_frozen_evaluation",
        "test_evaluation.json",
    )
    for token in forbidden:
        assert token not in source
