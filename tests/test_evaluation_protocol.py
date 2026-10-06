from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esaudio.checkpoints import (
    load_threshold_artifact,
    load_thresholds,
    save_thresholds,
)
from esaudio.evaluate_runner import (
    THRESHOLD_PROTOCOL_VERSION,
    file_sha256,
    save_evaluation,
    validate_threshold_provenance,
)
from esaudio.evaluation import (
    apply_thresholds,
    group_metrics,
    rejection_metrics,
    tune_per_class_thresholds,
)


def test_threshold_tuning_tie_break_is_order_independent_and_prefers_lower_equidistant():
    y_true = np.array([[1], [1], [0], [0]], dtype=np.int64)
    scores = np.array([[0.9], [0.8], [0.2], [0.1]], dtype=np.float32)

    first = tune_per_class_thresholds(y_true, scores, [0.6, 0.4])
    second = tune_per_class_thresholds(y_true, scores, [0.4, 0.6])

    assert first.thresholds[0] == pytest.approx(0.4)
    assert second.thresholds[0] == pytest.approx(0.4)
    assert first.per_class_f1[0] == pytest.approx(1.0)


def test_threshold_tuning_rejects_missing_positive_validation_class():
    y_true = np.array([[1, 0], [0, 0]], dtype=np.int64)
    scores = np.array([[0.8, 0.2], [0.2, 0.1]], dtype=np.float32)
    with pytest.raises(ValueError, match="missing positive"):
        tune_per_class_thresholds(y_true, scores, [0.3, 0.5, 0.7])


def test_threshold_tuning_rejects_missing_negative_validation_class():
    y_true = np.array([[1, 1], [0, 1]], dtype=np.int64)
    scores = np.array([[0.8, 0.9], [0.2, 0.8]], dtype=np.float32)
    with pytest.raises(ValueError, match="missing negative"):
        tune_per_class_thresholds(y_true, scores, [0.3, 0.5, 0.7])


def test_threshold_tuning_rejects_nonfinite_scores():
    y_true = np.array([[1], [0]], dtype=np.int64)
    scores = np.array([[np.nan], [0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        tune_per_class_thresholds(y_true, scores, [0.5])


def test_threshold_tuning_rejects_nonbinary_targets():
    y_true = np.array([[2], [0]], dtype=np.int64)
    scores = np.array([[0.8], [0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="binary"):
        tune_per_class_thresholds(y_true, scores, [0.5])


def test_apply_thresholds_rejects_nonfinite_thresholds():
    with pytest.raises(ValueError, match="non-finite"):
        apply_thresholds(
            np.array([[0.5]], dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
        )


def test_threshold_artifact_roundtrip_preserves_metadata(tmp_path):
    path = tmp_path / "thresholds.json"
    metadata = {
        "protocol_version": THRESHOLD_PROTOCOL_VERSION,
        "selection_split": "val",
    }
    save_thresholds(path, ["a", "b"], [0.35, 0.65], metadata)

    payload = load_threshold_artifact(path, expected_classes=["a", "b"])
    assert payload["format_version"] == 2
    assert payload["thresholds"] == pytest.approx([0.35, 0.65])
    assert payload["metadata"] == metadata
    assert load_thresholds(path, ["a", "b"]) == pytest.approx([0.35, 0.65])


def test_threshold_artifact_refuses_overwrite_when_frozen(tmp_path):
    path = tmp_path / "thresholds.json"
    save_thresholds(path, ["a"], [0.5], overwrite=False)
    with pytest.raises(FileExistsError):
        save_thresholds(path, ["a"], [0.6], overwrite=False)


def _strict_threshold_payload(checkpoint: Path, val_manifest: Path) -> dict:
    return {
        "class_names": ["a"],
        "thresholds": [0.5],
        "metadata": {
            "protocol_version": THRESHOLD_PROTOCOL_VERSION,
            "selection_split": "val",
            "selection_metric": "per_class_f1",
            "checkpoint_sha256": file_sha256(checkpoint),
            "validation_manifest_sha256": file_sha256(val_manifest),
            "threshold_grid": [0.4, 0.5, 0.6],
            "validation_samples": 4,
        },
    }


def test_threshold_provenance_accepts_matching_checkpoint_and_validation_manifest(tmp_path):
    checkpoint = tmp_path / "model.pt"
    validation = tmp_path / "known_val.csv"
    checkpoint.write_bytes(b"checkpoint-a")
    validation.write_text("split\nval\n", encoding="utf-8")

    validate_threshold_provenance(
        _strict_threshold_payload(checkpoint, validation),
        checkpoint,
        validation,
    )


def test_threshold_provenance_rejects_checkpoint_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pt"
    validation = tmp_path / "known_val.csv"
    checkpoint.write_bytes(b"checkpoint-a")
    validation.write_text("split\nval\n", encoding="utf-8")
    payload = _strict_threshold_payload(checkpoint, validation)

    checkpoint.write_bytes(b"checkpoint-b")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        validate_threshold_provenance(payload, checkpoint, validation)


def test_threshold_provenance_rejects_validation_manifest_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pt"
    validation = tmp_path / "known_val.csv"
    checkpoint.write_bytes(b"checkpoint-a")
    validation.write_text("split\nval\n", encoding="utf-8")
    payload = _strict_threshold_payload(checkpoint, validation)

    validation.write_text("split\nval\nval\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation manifest SHA-256 mismatch"):
        validate_threshold_provenance(payload, checkpoint, validation)


def test_frozen_evaluation_artifact_refuses_overwrite(tmp_path):
    path = tmp_path / "test_evaluation.json"
    save_evaluation({"value": 1}, path, overwrite=False)
    with pytest.raises(FileExistsError):
        save_evaluation({"value": 2}, path, overwrite=False)
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 1


def test_rejection_metrics_expose_clear_ood_rejection_and_known_false_rejection_names():
    thresholds = np.array([0.5, 0.5], dtype=np.float32)
    known_scores = np.array([[0.8, 0.1], [0.1, 0.1]], dtype=np.float32)
    ood_scores = np.array([[0.2, 0.1], [0.9, 0.1]], dtype=np.float32)

    metrics = rejection_metrics(known_scores, ood_scores, thresholds)

    assert metrics["known_false_rejection_rate"] == pytest.approx(0.5)
    assert metrics["known_acceptance_rate"] == pytest.approx(0.5)
    assert metrics["ood_rejection_rate"] == pytest.approx(0.5)
    assert metrics["ood_recall_rejected"] == pytest.approx(0.5)
    assert metrics["ood_false_acceptance_rate"] == pytest.approx(0.5)


def test_group_metrics_include_joint_mixture_condition():
    metadata = [
        {
            "sample_type": "mix",
            "relative_db": -6.0,
            "overlap_ratio": 0.5,
        },
        {
            "sample_type": "single",
            "relative_db": float("nan"),
            "overlap_ratio": 0.0,
        },
    ]
    y_true = np.array([[1, 1], [1, 0]], dtype=np.int64)
    scores = np.array([[0.9, 0.8], [0.9, 0.1]], dtype=np.float32)
    thresholds = np.array([0.5, 0.5], dtype=np.float32)

    groups = group_metrics(metadata, y_true, scores, thresholds, ["a", "b"])

    key = "mix_condition=relative_db=-6|overlap_ratio=0.5"
    assert key in groups
    assert groups[key]["n_samples"] == 1
