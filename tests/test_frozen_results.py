from __future__ import annotations

import json
from pathlib import Path

import pytest

from esaudio.experiments import (
    EXPERIMENT_FREEZE_VERSION,
    OFFICIAL_MODELS,
    OFFICIAL_SEEDS,
    file_sha256,
    json_sha256,
)
from esaudio.evaluate_runner import FROZEN_EVALUATION_PROTOCOL_VERSION
from esaudio.frozen_results import aggregate_frozen_experiments


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metric_block(offset: float) -> dict:
    return {
        "precision_micro": 0.60 + offset,
        "recall_micro": 0.61 + offset,
        "f1_micro": 0.605 + offset,
        "precision_macro": 0.55 + offset,
        "recall_macro": 0.56 + offset,
        "f1_macro": 0.555 + offset,
        "hamming_loss": 0.20 - offset,
        "mAP": 0.65 + offset,
        "per_class": {
            "a": {
                "precision": 0.5 + offset,
                "recall": 0.6 + offset,
                "f1": 0.55 + offset,
                "average_precision": 0.7 + offset,
                "threshold": 0.4,
            },
            "b": {
                "precision": 0.6 + offset,
                "recall": 0.7 + offset,
                "f1": 0.65 + offset,
                "average_precision": 0.8 + offset,
                "threshold": 0.6,
            },
        },
    }


def _group_block(offset: float) -> dict:
    block = _metric_block(offset).copy()
    block.pop("per_class")
    block["n_samples"] = 5
    return block


def _build_fixture(root: Path) -> None:
    frozen_runs = []
    known_test_hash = "known-test-hash"
    ood_test_hash = "ood-test-hash"
    validation_hash = "validation-hash"

    for model_index, model in enumerate(OFFICIAL_MODELS):
        for seed_index, seed in enumerate(OFFICIAL_SEEDS):
            run_dir = root / f"{model}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = run_dir / "best_model.pt"
            thresholds = run_dir / "thresholds.json"
            checkpoint.write_bytes(f"{model}-{seed}-checkpoint".encode())
            thresholds.write_text(
                json.dumps({"thresholds": [0.4, 0.6]}),
                encoding="utf-8",
            )
            checkpoint_sha = file_sha256(checkpoint)
            thresholds_sha = file_sha256(thresholds)

            offset = model_index * 0.05 + seed_index * 0.01
            evaluation = {
                "known": _metric_block(offset),
                "groups": {
                    "sample_type=single": _group_block(offset),
                },
                "known_loss": 0.5,
                "known_loss_unweighted_bce": 0.5,
                "n_known": 100,
                "rejection": {
                    "known_false_rejection_rate": 0.10 + offset,
                    "known_acceptance_rate": 0.90 - offset,
                    "ood_rejection_rate": 0.70 + offset,
                    "ood_recall_rejected": 0.70 + offset,
                    "ood_false_acceptance_rate": 0.30 - offset,
                },
                "n_ood": 20,
                "provenance": {
                    "protocol_version": FROZEN_EVALUATION_PROTOCOL_VERSION,
                    "checkpoint_sha256": checkpoint_sha,
                    "threshold_artifact_sha256": thresholds_sha,
                    "known_manifest_sha256": known_test_hash,
                    "ood_manifest_sha256": ood_test_hash,
                    "validation_manifest_sha256": validation_hash,
                    "class_names": ["a", "b"],
                    "thresholds": [0.4, 0.6],
                    "model_name": model,
                    "experiment_seed": seed,
                },
            }
            evaluation_path = run_dir / "test_evaluation.json"
            _write_json(evaluation_path, evaluation)

            lock = {
                "protocol_version": FROZEN_EVALUATION_PROTOCOL_VERSION,
                "checkpoint_sha256": checkpoint_sha,
                "threshold_artifact_sha256": thresholds_sha,
                "validation_manifest_sha256": validation_hash,
                "known_test_manifest_sha256": known_test_hash,
                "ood_test_manifest_sha256": ood_test_hash,
                "evaluation_artifact_sha256": file_sha256(evaluation_path),
            }
            _write_json(run_dir / ".frozen_evaluation.lock.json", lock)

            frozen_runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_sha,
                    "threshold_artifact_sha256": thresholds_sha,
                }
            )

    freeze_core = {
        "freeze_version": EXPERIMENT_FREEZE_VERSION,
        "training_protocol_version": "cnn_crnn_multiseed_v1",
        "threshold_protocol_version": "per_class_validation_f1_v1",
        "git_commit": "deadbeef",
        "experiment_plan_sha256": "plan",
        "training_index_sha256": "training",
        "threshold_index_sha256": "threshold",
        "validation_manifest_sha256": validation_hash,
        "runs": frozen_runs,
    }
    freeze = {
        **freeze_core,
        "freeze_sha256": json_sha256(freeze_core),
        "created_utc": "2026-10-06T00:00:00+00:00",
    }
    _write_json(root / "experiment_freeze.json", freeze)


def test_aggregation_verifies_and_aggregates_exact_six_run_matrix(tmp_path):
    _build_fixture(tmp_path)
    summary = aggregate_frozen_experiments(tmp_path)

    assert summary["scope"] == "frozen_heldout"
    assert summary["test_metrics_included"] is True
    assert summary["retuning_performed"] is False
    assert len(summary["verified_runs"]) == 6
    assert summary["n_known"] == 100
    assert summary["n_ood"] == 20

    assert summary["models"]["cnn"]["known"]["mAP"]["n"] == 3
    assert summary["models"]["cnn"]["known"]["mAP"]["mean"] == pytest.approx(0.66)
    assert summary["models"]["cnn"]["known"]["mAP"]["std"] == pytest.approx(0.01)

    assert summary["models"]["crnn"]["known"]["mAP"]["mean"] == pytest.approx(0.71)
    assert summary["models"]["cnn"]["rejection"]["ood_rejection_rate"]["mean"] == pytest.approx(0.71)
    assert summary["models"]["cnn"]["per_class"]["a"]["f1"]["mean"] == pytest.approx(0.56)
    assert summary["models"]["cnn"]["groups"]["sample_type=single"]["n_samples"] == 5


def test_aggregation_rejects_modified_evaluation_artifact(tmp_path):
    _build_fixture(tmp_path)
    path = tmp_path / "cnn_seed13" / "test_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["known"]["mAP"] = 0.999
    _write_json(path, payload)

    with pytest.raises(ValueError, match="evaluation artifact SHA-256 mismatch"):
        aggregate_frozen_experiments(tmp_path)


def test_aggregation_rejects_checkpoint_modified_after_freeze(tmp_path):
    _build_fixture(tmp_path)
    path = tmp_path / "cnn_seed13" / "best_model.pt"
    path.write_bytes(b"modified")

    with pytest.raises(ValueError, match="checkpoint no longer matches"):
        aggregate_frozen_experiments(tmp_path)


def test_aggregation_rejects_incomplete_matrix(tmp_path):
    _build_fixture(tmp_path)
    freeze_path = tmp_path / "experiment_freeze.json"
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    payload["runs"] = payload["runs"][:-1]
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"freeze_sha256", "created_utc"}
    }
    payload["freeze_sha256"] = json_sha256(core)
    _write_json(freeze_path, payload)

    with pytest.raises(ValueError, match="run matrix mismatch"):
        aggregate_frozen_experiments(tmp_path)


def test_aggregation_script_performs_no_inference_or_retuning():
    source = (
        ROOT / "scripts" / "summarize_frozen_experiments.py"
    ).read_text(encoding="utf-8").lower()

    forbidden = (
        "evaluate_checkpoint",
        "select_thresholds_from_validation",
        "tune_per_class_thresholds",
        "audiomanifestdataset",
        "dataloader",
        "torch.",
    )
    for token in forbidden:
        assert token not in source
