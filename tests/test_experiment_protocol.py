from __future__ import annotations

from pathlib import Path

import pytest

from esaudio.experiments import (
    OFFICIAL_MODELS,
    OFFICIAL_SEEDS,
    aggregate_validation_records,
    official_run_matrix,
    validate_completed_training_summary,
    validate_training_index,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return {
        "training": {
            "seeds": [13, 23, 37],
        }
    }


def test_official_matrix_is_exactly_two_models_times_three_seeds():
    matrix = official_run_matrix(_config())
    assert len(matrix) == 6
    assert matrix == [
        ("cnn", 13),
        ("cnn", 23),
        ("cnn", 37),
        ("crnn", 13),
        ("crnn", 23),
        ("crnn", 37),
    ]
    assert tuple(OFFICIAL_MODELS) == ("cnn", "crnn")
    assert tuple(OFFICIAL_SEEDS) == (13, 23, 37)


def test_official_matrix_rejects_changed_seed_protocol():
    config = _config()
    config["training"]["seeds"] = [1, 2, 3]
    with pytest.raises(ValueError, match="frozen"):
        official_run_matrix(config)


def test_training_summary_validation_requires_no_training_time_threshold_tuning():
    summary = {
        "model_name": "cnn",
        "seed": 13,
        "thresholds_tuned": True,
        "train_manifest_sha256": "train",
        "val_manifest_sha256": "val",
        "best_validation_mAP": 0.5,
        "best_epoch": 2,
        "epochs_ran": 4,
    }
    with pytest.raises(ValueError, match="must not tune thresholds"):
        validate_completed_training_summary(
            summary,
            model_name="cnn",
            seed=13,
            train_manifest_sha256="train",
            val_manifest_sha256="val",
        )


def test_training_index_requires_all_six_unique_runs():
    records = [
        {"model": model, "seed": seed}
        for model, seed in official_run_matrix(_config())
    ]
    validate_training_index(
        records,
        expected_matrix=official_run_matrix(_config()),
    )

    with pytest.raises(ValueError, match="matrix mismatch"):
        validate_training_index(
            records[:-1],
            expected_matrix=official_run_matrix(_config()),
        )


def _training_records():
    records = []
    for model in ("cnn", "crnn"):
        for offset, seed in enumerate((13, 23, 37)):
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "training": {
                        "best_validation_mAP": 0.50 + 0.01 * offset,
                        "best_epoch": 10 + offset,
                        "epochs_ran": 15 + offset,
                        "elapsed_seconds": 100.0 + 10.0 * offset,
                    },
                }
            )
    return records


def _threshold_records():
    records = []
    for model in ("cnn", "crnn"):
        for offset, seed in enumerate((13, 23, 37)):
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "selection": {
                        "class_names": ["a", "b"],
                        "thresholds": [0.4 + 0.05 * offset, 0.6],
                        "validation_metrics_tuned_thresholds": {
                            "f1_micro": 0.60 + 0.01 * offset,
                            "f1_macro": 0.55 + 0.01 * offset,
                        },
                    },
                }
            )
    return records


def test_validation_aggregation_reports_mean_and_sample_std():
    summary = aggregate_validation_records(
        _training_records(),
        _threshold_records(),
    )

    assert summary["cnn"]["best_validation_mAP"]["n"] == 3
    assert summary["cnn"]["best_validation_mAP"]["mean"] == pytest.approx(0.51)
    assert summary["cnn"]["best_validation_mAP"]["std"] == pytest.approx(0.01)
    assert summary["crnn"]["validation_f1_micro_tuned_thresholds"]["mean"] == pytest.approx(0.61)
    assert summary["cnn"]["thresholds_by_class"]["a"]["mean"] == pytest.approx(0.45)


@pytest.mark.parametrize(
    "script_name",
    [
        "run_official_experiments.py",
        "select_experiment_thresholds.py",
        "summarize_validation_experiments.py",
    ],
)
def test_phase11_scripts_do_not_reference_heldout_manifests_or_evaluator(script_name):
    source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8").lower()
    forbidden = (
        "known_test",
        "ood_test",
        "evaluate_checkpoint",
        "run_frozen_evaluation",
    )
    for token in forbidden:
        assert token not in source
