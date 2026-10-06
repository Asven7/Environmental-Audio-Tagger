from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .evaluate_runner import FROZEN_EVALUATION_PROTOCOL_VERSION
from .experiments import (
    EXPERIMENT_FREEZE_VERSION,
    OFFICIAL_MODELS,
    OFFICIAL_SEEDS,
    atomic_write_json,
    file_sha256,
    json_sha256,
)


HEADLINE_KNOWN_METRICS = (
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "hamming_loss",
    "mAP",
)

REJECTION_METRICS = (
    "known_false_rejection_rate",
    "known_acceptance_rate",
    "ood_rejection_rate",
    "ood_false_acceptance_rate",
)

PER_CLASS_METRICS = (
    "precision",
    "recall",
    "f1",
    "average_precision",
)

GROUP_METRICS = (
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "hamming_loss",
    "mAP",
)


def _aggregate(values: Iterable[float]) -> dict:
    array = np.asarray([float(x) for x in values], dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot aggregate empty or non-finite values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _freeze_core(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"freeze_sha256", "created_utc"}
    }


def _validate_freeze_payload(payload: dict) -> list[dict]:
    if payload.get("freeze_version") != EXPERIMENT_FREEZE_VERSION:
        raise ValueError(
            f"Unexpected experiment freeze version: {payload.get('freeze_version')!r}"
        )
    expected_sha = json_sha256(_freeze_core(payload))
    if payload.get("freeze_sha256") != expected_sha:
        raise ValueError("experiment_freeze.json self-hash mismatch")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Freeze payload runs must be a list")

    observed = {(str(x["model"]), int(x["seed"])) for x in runs}
    expected = {
        (model_name, seed)
        for model_name in OFFICIAL_MODELS
        for seed in OFFICIAL_SEEDS
    }
    if observed != expected or len(runs) != len(expected):
        raise ValueError(
            f"Freeze run matrix mismatch: expected {sorted(expected)}, "
            f"got {sorted(observed)}"
        )
    return runs


def _load_verified_run(
    output_root: Path,
    frozen_run: dict,
    *,
    expected_validation_manifest_sha256: str,
) -> dict:
    model_name = str(frozen_run["model"])
    seed = int(frozen_run["seed"])
    run_name = f"{model_name}_seed{seed}"
    run_dir = output_root / run_name

    checkpoint_path = run_dir / "best_model.pt"
    thresholds_path = run_dir / "thresholds.json"
    evaluation_path = run_dir / "test_evaluation.json"
    lock_path = run_dir / ".frozen_evaluation.lock.json"

    for path in (
        checkpoint_path,
        thresholds_path,
        evaluation_path,
        lock_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")

    checkpoint_sha = file_sha256(checkpoint_path)
    thresholds_sha = file_sha256(thresholds_path)
    evaluation_sha = file_sha256(evaluation_path)

    if checkpoint_sha != frozen_run["checkpoint_sha256"]:
        raise ValueError(f"{run_name}: checkpoint no longer matches experiment freeze")
    if thresholds_sha != frozen_run["threshold_artifact_sha256"]:
        raise ValueError(f"{run_name}: thresholds no longer match experiment freeze")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("protocol_version") != FROZEN_EVALUATION_PROTOCOL_VERSION:
        raise ValueError(f"{run_name}: unexpected frozen-evaluation protocol")
    if lock.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{run_name}: lock/checkpoint SHA-256 mismatch")
    if lock.get("threshold_artifact_sha256") != thresholds_sha:
        raise ValueError(f"{run_name}: lock/threshold SHA-256 mismatch")
    if lock.get("evaluation_artifact_sha256") != evaluation_sha:
        raise ValueError(f"{run_name}: evaluation artifact SHA-256 mismatch")
    if lock.get("validation_manifest_sha256") != expected_validation_manifest_sha256:
        raise ValueError(f"{run_name}: validation manifest SHA-256 mismatch")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    provenance = evaluation.get("provenance", {})
    if provenance.get("protocol_version") != FROZEN_EVALUATION_PROTOCOL_VERSION:
        raise ValueError(f"{run_name}: evaluation provenance protocol mismatch")
    if provenance.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{run_name}: evaluation/checkpoint SHA-256 mismatch")
    if provenance.get("threshold_artifact_sha256") != thresholds_sha:
        raise ValueError(f"{run_name}: evaluation/threshold SHA-256 mismatch")
    if provenance.get("validation_manifest_sha256") != expected_validation_manifest_sha256:
        raise ValueError(f"{run_name}: evaluation validation SHA-256 mismatch")
    if provenance.get("model_name") != model_name:
        raise ValueError(f"{run_name}: model-name provenance mismatch")
    if int(provenance.get("experiment_seed", -1)) != seed:
        raise ValueError(f"{run_name}: experiment-seed provenance mismatch")

    known_manifest_sha = provenance.get("known_manifest_sha256")
    ood_manifest_sha = provenance.get("ood_manifest_sha256")
    if not known_manifest_sha or not ood_manifest_sha:
        raise ValueError(f"{run_name}: held-out manifest provenance is incomplete")
    if lock.get("known_test_manifest_sha256") != known_manifest_sha:
        raise ValueError(f"{run_name}: lock/known-test manifest SHA-256 mismatch")
    if lock.get("ood_test_manifest_sha256") != ood_manifest_sha:
        raise ValueError(f"{run_name}: lock/OOD-test manifest SHA-256 mismatch")

    if "known" not in evaluation or "rejection" not in evaluation:
        raise ValueError(f"{run_name}: evaluation lacks known/rejection metrics")
    if int(evaluation.get("n_known", 0)) <= 0:
        raise ValueError(f"{run_name}: n_known must be positive")
    if int(evaluation.get("n_ood", 0)) <= 0:
        raise ValueError(f"{run_name}: n_ood must be positive")

    return {
        "model": model_name,
        "seed": seed,
        "run_name": run_name,
        "evaluation": evaluation,
        "evaluation_sha256": evaluation_sha,
        "lock_sha256": file_sha256(lock_path),
        "known_manifest_sha256": known_manifest_sha,
        "ood_manifest_sha256": ood_manifest_sha,
    }


def _ensure_constant(records: list[dict], key: str):
    values = [record[key] for record in records]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"Frozen runs disagree on {key}")
    return first


def aggregate_frozen_experiments(output_root: str | Path) -> dict:
    output_root = Path(output_root)
    freeze_path = output_root / "experiment_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError(f"Experiment freeze not found: {freeze_path}")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_runs = _validate_freeze_payload(freeze)
    expected_validation_sha = str(freeze["validation_manifest_sha256"])

    verified = [
        _load_verified_run(
            output_root,
            frozen_run,
            expected_validation_manifest_sha256=expected_validation_sha,
        )
        for frozen_run in frozen_runs
    ]

    known_manifest_sha = _ensure_constant(verified, "known_manifest_sha256")
    ood_manifest_sha = _ensure_constant(verified, "ood_manifest_sha256")

    n_known_values = [
        int(item["evaluation"]["n_known"])
        for item in verified
    ]
    n_ood_values = [
        int(item["evaluation"]["n_ood"])
        for item in verified
    ]
    if len(set(n_known_values)) != 1:
        raise ValueError("Frozen runs disagree on n_known")
    if len(set(n_ood_values)) != 1:
        raise ValueError("Frozen runs disagree on n_ood")

    all_class_names = [
        tuple(item["evaluation"]["provenance"]["class_names"])
        for item in verified
    ]
    if len(set(all_class_names)) != 1:
        raise ValueError("Frozen runs disagree on class order")
    class_names = list(all_class_names[0])

    summary_models: dict[str, dict] = {}

    for model_name in OFFICIAL_MODELS:
        model_records = sorted(
            [item for item in verified if item["model"] == model_name],
            key=lambda item: item["seed"],
        )
        seeds = [item["seed"] for item in model_records]
        if tuple(seeds) != OFFICIAL_SEEDS:
            raise ValueError(
                f"{model_name}: expected seeds {list(OFFICIAL_SEEDS)}, got {seeds}"
            )

        known_summary = {
            metric: _aggregate(
                item["evaluation"]["known"][metric]
                for item in model_records
            )
            for metric in HEADLINE_KNOWN_METRICS
        }
        rejection_summary = {
            metric: _aggregate(
                item["evaluation"]["rejection"][metric]
                for item in model_records
            )
            for metric in REJECTION_METRICS
        }

        per_class: dict[str, dict] = {}
        for class_name in class_names:
            per_class[class_name] = {
                metric: _aggregate(
                    item["evaluation"]["known"]["per_class"][class_name][metric]
                    for item in model_records
                )
                for metric in PER_CLASS_METRICS
            }

        group_names = [
            set(item["evaluation"]["groups"].keys())
            for item in model_records
        ]
        if any(names != group_names[0] for names in group_names[1:]):
            raise ValueError(f"{model_name}: group names differ across seeds")

        groups: dict[str, dict] = {}
        for group_name in sorted(group_names[0]):
            sample_counts = [
                int(item["evaluation"]["groups"][group_name]["n_samples"])
                for item in model_records
            ]
            if len(set(sample_counts)) != 1:
                raise ValueError(
                    f"{model_name}/{group_name}: group sample count differs across seeds"
                )
            groups[group_name] = {
                "n_samples": sample_counts[0],
                "metrics": {
                    metric: _aggregate(
                        item["evaluation"]["groups"][group_name][metric]
                        for item in model_records
                    )
                    for metric in GROUP_METRICS
                },
            }

        summary_models[model_name] = {
            "seeds": seeds,
            "known": known_summary,
            "rejection": rejection_summary,
            "per_class": per_class,
            "groups": groups,
        }

    return {
        "scope": "frozen_heldout",
        "aggregation": "mean_sample_std_min_max_across_3_predeclared_seeds",
        "test_metrics_included": True,
        "retuning_performed": False,
        "freeze_version": freeze["freeze_version"],
        "freeze_sha256": freeze["freeze_sha256"],
        "known_test_manifest_sha256": known_manifest_sha,
        "ood_test_manifest_sha256": ood_manifest_sha,
        "n_known": n_known_values[0],
        "n_ood": n_ood_values[0],
        "class_names": class_names,
        "verified_runs": [
            {
                "model": item["model"],
                "seed": item["seed"],
                "evaluation_sha256": item["evaluation_sha256"],
                "lock_sha256": item["lock_sha256"],
            }
            for item in sorted(
                verified,
                key=lambda item: (
                    OFFICIAL_MODELS.index(item["model"]),
                    item["seed"],
                ),
            )
        ],
        "models": summary_models,
    }


def write_headline_csv(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for model_name, model_summary in summary["models"].items():
        for section_name in ("known", "rejection"):
            for metric_name, stats in model_summary[section_name].items():
                rows.append(
                    {
                        "model": model_name,
                        "section": section_name,
                        "metric": metric_name,
                        **stats,
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "section",
                "metric",
                "n",
                "mean",
                "std",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_per_class_csv(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for model_name, model_summary in summary["models"].items():
        for class_name, metrics in model_summary["per_class"].items():
            for metric_name, stats in metrics.items():
                rows.append(
                    {
                        "model": model_name,
                        "class_name": class_name,
                        "metric": metric_name,
                        **stats,
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "class_name",
                "metric",
                "n",
                "mean",
                "std",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_groups_csv(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for model_name, model_summary in summary["models"].items():
        for group_name, group in model_summary["groups"].items():
            for metric_name, stats in group["metrics"].items():
                rows.append(
                    {
                        "model": model_name,
                        "group": group_name,
                        "n_samples": group["n_samples"],
                        "metric": metric_name,
                        **stats,
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "group",
                "n_samples",
                "metric",
                "n",
                "mean",
                "std",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
