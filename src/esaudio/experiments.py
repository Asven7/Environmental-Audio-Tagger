from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


OFFICIAL_MODELS = ("cnn", "crnn")
OFFICIAL_SEEDS = (13, 23, 37)
EXPERIMENT_PROTOCOL_VERSION = "cnn_crnn_multiseed_v1"
EXPERIMENT_FREEZE_VERSION = "cnn_crnn_validation_freeze_v1"


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def atomic_write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


def official_run_matrix(config: dict) -> list[tuple[str, int]]:
    configured_seeds = tuple(int(x) for x in config["training"]["seeds"])
    if configured_seeds != OFFICIAL_SEEDS:
        raise ValueError(
            "Official experiment seeds are frozen as "
            f"{list(OFFICIAL_SEEDS)}, got {list(configured_seeds)}"
        )
    return [
        (model_name, seed)
        for model_name in OFFICIAL_MODELS
        for seed in OFFICIAL_SEEDS
    ]


def validate_completed_training_summary(
    summary: dict,
    *,
    model_name: str,
    seed: int,
    train_manifest_sha256: str,
    val_manifest_sha256: str,
) -> None:
    if summary.get("model_name") != model_name:
        raise ValueError(
            f"Training summary model mismatch: expected {model_name!r}, "
            f"got {summary.get('model_name')!r}"
        )
    if int(summary.get("seed", -1)) != int(seed):
        raise ValueError(
            f"Training summary seed mismatch: expected {seed}, "
            f"got {summary.get('seed')!r}"
        )
    if bool(summary.get("thresholds_tuned", True)):
        raise ValueError(
            "Official Phase-11 training must not tune thresholds inside training"
        )
    if summary.get("train_manifest_sha256") != train_manifest_sha256:
        raise ValueError("Training summary train-manifest SHA-256 mismatch")
    if summary.get("val_manifest_sha256") != val_manifest_sha256:
        raise ValueError("Training summary validation-manifest SHA-256 mismatch")
    if not math.isfinite(float(summary.get("best_validation_mAP", float("nan")))):
        raise ValueError("Training summary best_validation_mAP must be finite")
    if int(summary.get("best_epoch", 0)) <= 0:
        raise ValueError("Training summary best_epoch must be positive")
    if int(summary.get("epochs_ran", 0)) <= 0:
        raise ValueError("Training summary epochs_ran must be positive")


def validate_training_index(
    records: list[dict],
    *,
    expected_matrix: Iterable[tuple[str, int]],
) -> None:
    expected = {(model, int(seed)) for model, seed in expected_matrix}
    observed = {
        (str(record["model"]), int(record["seed"]))
        for record in records
    }
    if observed != expected:
        raise ValueError(
            f"Training index matrix mismatch: expected {sorted(expected)}, "
            f"got {sorted(observed)}"
        )
    if len(records) != len(expected):
        raise ValueError("Training index contains duplicate model/seed records")


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _aggregate_numeric(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot aggregate empty or non-finite values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": _sample_std([float(x) for x in array]),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_validation_records(
    training_records: list[dict],
    threshold_records: list[dict],
) -> dict:
    threshold_by_key = {
        (str(item["model"]), int(item["seed"])): item
        for item in threshold_records
    }
    if len(threshold_by_key) != len(threshold_records):
        raise ValueError("Threshold index contains duplicate model/seed records")

    result: dict[str, dict] = {}
    for model_name in OFFICIAL_MODELS:
        model_training = [
            item for item in training_records
            if str(item["model"]) == model_name
        ]
        if not model_training:
            raise ValueError(f"No training records found for model {model_name!r}")

        seeds = sorted(int(item["seed"]) for item in model_training)
        if tuple(seeds) != OFFICIAL_SEEDS:
            raise ValueError(
                f"Model {model_name!r} does not contain the official seeds "
                f"{list(OFFICIAL_SEEDS)}"
            )

        best_maps: list[float] = []
        best_epochs: list[float] = []
        epochs_ran: list[float] = []
        elapsed: list[float] = []
        f1_micro: list[float] = []
        f1_macro: list[float] = []
        thresholds_by_class: dict[str, list[float]] = {}

        for item in sorted(model_training, key=lambda x: int(x["seed"])):
            summary = item["training"]
            key = (model_name, int(item["seed"]))
            if key not in threshold_by_key:
                raise ValueError(f"Missing threshold record for {key}")

            threshold_record = threshold_by_key[key]
            selection = threshold_record["selection"]
            metrics = selection["validation_metrics_tuned_thresholds"]

            best_maps.append(float(summary["best_validation_mAP"]))
            best_epochs.append(float(summary["best_epoch"]))
            epochs_ran.append(float(summary["epochs_ran"]))
            elapsed.append(float(summary["elapsed_seconds"]))
            f1_micro.append(float(metrics["f1_micro"]))
            f1_macro.append(float(metrics["f1_macro"]))

            class_names = list(selection["class_names"])
            threshold_values = [float(x) for x in selection["thresholds"]]
            if len(class_names) != len(threshold_values):
                raise ValueError("Threshold selection class/value length mismatch")
            for class_name, value in zip(class_names, threshold_values):
                thresholds_by_class.setdefault(str(class_name), []).append(value)

        result[model_name] = {
            "seeds": seeds,
            "best_validation_mAP": _aggregate_numeric(best_maps),
            "best_epoch": _aggregate_numeric(best_epochs),
            "epochs_ran": _aggregate_numeric(epochs_ran),
            "elapsed_seconds": _aggregate_numeric(elapsed),
            "validation_f1_micro_tuned_thresholds": _aggregate_numeric(f1_micro),
            "validation_f1_macro_tuned_thresholds": _aggregate_numeric(f1_macro),
            "thresholds_by_class": {
                class_name: _aggregate_numeric(values)
                for class_name, values in thresholds_by_class.items()
            },
        }

    return result
