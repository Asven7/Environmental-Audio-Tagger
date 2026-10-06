#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from esaudio.checkpoints import load_threshold_artifact
from esaudio.config import load_config
from esaudio.evaluate_runner import (
    THRESHOLD_PROTOCOL_VERSION,
    file_sha256 as evaluation_file_sha256,
    select_thresholds_from_validation,
    validate_threshold_provenance,
)
from esaudio.experiments import (
    EXPERIMENT_FREEZE_VERSION,
    atomic_write_json,
    file_sha256,
    json_sha256,
    official_run_matrix,
    validate_training_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select validation-only thresholds for every completed official Phase-11 run "
            "and create the pre-test experiment freeze artifact."
        )
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--manifests-dir", default="artifacts/manifests_phase5")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", default="artifacts/experiments_phase11")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    matrix = official_run_matrix(config)
    output_root = Path(args.output_root)
    plan_path = output_root / "experiment_plan.json"
    training_index_path = output_root / "training_index.json"
    threshold_index_path = output_root / "threshold_index.json"
    freeze_path = output_root / "experiment_freeze.json"
    val_manifest = Path(args.manifests_dir) / "known_val.csv"

    if freeze_path.exists():
        raise RuntimeError(
            f"Experiment is already frozen for downstream held-out evaluation: {freeze_path}"
        )
    if not plan_path.exists() or not training_index_path.exists():
        raise FileNotFoundError(
            "Official training plan/index is missing; complete all six training runs first."
        )
    if not val_manifest.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("validation_manifest_sha256") != file_sha256(val_manifest):
        raise RuntimeError("Validation manifest no longer matches the official experiment plan")

    training_records = json.loads(training_index_path.read_text(encoding="utf-8"))
    validate_training_index(training_records, expected_matrix=matrix)

    by_key = {
        (str(record["model"]), int(record["seed"])): record
        for record in training_records
    }
    threshold_records: list[dict] = []

    for model_name, seed in matrix:
        training_record = by_key[(model_name, seed)]
        run_dir = Path(training_record["run_dir"])
        checkpoint = run_dir / "best_model.pt"
        threshold_path = run_dir / "thresholds.json"
        report_path = run_dir / "threshold_selection.json"

        if threshold_path.exists() or report_path.exists():
            if not args.resume:
                raise RuntimeError(
                    f"Threshold artifacts already exist for {model_name} seed {seed}: {run_dir}"
                )
            if not threshold_path.exists() or not report_path.exists():
                raise RuntimeError(
                    f"Incomplete threshold-selection artifacts: {run_dir}"
                )
            payload = load_threshold_artifact(
                threshold_path,
                expected_classes=config["project"]["target_classes"],
            )
            validate_threshold_provenance(payload, checkpoint, val_manifest)
            selection = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"SKIP completed thresholds: {model_name} seed={seed}")
        else:
            print(f"SELECT thresholds: {model_name} seed={seed}")
            selection = select_thresholds_from_validation(
                config,
                checkpoint,
                val_manifest,
                args.audio_root,
                threshold_path,
                device_name=args.device,
                overwrite=False,
            )
            atomic_write_json(report_path, selection)

        threshold_records.append(
            {
                "model": model_name,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "checkpoint_sha256": evaluation_file_sha256(checkpoint),
                "threshold_artifact_sha256": evaluation_file_sha256(threshold_path),
                "selection": selection,
            }
        )
        atomic_write_json(threshold_index_path, threshold_records)

    freeze_core = {
        "freeze_version": EXPERIMENT_FREEZE_VERSION,
        "training_protocol_version": plan["protocol_version"],
        "threshold_protocol_version": THRESHOLD_PROTOCOL_VERSION,
        "git_commit": plan["git_commit"],
        "experiment_plan_sha256": file_sha256(plan_path),
        "training_index_sha256": file_sha256(training_index_path),
        "threshold_index_sha256": file_sha256(threshold_index_path),
        "validation_manifest_sha256": file_sha256(val_manifest),
        "runs": [
            {
                "model": item["model"],
                "seed": item["seed"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "threshold_artifact_sha256": item["threshold_artifact_sha256"],
            }
            for item in threshold_records
        ],
    }
    freeze_payload = {
        **freeze_core,
        "freeze_sha256": json_sha256(freeze_core),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(freeze_path, freeze_payload)

    print(f"Threshold selection complete: {len(threshold_records)} runs")
    print(f"Threshold index: {threshold_index_path}")
    print(f"Pre-test experiment freeze: {freeze_path}")
    print("Held-out evaluation: NOT PERFORMED")


if __name__ == "__main__":
    main()
