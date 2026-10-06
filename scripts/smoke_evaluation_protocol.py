from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from esaudio.checkpoints import load_threshold_artifact
from esaudio.config import load_config
from esaudio.evaluate_runner import (
    evaluate_checkpoint,
    select_thresholds_from_validation,
    validate_threshold_provenance,
)
from esaudio.manifests import parse_json_list
from esaudio.training import train_model


def _labels(value) -> list[int]:
    return [int(x) for x in parse_json_list(value)]


def _coverage_subset(
    frame: pd.DataFrame,
    num_classes: int,
    *,
    singles_per_class: int,
    extra_mixtures: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["_sample_id_sort"] = frame["sample_id"].astype(str)
    singles = frame[
        frame["sample_type"].astype(str) == "single"
    ].sort_values("_sample_id_sort")

    selected_indices: list[int] = []
    for class_index in range(num_classes):
        candidates = singles[
            singles["label_indices"].apply(
                lambda value: class_index in _labels(value)
            )
        ]
        if len(candidates) < singles_per_class:
            raise RuntimeError(
                f"Not enough examples for class {class_index}: "
                f"need {singles_per_class}, found {len(candidates)}"
            )
        selected_indices.extend(
            candidates.index[:singles_per_class].tolist()
        )

    mixtures = frame[
        frame["sample_type"].astype(str) == "mix"
    ].sort_values("_sample_id_sort")
    if extra_mixtures > 0:
        selected_indices.extend(
            mixtures.index[:extra_mixtures].tolist()
        )

    return (
        frame.loc[sorted(set(selected_indices))]
        .drop(columns=["_sample_id_sort"])
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test validation threshold selection and strict provenance "
            "without reading known_test/ood_test."
        )
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--manifest-dir",
        default="artifacts/manifests_phase5",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    base_config = load_config(args.config)
    manifest_dir = Path(args.manifest_dir)
    audio_root = Path(args.dataset_root)
    num_classes = len(base_config["project"]["target_classes"])

    train_full = pd.read_csv(manifest_dir / "known_train.csv")
    val_full = pd.read_csv(manifest_dir / "known_val.csv")
    ood_val_full = pd.read_csv(manifest_dir / "ood_val.csv")

    train_tiny = _coverage_subset(
        train_full,
        num_classes,
        singles_per_class=1,
        extra_mixtures=4,
    )
    val_tiny = _coverage_subset(
        val_full,
        num_classes,
        singles_per_class=1,
        extra_mixtures=4,
    )
    ood_val_tiny = ood_val_full.head(min(8, len(ood_val_full))).copy()

    print("=== Phase 10 Threshold / Frozen-Evaluation Protocol Smoke Test ===")
    print(f"tiny_train_samples={len(train_tiny)}")
    print(f"tiny_val_samples={len(val_tiny)}")
    print(f"tiny_ood_val_samples={len(ood_val_tiny)}")
    print("known_test_read=False")
    print("ood_test_read=False")

    with tempfile.TemporaryDirectory(prefix="esaudio_phase10_") as temp_name:
        temp = Path(temp_name)
        train_manifest = temp / "known_train_tiny.csv"
        val_manifest = temp / "known_val_tiny.csv"
        ood_val_manifest = temp / "ood_val_tiny.csv"
        train_tiny.to_csv(train_manifest, index=False)
        val_tiny.to_csv(val_manifest, index=False)
        ood_val_tiny.to_csv(ood_val_manifest, index=False)

        config = copy.deepcopy(base_config)
        config["training"]["device"] = "cpu"
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 8
        config["training"]["early_stopping_patience"] = 1

        run_dir = temp / "run"
        train_summary = train_model(
            config,
            "cnn",
            train_manifest,
            val_manifest,
            audio_root,
            run_dir,
            args.seed,
            tune_thresholds=False,
        )
        checkpoint = run_dir / "best_model.pt"
        thresholds = run_dir / "thresholds.json"

        selection = select_thresholds_from_validation(
            config,
            checkpoint,
            val_manifest,
            audio_root,
            thresholds,
            device_name="cpu",
            overwrite=False,
        )
        payload = load_threshold_artifact(
            thresholds,
            expected_classes=config["project"]["target_classes"],
        )
        validate_threshold_provenance(
            payload,
            checkpoint,
            val_manifest,
        )

        print(
            "Validation threshold selection: PASSED "
            f"(classes={len(selection['thresholds'])})"
        )
        print("Threshold provenance binding: PASSED")

        val_result = evaluate_checkpoint(
            config,
            checkpoint,
            thresholds,
            val_manifest,
            audio_root,
            split="val",
            ood_manifest=ood_val_manifest,
            device_name="cpu",
            require_threshold_provenance=True,
            validation_manifest_for_thresholds=val_manifest,
        )
        if "rejection" not in val_result:
            raise RuntimeError("Validation-safe OOD rejection plumbing did not run")
        print("Strict validation-only evaluation: PASSED")
        print("OOD validation rejection plumbing: PASSED")

        tampered = temp / "tampered_model.pt"
        shutil.copyfile(checkpoint, tampered)
        with tampered.open("ab") as handle:
            handle.write(b"phase10-tamper")
        try:
            validate_threshold_provenance(
                payload,
                tampered,
                val_manifest,
            )
        except ValueError:
            print("Checkpoint/threshold mismatch rejection: PASSED")
        else:
            raise RuntimeError(
                "Threshold provenance failed to reject a modified checkpoint"
            )

        print(
            "selection_summary="
            f"best_epoch={train_summary['best_epoch']} "
            f"validation_samples={selection['validation_samples']}"
        )

    print("Phase 10 protocol smoke test: PASSED")
    print("Known test manifest was NOT read.")
    print("OOD test manifest was NOT read.")
    print("Temporary smoke artifacts were deleted.")


if __name__ == "__main__":
    main()
