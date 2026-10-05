from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from esaudio.config import load_config
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
    singles = frame[frame["sample_type"].astype(str) == "single"].sort_values("_sample_id_sort")

    selected_indices: list[int] = []
    for class_index in range(num_classes):
        candidates = singles[
            singles["label_indices"].apply(lambda value: class_index in _labels(value))
        ]
        if len(candidates) < singles_per_class:
            raise RuntimeError(
                f"Not enough single examples for class index {class_index}: "
                f"need {singles_per_class}, found {len(candidates)}"
            )
        selected_indices.extend(candidates.index[:singles_per_class].tolist())

    mixtures = frame[frame["sample_type"].astype(str) == "mix"].sort_values("_sample_id_sort")
    if extra_mixtures > 0 and len(mixtures):
        selected_indices.extend(mixtures.index[:extra_mixtures].tolist())

    result = frame.loc[sorted(set(selected_indices))].drop(columns=["_sample_id_sort"])
    return result.sort_values("sample_id").reset_index(drop=True)


def _load_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["state_dict"]


def _assert_state_dict_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    if first.keys() != second.keys():
        raise RuntimeError("Same-seed checkpoints have different parameter keys")
    for key in first:
        if not torch.equal(first[key], second[key]):
            raise RuntimeError(f"Same-seed checkpoint mismatch at parameter: {key}")


def _run(
    config: dict,
    model_name: str,
    train_manifest: Path,
    val_manifest: Path,
    audio_root: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    return train_model(
        config,
        model_name,
        train_manifest,
        val_manifest,
        audio_root,
        output_dir,
        seed,
        tune_thresholds=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Phase-9 training/reproducibility without touching test data."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-dir", default="artifacts/manifests_phase5")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)
    manifest_dir = Path(args.manifest_dir)
    audio_root = Path(args.dataset_root)
    train_full = pd.read_csv(manifest_dir / "known_train.csv")
    val_full = pd.read_csv(manifest_dir / "known_val.csv")
    num_classes = len(base_config["project"]["target_classes"])

    train_tiny = _coverage_subset(
        train_full,
        num_classes,
        singles_per_class=2,
        extra_mixtures=8,
    )
    val_tiny = _coverage_subset(
        val_full,
        num_classes,
        singles_per_class=1,
        extra_mixtures=4,
    )

    print("=== Phase 9 Training / Reproducibility Smoke Test ===")
    print(f"tiny_train_samples={len(train_tiny)} tiny_val_samples={len(val_tiny)}")
    print(f"seed={args.seed}")
    print("test_manifests_used=False")
    print("threshold_tuning=False")

    with tempfile.TemporaryDirectory(prefix="esaudio_phase9_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        train_manifest = temp_dir / "known_train_tiny.csv"
        val_manifest = temp_dir / "known_val_tiny.csv"
        train_tiny.to_csv(train_manifest, index=False)
        val_tiny.to_csv(val_manifest, index=False)

        cpu_config = copy.deepcopy(base_config)
        cpu_config["training"]["device"] = "cpu"
        cpu_config["training"]["epochs"] = 2
        cpu_config["training"]["batch_size"] = 8
        cpu_config["training"]["early_stopping_patience"] = 2

        first_dir = temp_dir / "cpu_repeat_a"
        second_dir = temp_dir / "cpu_repeat_b"

        first = _run(
            cpu_config,
            "cnn",
            train_manifest,
            val_manifest,
            audio_root,
            first_dir,
            args.seed,
        )
        second = _run(
            cpu_config,
            "cnn",
            train_manifest,
            val_manifest,
            audio_root,
            second_dir,
            args.seed,
        )

        pd.testing.assert_frame_equal(
            _load_history(first_dir / "history.csv"),
            _load_history(second_dir / "history.csv"),
            check_exact=True,
        )
        _assert_state_dict_equal(
            _load_state_dict(first_dir / "best_model.pt"),
            _load_state_dict(second_dir / "best_model.pt"),
        )

        if first["thresholds_tuned"] or second["thresholds_tuned"]:
            raise RuntimeError("Threshold tuning unexpectedly occurred")
        if (first_dir / "thresholds.json").exists() or (second_dir / "thresholds.json").exists():
            raise RuntimeError("thresholds.json unexpectedly exists in Phase-9 smoke run")
        if first["best_epoch"] != second["best_epoch"]:
            raise RuntimeError("Same-seed runs selected different best epochs")
        if first["best_validation_mAP"] != second["best_validation_mAP"]:
            raise RuntimeError("Same-seed runs produced different best validation mAP")

        print(
            "CPU same-seed reproducibility: PASSED "
            f"(best_epoch={first['best_epoch']}, "
            f"best_val_mAP={first['best_validation_mAP']:.6f})"
        )
        print("Threshold isolation: PASSED")
        print(
            "Manifest fingerprints: PASSED "
            f"(train={first['train_manifest_sha256'][:12]}..., "
            f"val={first['val_manifest_sha256'][:12]}...)"
        )

        if torch.cuda.is_available() and not args.skip_cuda:
            cuda_config = copy.deepcopy(base_config)
            cuda_config["training"]["device"] = "cuda"
            cuda_config["training"]["epochs"] = 1
            cuda_config["training"]["batch_size"] = 8
            cuda_config["training"]["early_stopping_patience"] = 1

            cuda_dir = temp_dir / "cuda_crnn"
            cuda_summary = _run(
                cuda_config,
                "crnn",
                train_manifest,
                val_manifest,
                audio_root,
                cuda_dir,
                args.seed,
            )
            if cuda_summary["thresholds_tuned"]:
                raise RuntimeError("CUDA smoke run unexpectedly tuned thresholds")
            if not (cuda_dir / "best_model.pt").exists():
                raise RuntimeError("CUDA smoke run did not write a best checkpoint")
            print(
                "CUDA tiny CRNN training: PASSED "
                f"(device={torch.cuda.get_device_name(0)}, "
                f"best_epoch={cuda_summary['best_epoch']})"
            )
        elif args.skip_cuda:
            print("CUDA tiny CRNN training: SKIPPED (--skip-cuda)")
        else:
            print("CUDA tiny CRNN training: SKIPPED (CUDA unavailable)")

        summary_preview = {
            "cpu_best_epoch": first["best_epoch"],
            "cpu_best_validation_mAP": first["best_validation_mAP"],
            "cpu_epochs_ran": first["epochs_ran"],
            "cpu_stop_reason": first["stop_reason"],
            "thresholds_tuned": first["thresholds_tuned"],
        }
        print("summary=" + json.dumps(summary_preview, sort_keys=True))

    print("Phase 9 training protocol smoke test: PASSED")
    print("No test/OOD manifest was read.")
    print("No threshold tuning or frozen test evaluation was performed.")
    print("Temporary smoke-training artifacts were deleted.")


if __name__ == "__main__":
    main()
