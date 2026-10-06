#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from esaudio.config import load_config
from esaudio.experiments import (
    EXPERIMENT_PROTOCOL_VERSION,
    atomic_write_json,
    file_sha256,
    json_sha256,
    official_run_matrix,
    validate_completed_training_summary,
)
from esaudio.training import train_model


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_git() -> str:
    status = _git(["status", "--porcelain"])
    if status:
        raise RuntimeError(
            "Official experiments require a clean Git working tree. "
            "Commit the Phase-11 protocol code before training."
        )
    return _git(["rev-parse", "HEAD"])


def _require_ignored_output(path: Path) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path)],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Official experiment output must be Git-ignored: {path}. "
            "Do not store generated checkpoints in version control."
        )


def _current_plan(
    *,
    config_path: Path,
    config: dict,
    manifests_dir: Path,
    output_root: Path,
    git_commit: str,
    device: str,
) -> dict:
    train_manifest = manifests_dir / "known_train.csv"
    val_manifest = manifests_dir / "known_val.csv"
    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")

    matrix = official_run_matrix(config)
    return {
        "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "git_commit": git_commit,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "manifests_dir": str(manifests_dir),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest": str(val_manifest),
        "validation_manifest_sha256": file_sha256(val_manifest),
        "output_root": str(output_root),
        "models": ["cnn", "crnn"],
        "seeds": [13, 23, 37],
        "run_matrix": [
            {"model": model_name, "seed": seed}
            for model_name, seed in matrix
        ],
        "device": device,
    }


def _plan_signature(plan: dict) -> str:
    frozen = {
        key: value
        for key, value in plan.items()
        if key not in {"created_utc", "plan_sha256"}
    }
    return json_sha256(frozen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official six-run CNN/CRNN training matrix using train+validation "
            "only. This script never selects thresholds and never evaluates held-out data."
        )
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--manifests-dir", default="artifacts/manifests_phase5")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", default="artifacts/experiments_phase11")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed valid runs from the exact same frozen experiment plan.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    manifests_dir = Path(args.manifests_dir)
    output_root = Path(args.output_root)

    git_commit = _require_clean_git()
    _require_ignored_output(output_root)

    config = load_config(config_path)
    config["training"]["device"] = args.device
    matrix = official_run_matrix(config)

    candidate = _current_plan(
        config_path=config_path,
        config=config,
        manifests_dir=manifests_dir,
        output_root=output_root,
        git_commit=git_commit,
        device=args.device,
    )
    plan_path = output_root / "experiment_plan.json"
    index_path = output_root / "training_index.json"

    if plan_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Experiment plan already exists: {plan_path}. "
                "Use --resume only for the exact same protocol."
            )
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
        if _plan_signature(stored) != _plan_signature(candidate):
            raise RuntimeError(
                "Current environment/config/manifests do not match the stored "
                "experiment plan; refusing to resume."
            )
        plan = stored
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise RuntimeError(
                f"Output root is not empty and has no experiment plan: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        plan = {
            **candidate,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        plan["plan_sha256"] = _plan_signature(plan)
        atomic_write_json(plan_path, plan)

    train_hash = str(plan["train_manifest_sha256"])
    val_hash = str(plan["validation_manifest_sha256"])
    records: list[dict] = []

    for model_name, seed in matrix:
        run_dir = output_root / f"{model_name}_seed{seed}"
        summary_path = run_dir / "training_summary.json"
        checkpoint_path = run_dir / "best_model.pt"

        if summary_path.exists() or checkpoint_path.exists():
            if not args.resume:
                raise RuntimeError(
                    f"Run artifacts already exist for {model_name} seed {seed}: {run_dir}"
                )
            if not summary_path.exists() or not checkpoint_path.exists():
                raise RuntimeError(
                    f"Incomplete run directory cannot be resumed safely: {run_dir}. "
                    "Inspect it, remove only that incomplete generated run directory, "
                    "then rerun with --resume."
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_completed_training_summary(
                summary,
                model_name=model_name,
                seed=seed,
                train_manifest_sha256=train_hash,
                val_manifest_sha256=val_hash,
            )
            print(f"SKIP completed run: {model_name} seed={seed}")
        else:
            print(f"START run: {model_name} seed={seed}")
            summary = train_model(
                config,
                model_name,
                plan["train_manifest"],
                plan["validation_manifest"],
                args.audio_root,
                run_dir,
                seed,
                tune_thresholds=False,
            )
            validate_completed_training_summary(
                summary,
                model_name=model_name,
                seed=seed,
                train_manifest_sha256=train_hash,
                val_manifest_sha256=val_hash,
            )
            print(
                f"DONE run: {model_name} seed={seed} "
                f"best_epoch={summary['best_epoch']} "
                f"best_val_mAP={summary['best_validation_mAP']:.6f}"
            )

        records.append(
            {
                "model": model_name,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "training": summary,
                "checkpoint_sha256": file_sha256(checkpoint_path),
            }
        )
        atomic_write_json(index_path, records)

    print(f"Completed official training matrix: {len(records)} runs")
    print(f"Experiment plan: {plan_path}")
    print(f"Training index: {index_path}")
    print("Threshold selection: NOT PERFORMED")
    print("Held-out evaluation: NOT PERFORMED")


if __name__ == "__main__":
    main()
