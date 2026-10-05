#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.config import load_config
from esaudio.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible CNN/CRNN TRAINING seeds using train+validation only. "
            "Test/OOD evaluation is intentionally deferred to the frozen-evaluation phase."
        )
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--manifests-dir", default="artifacts/manifests_phase5")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn", "crnn"],
        choices=["cnn", "crnn"],
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device

    seeds = args.seeds or [int(x) for x in config["training"]["seeds"]]
    manifests_dir = Path(args.manifests_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_manifest = manifests_dir / "known_train.csv"
    val_manifest = manifests_dir / "known_val.csv"
    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")

    summaries = []
    for model_name in args.models:
        for seed in seeds:
            run_dir = output_root / f"{model_name}_seed{seed}"
            train_summary = train_model(
                config,
                model_name,
                train_manifest,
                val_manifest,
                args.audio_root,
                run_dir,
                seed,
                tune_thresholds=False,
            )
            summaries.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "training": train_summary,
                }
            )

    index_path = output_root / "training_index.json"
    index_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Completed {len(summaries)} train/validation runs.")
    print(f"Training index: {index_path}")
    print("Test/OOD evaluation: NOT PERFORMED")
    print("Threshold tuning: NOT PERFORMED")


if __name__ == "__main__":
    main()
