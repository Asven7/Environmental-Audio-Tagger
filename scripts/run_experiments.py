#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.config import load_config
from esaudio.evaluate_runner import evaluate_checkpoint, save_evaluation
from esaudio.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible CNN/CRNN experiment seeds")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--manifests-dir", default="artifacts/manifests")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--models", nargs="+", default=["cnn", "crnn"], choices=["cnn", "crnn"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    seeds = args.seeds or [int(x) for x in config["training"]["seeds"]]
    manifests_dir = Path(args.manifests_dir)
    output_root = Path(args.output_root)
    summaries = []

    for model_name in args.models:
        for seed in seeds:
            run_dir = output_root / f"{model_name}_seed{seed}"
            train_summary = train_model(
                config,
                model_name,
                manifests_dir / "known_train.csv",
                manifests_dir / "known_val.csv",
                args.audio_root,
                run_dir,
                seed,
            )
            evaluation = evaluate_checkpoint(
                config,
                run_dir / "best_model.pt",
                run_dir / "thresholds.json",
                manifests_dir / "known_test.csv",
                args.audio_root,
                split="test",
                ood_manifest=manifests_dir / "ood_test.csv",
                device_name=str(config["training"].get("device", "cpu")),
            )
            save_evaluation(evaluation, run_dir / "test_evaluation.json")
            summaries.append({"model": model_name, "seed": seed, "training": train_summary, "test": evaluation})

    (output_root / "experiment_index.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Completed {len(summaries)} runs. See {output_root / 'experiment_index.json'}")


if __name__ == "__main__":
    main()
