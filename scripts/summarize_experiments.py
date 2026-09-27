#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed CNN/CRNN experiment results")
    parser.add_argument("--experiment-index", default="artifacts/experiments/experiment_index.json")
    parser.add_argument("--output-dir", default="artifacts/experiments/summary")
    args = parser.parse_args()

    index_path = Path(args.experiment_index)
    runs = json.loads(index_path.read_text(encoding="utf-8"))
    rows = []
    for run in runs:
        known = run["test"]["known"]
        rows.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "parameter_count": run["training"]["parameter_count"],
                "f1_micro": known["f1_micro"],
                "f1_macro": known["f1_macro"],
                "mAP": known["mAP"],
                "hamming_loss": known["hamming_loss"],
                "ood_recall_rejected": run["test"].get("rejection", {}).get("ood_recall_rejected"),
                "known_false_rejection_rate": run["test"].get("rejection", {}).get("known_false_rejection_rate"),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_run.csv", index=False)

    numeric = [
        "parameter_count",
        "f1_micro",
        "f1_macro",
        "mAP",
        "hamming_loss",
        "ood_recall_rejected",
        "known_false_rejection_rate",
    ]
    summary = frame.groupby("model")[numeric].agg(["mean", "std"])
    summary.to_csv(output_dir / "mean_std.csv")
    print(frame.to_string(index=False))
    print("\nMean ± standard deviation:\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
