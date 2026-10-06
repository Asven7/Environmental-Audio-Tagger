#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.frozen_results import (
    aggregate_frozen_experiments,
    write_groups_csv,
    write_headline_csv,
    write_per_class_csv,
)
from esaudio.experiments import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate already-frozen held-out evaluation JSON files across the "
            "three predeclared seeds. This script performs no inference and no retuning."
        )
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/experiments_phase11",
    )
    parser.add_argument(
        "--summary",
        default="artifacts/experiments_phase11/frozen_test_summary.json",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary_path = Path(args.summary)

    summary = aggregate_frozen_experiments(output_root)
    atomic_write_json(summary_path, summary)

    headline_csv = summary_path.with_name("frozen_test_headline.csv")
    per_class_csv = summary_path.with_name("frozen_test_per_class.csv")
    groups_csv = summary_path.with_name("frozen_test_groups.csv")

    write_headline_csv(summary, headline_csv)
    write_per_class_csv(summary, per_class_csv)
    write_groups_csv(summary, groups_csv)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Frozen held-out summary: {summary_path}")
    print(f"Headline CSV: {headline_csv}")
    print(f"Per-class CSV: {per_class_csv}")
    print(f"Group CSV: {groups_csv}")
    print("Inference performed: NO")
    print("Threshold retuning performed: NO")


if __name__ == "__main__":
    main()
