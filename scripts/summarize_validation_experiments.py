#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.experiments import (
    aggregate_validation_records,
    atomic_write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase-11 validation-only multi-seed results as mean/std."
    )
    parser.add_argument("--output-root", default="artifacts/experiments_phase11")
    parser.add_argument(
        "--output",
        default="artifacts/experiments_phase11/validation_summary.json",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    training_index = json.loads(
        (output_root / "training_index.json").read_text(encoding="utf-8")
    )
    threshold_index = json.loads(
        (output_root / "threshold_index.json").read_text(encoding="utf-8")
    )

    summary = {
        "scope": "validation_only",
        "test_metrics_included": False,
        "models": aggregate_validation_records(
            training_index,
            threshold_index,
        ),
    }
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Validation-only summary: {args.output}")
    print("Held-out metrics included: NO")


if __name__ == "__main__":
    main()
