#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.config import load_config
from esaudio.evaluate_runner import select_thresholds_from_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and freeze per-class thresholds using KNOWN VALIDATION only."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    result = select_thresholds_from_validation(
        config,
        args.checkpoint,
        args.validation_manifest,
        args.audio_root,
        args.output,
        device_name=args.device,
        overwrite=False,
    )

    report_path = (
        Path(args.report)
        if args.report
        else Path(args.output).with_name("threshold_selection.json")
    )
    if report_path.exists():
        raise FileExistsError(f"Threshold-selection report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Threshold selection: COMPLETED")
    print("Selection split: val")
    print("Selection metric: per-class F1")
    print(f"Threshold artifact: {args.output}")
    print(f"Selection report: {report_path}")
    print("Known test data read: NO")
    print("OOD test data read: NO")


if __name__ == "__main__":
    main()
