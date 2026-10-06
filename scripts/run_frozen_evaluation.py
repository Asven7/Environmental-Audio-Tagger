#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.config import load_config
from esaudio.evaluate_runner import (
    FROZEN_EVALUATION_PROTOCOL_VERSION,
    evaluate_checkpoint,
    file_sha256,
    save_evaluation,
)


LOCK_NAME = ".frozen_evaluation.lock.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen test/OOD evaluation after checkpoint and "
            "validation-selected thresholds have been finalized."
        )
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--known-test-manifest", required=True)
    parser.add_argument("--ood-test-manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    lock_path = checkpoint_path.parent / LOCK_NAME

    if lock_path.exists():
        raise RuntimeError(
            "Frozen evaluation has already been recorded for this run directory: "
            f"{lock_path}"
        )
    if output_path.exists():
        raise FileExistsError(
            "Frozen evaluation output already exists and will not be overwritten: "
            f"{output_path}"
        )

    config = load_config(args.config)
    result = evaluate_checkpoint(
        config,
        checkpoint_path,
        args.thresholds,
        args.known_test_manifest,
        args.audio_root,
        split="test",
        ood_manifest=args.ood_test_manifest,
        device_name=args.device,
        require_threshold_provenance=True,
        validation_manifest_for_thresholds=args.validation_manifest,
    )

    save_evaluation(result, output_path, overwrite=False)

    lock_payload = {
        "protocol_version": FROZEN_EVALUATION_PROTOCOL_VERSION,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "threshold_artifact_sha256": file_sha256(args.thresholds),
        "validation_manifest_sha256": file_sha256(args.validation_manifest),
        "known_test_manifest_sha256": file_sha256(args.known_test_manifest),
        "ood_test_manifest_sha256": file_sha256(args.ood_test_manifest),
        "evaluation_artifact_sha256": file_sha256(output_path),
    }
    lock_path.write_text(
        json.dumps(lock_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Frozen evaluation: COMPLETED")
    print(f"Protocol: {FROZEN_EVALUATION_PROTOCOL_VERSION}")
    print(f"Evaluation artifact: {output_path}")
    print(f"Evaluation lock: {lock_path}")
    print("Thresholds were NOT retuned on test data.")
    print("Do not rerun or alter the protocol based on observed test metrics.")


if __name__ == "__main__":
    main()
