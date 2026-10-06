from __future__ import annotations

import argparse
import json
import logging

from .config import load_config
from .evaluate_runner import evaluate_checkpoint, save_evaluation
from .inference import AudioTagger
from .training import train_model


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def train_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN/CRNN environmental audio tagger"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["cnn", "crnn"], required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help=(
            "Opt in to validation threshold tuning during training. "
            "Research runs should leave this disabled and use the Phase-10 "
            "threshold-selection command after training."
        ),
    )
    args = parser.parse_args()
    _setup_logging(args.log_level)
    config = load_config(args.config)
    summary = train_model(
        config,
        args.model,
        args.train_manifest,
        args.val_manifest,
        args.audio_root,
        args.output_dir,
        args.seed,
        tune_thresholds=args.tune_thresholds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def evaluate_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained environmental audio tagger"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--known-manifest", required=True)
    parser.add_argument("--ood-manifest")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Require threshold provenance for test evaluation.",
    )
    parser.add_argument(
        "--validation-manifest",
        help="Validation manifest used to select thresholds; required with --frozen.",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.frozen:
        parser.error(
            "Test evaluation is protected. Use the dedicated "
            "scripts/run_frozen_evaluation.py workflow or pass --frozen."
        )
    if args.frozen and not args.validation_manifest:
        parser.error("--validation-manifest is required with --frozen")
    if args.split == "test" and not args.output:
        parser.error("--output is required for test evaluation")

    config = load_config(args.config)
    result = evaluate_checkpoint(
        config,
        args.checkpoint,
        args.thresholds,
        args.known_manifest,
        args.audio_root,
        split=args.split,
        ood_manifest=args.ood_manifest,
        device_name=args.device,
        require_threshold_provenance=args.frozen,
        validation_manifest_for_thresholds=args.validation_manifest,
    )
    if args.output:
        save_evaluation(
            result,
            args.output,
            overwrite=(args.split != "test"),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def infer_entry() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one audio file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    tagger = AudioTagger.from_files(
        args.checkpoint,
        args.thresholds,
        device=args.device,
    )
    result = tagger.predict_file(args.audio)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
