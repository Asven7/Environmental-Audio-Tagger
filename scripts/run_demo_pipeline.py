#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from esaudio.config import load_config
from esaudio.demo_data import generate_demo_dataset
from esaudio.evaluate_runner import evaluate_checkpoint, save_evaluation
from esaudio.inference import AudioTagger
from esaudio.runtime import benchmark_manifest
from esaudio.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete synthetic engineering smoke pipeline")
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--data-root", default="sample_data/demo")
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = generate_demo_dataset(args.data_root, config)
    summaries = {}
    for model_name in ("cnn", "crnn"):
        out_dir = Path(args.artifact_root) / f"demo_{model_name}"
        summary = train_model(
            config,
            model_name,
            paths["known_train"],
            paths["known_val"],
            args.data_root,
            out_dir,
            seed=int(config["training"]["seeds"][0]),
        )
        evaluation = evaluate_checkpoint(
            config,
            out_dir / "best_model.pt",
            out_dir / "thresholds.json",
            paths["known_test"],
            args.data_root,
            split="test",
            ood_manifest=paths["ood_test"],
            device_name="cpu",
        )
        save_evaluation(evaluation, out_dir / "test_evaluation.json")
        tagger = AudioTagger.from_files(out_dir / "best_model.pt", out_dir / "thresholds.json")
        runtime = benchmark_manifest(tagger, paths["known_test"], args.data_root, max_samples=8)
        runtime["stream_hop_ms"] = tagger.hop_seconds * 1000.0
        runtime["meets_no_backlog_criterion_p95"] = bool(runtime["total_ms_p95"] < runtime["stream_hop_ms"])
        (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
        summaries[model_name] = {"training": summary, "evaluation": evaluation, "runtime": runtime}

    report_path = Path(args.artifact_root) / "demo_pipeline_summary.json"
    report_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Synthetic engineering pipeline completed: {report_path}")
    print("These demo metrics are NOT research results and must not be reported as UrbanSound8K performance.")


if __name__ == "__main__":
    main()
