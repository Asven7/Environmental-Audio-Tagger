#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from esaudio.deployment import select_frozen_deployment
from esaudio.runtime_profile import (
    RuntimeProfiler,
    representative_waveform_from_manifest,
    runtime_environment,
    save_runtime_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the frozen deployment model with batch size one. "
            "The deployment seed is selected from validation mAP only."
        )
    )
    parser.add_argument(
        "--experiment-root",
        default="artifacts/experiments_phase11",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/manifests_phase5/known_val.csv",
        help="Representative validation manifest; audio loading is outside timed region.",
    )
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--model", choices=["cnn", "crnn"], default="crnn")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selection = select_frozen_deployment(
        args.experiment_root,
        model_name=args.model,
    )
    profiler = RuntimeProfiler(
        selection.checkpoint,
        selection.thresholds,
        device=args.device,
    )

    waveform, input_metadata = representative_waveform_from_manifest(
        args.manifest,
        args.audio_root,
        profiler.sample_rate,
    )
    report = profiler.benchmark(
        waveform,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    report["deployment_selection"] = selection.as_dict()
    report["representative_input"] = input_metadata
    report["environment"] = runtime_environment()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path("artifacts/runtime_phase12")
            / f"{selection.model_name}_seed{selection.seed}_{report['device']}.json"
        )

    saved = save_runtime_report(
        report,
        output_path,
        overwrite=args.overwrite,
    )

    total = report["canonical_total"]
    print("=== Phase 12 Frozen Runtime Benchmark ===")
    print(
        f"deployment={selection.model_name} seed={selection.seed} "
        f"selection=validation_mAP({selection.best_validation_mAP:.6f})"
    )
    print(f"device={report['device']} batch_size=1")
    print(
        "canonical_total_ms: "
        f"mean={total['mean_ms']:.3f} "
        f"p50={total['p50_ms']:.3f} "
        f"p95={total['p95_ms']:.3f} "
        f"p99={total['p99_ms']:.3f}"
    )
    print(
        f"hop_ms={report['stream_hop_ms']:.3f} "
        f"p95_margin_ms={report['p95_no_backlog_margin_ms']:.3f}"
    )
    print(
        "meets_no_backlog_criterion_p95="
        f"{report['meets_no_backlog_criterion_p95']}"
    )
    print(
        "initial_prediction_latency_p95_ms="
        f"{report['initial_prediction_latency_p95_ms']:.3f}"
    )
    print(
        "parity_max_abs_score_diff="
        f"{report['parity_with_canonical_inference']['max_abs_score_diff']:.8g}"
    )
    for stage_name in (
        "preprocess",
        "host_to_device",
        "feature",
        "model",
        "postprocess",
    ):
        stage = report["stages"][stage_name]
        print(
            f"{stage_name}_ms: mean={stage['mean_ms']:.3f} "
            f"p95={stage['p95_ms']:.3f}"
        )
    print(f"Runtime report: {saved}")
    print("Accuracy/test metrics were NOT recomputed.")
    print("Thresholds were NOT changed.")


if __name__ == "__main__":
    main()
