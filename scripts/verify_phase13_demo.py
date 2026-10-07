#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from esaudio.deployment import select_frozen_deployment
from esaudio.inference import AudioTagger
from esaudio.streaming import analyze_file


DEFAULT_AUDIO = Path(
    "data/UrbanSound8K/audio/fold8/103076-3-0-0.wav"
)
DEFAULT_OUTPUT = Path("artifacts/phase13c_acceptance.json")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def _validate_predictions(
    tagger: AudioTagger,
    predictions: list[Any],
) -> dict[str, Any]:
    if not predictions:
        raise RuntimeError("No sequential file predictions were produced")

    expected_classes = list(tagger.class_names)
    hop = float(tagger.hop_seconds)
    window = float(tagger.window_seconds)

    max_processing_ms = 0.0
    for index, prediction in enumerate(predictions):
        if int(prediction.index) != index:
            raise RuntimeError(
                f"Unexpected window index: expected {index}, got "
                f"{prediction.index}"
            )

        expected_start = index * hop
        expected_end = expected_start + window
        if not math.isclose(
            float(prediction.start_seconds),
            expected_start,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"Window {index} start mismatch: "
                f"{prediction.start_seconds} != {expected_start}"
            )
        if not math.isclose(
            float(prediction.end_seconds),
            expected_end,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"Window {index} end mismatch: "
                f"{prediction.end_seconds} != {expected_end}"
            )

        if list(prediction.scores.keys()) != expected_classes:
            raise RuntimeError(
                f"Window {index} score/class order does not match deployment"
            )

        for class_name, score in prediction.scores.items():
            score = float(score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(
                    f"Invalid score for {class_name} in window {index}: "
                    f"{score}"
                )

        processing_ms = float(prediction.processing_ms)
        if not math.isfinite(processing_ms) or processing_ms < 0:
            raise RuntimeError(
                f"Invalid processing time in window {index}: {processing_ms}"
            )
        max_processing_ms = max(max_processing_ms, processing_ms)

    return {
        "n_windows": len(predictions),
        "first_start_seconds": float(predictions[0].start_seconds),
        "last_end_seconds": float(predictions[-1].end_seconds),
        "max_processing_ms_observed": max_processing_ms,
        "hop_ms": hop * 1000.0,
        "observed_file_smoke_below_hop": bool(
            max_processing_ms < hop * 1000.0
        ),
    }


def _load_ui_module(project_root: Path):
    path = project_root / "scripts" / "run_ui.py"
    if not path.exists():
        raise FileNotFoundError(f"UI script not found: {path}")

    spec = importlib.util.spec_from_file_location(
        "phase13c_run_ui",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import UI script: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sounddevice_info() -> dict[str, Any]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required for Phase 13C. "
            'Install project extras with: python -m pip install -e ".[all]"'
        ) from exc

    default_input = None
    try:
        input_index = int(sd.default.device[0])
        if input_index >= 0:
            device = sd.query_devices(input_index)
            default_input = {
                "index": input_index,
                "name": str(device["name"]),
                "max_input_channels": int(device["max_input_channels"]),
                "default_samplerate": float(device["default_samplerate"]),
            }
    except Exception as exc:  # device enumeration is platform-specific
        default_input = {
            "warning": f"{type(exc).__name__}: {exc}",
        }

    return {
        "package_version": _package_version("sounddevice"),
        "default_input": default_input,
    }


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    audio_file = (project_root / args.audio_file).resolve()
    experiment_root = (project_root / args.experiment_root).resolve()
    output = (project_root / args.output).resolve()

    if not audio_file.exists():
        raise FileNotFoundError(
            f"Acceptance audio file not found: {audio_file}"
        )

    selection = select_frozen_deployment(
        experiment_root,
        model_name="crnn",
    )
    tagger = AudioTagger.from_files(
        selection.checkpoint,
        selection.thresholds,
        device=args.device,
    )

    predictions = analyze_file(tagger, audio_file)
    file_checks = _validate_predictions(tagger, predictions)

    ui_module = _load_ui_module(project_root)
    demo = ui_module.build_demo(
        tagger,
        deployment_text="Phase 13C end-to-end acceptance",
    )
    ui_type = type(demo).__name__
    if ui_type != "Blocks":
        raise RuntimeError(
            f"Expected Gradio Blocks UI, got {ui_type}"
        )

    sounddevice_info = _sounddevice_info()

    report = {
        "scope": "phase13c_end_to_end_demo_acceptance",
        "status": "PASS",
        "scientific_boundary": {
            "retrained": False,
            "thresholds_retuned": False,
            "heldout_test_read_for_development": False,
            "accuracy_metrics_recomputed": False,
        },
        "deployment": {
            "model": selection.model_name,
            "seed": int(selection.seed),
            "best_validation_mAP": float(
                selection.best_validation_mAP
            ),
            "checkpoint": str(selection.checkpoint),
            "thresholds": str(selection.thresholds),
            "checkpoint_sha256": str(selection.checkpoint_sha256),
            "threshold_sha256": str(
                selection.threshold_artifact_sha256
            ),
            "sample_rate": int(tagger.sample_rate),
            "window_seconds": float(tagger.window_seconds),
            "hop_seconds": float(tagger.hop_seconds),
            "class_names": list(tagger.class_names),
            "device": str(args.device),
        },
        "file_mode": {
            "audio_file": str(audio_file),
            **file_checks,
        },
        "ui": {
            "gradio_version": _package_version("gradio"),
            "build_type": ui_type,
            "build_passed": True,
        },
        "live_audio": sounddevice_info,
        "manual_checks_required": [
            "Launch Gradio UI on localhost and verify File tab once.",
            "Record browser microphone for at least 8 seconds.",
            "Verify complete history persists after Stop.",
            "Verify Newest first and Oldest first both work.",
            "Verify Clear results explicitly clears the session.",
        ],
        "limitations": [
            (
                "Live/background audio may be false-accepted as trained "
                "classes; the frozen rejection heuristic is not robust "
                "open-set recognition."
            ),
            (
                "Scores are sigmoid model outputs and are not calibrated "
                "probabilities."
            ),
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 13C end-to-end demo acceptance checks without "
            "retraining, threshold tuning, or held-out evaluation."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--experiment-root",
        default="artifacts/experiments_phase11",
    )
    parser.add_argument(
        "--audio-file",
        default=str(DEFAULT_AUDIO),
        help=(
            "Known validation/demo audio used only for sequential "
            "inference smoke verification"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    args = parser.parse_args()

    report = run_acceptance(args)

    print("=== Phase 13C End-to-End Demo Acceptance ===")
    print(f"status={report['status']}")
    deployment = report["deployment"]
    print(
        "deployment="
        f"{deployment['model']}_seed{deployment['seed']} "
        f"validation_mAP={deployment['best_validation_mAP']:.6f}"
    )
    print(
        "protocol="
        f"{deployment['sample_rate']}Hz "
        f"{deployment['window_seconds']:.1f}s-window "
        f"{deployment['hop_seconds']:.1f}s-hop"
    )
    file_mode = report["file_mode"]
    print(
        "file_mode="
        f"{file_mode['n_windows']} windows "
        f"max_processing_ms={file_mode['max_processing_ms_observed']:.3f} "
        f"below_hop={file_mode['observed_file_smoke_below_hop']}"
    )
    print(
        "ui="
        f"{report['ui']['build_type']} "
        f"gradio={report['ui']['gradio_version']}"
    )
    default_input = report["live_audio"]["default_input"]
    if isinstance(default_input, dict) and "name" in default_input:
        print(
            "default_input="
            f"{default_input['index']} {default_input['name']}"
        )
    else:
        print(f"default_input={default_input}")
    print(f"report={Path(args.output)}")
    print("Held-out accuracy metrics were NOT recomputed.")
    print("Thresholds were NOT changed.")


if __name__ == "__main__":
    main()
