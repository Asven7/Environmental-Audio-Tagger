from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esaudio.checkpoints import save_checkpoint, save_thresholds
from esaudio.config import load_config
from esaudio.deployment import select_frozen_deployment
from esaudio.experiments import (
    EXPERIMENT_FREEZE_VERSION,
    file_sha256,
)
from esaudio.features import feature_extractor_from_config
from esaudio.models import build_model
from esaudio.runtime_profile import RuntimeProfiler, _stats


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_experiment_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "experiments"
    training = []
    frozen_runs = []
    scores = {
        ("cnn", 13): 0.50,
        ("cnn", 23): 0.52,
        ("cnn", 37): 0.51,
        ("crnn", 13): 0.66,
        ("crnn", 23): 0.68,
        ("crnn", 37): 0.67,
    }

    for model in ("cnn", "crnn"):
        for seed in (13, 23, 37):
            run_dir = root / f"{model}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "best_model.pt"
            thresholds = run_dir / "thresholds.json"
            checkpoint.write_bytes(f"{model}-{seed}-model".encode())
            thresholds.write_text(
                json.dumps({"thresholds": [0.5]}),
                encoding="utf-8",
            )
            training.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "training": {
                        "best_validation_mAP": scores[(model, seed)],
                    },
                }
            )
            frozen_runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "threshold_artifact_sha256": file_sha256(thresholds),
                }
            )

    _write_json(root / "training_index.json", training)
    _write_json(
        root / "experiment_freeze.json",
        {
            "freeze_version": EXPERIMENT_FREEZE_VERSION,
            "runs": frozen_runs,
        },
    )
    return root


def test_deployment_selects_highest_validation_map_without_best_seed_on_test(tmp_path):
    root = _make_experiment_fixture(tmp_path)
    selected = select_frozen_deployment(root, model_name="crnn")
    assert selected.seed == 23
    assert selected.best_validation_mAP == pytest.approx(0.68)
    assert "validation" in selected.selection_rule


def test_deployment_tie_breaks_to_lower_seed(tmp_path):
    root = _make_experiment_fixture(tmp_path)
    index_path = root / "training_index.json"
    records = json.loads(index_path.read_text(encoding="utf-8"))
    for record in records:
        if record["model"] == "crnn" and record["seed"] in (13, 23):
            record["training"]["best_validation_mAP"] = 0.70
    _write_json(index_path, records)

    selected = select_frozen_deployment(root, model_name="crnn")
    assert selected.seed == 13


def test_deployment_rejects_checkpoint_modified_after_freeze(tmp_path):
    root = _make_experiment_fixture(tmp_path)
    (root / "crnn_seed23" / "best_model.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="no longer matches experiment freeze"):
        select_frozen_deployment(root, model_name="crnn")


def test_timing_stats_include_p95_and_sample_std():
    stats = _stats([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["std_ms"] > 0
    assert stats["p95_ms"] >= stats["p50_ms"]


def _make_demo_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    config = load_config(ROOT / "config" / "demo.yaml")
    class_names = list(config["project"]["target_classes"])
    model, spec = build_model(config, "crnn", len(class_names))
    extractor = feature_extractor_from_config(config)

    checkpoint = tmp_path / "model.pt"
    thresholds = tmp_path / "thresholds.json"
    save_checkpoint(
        checkpoint,
        model,
        spec.as_dict(),
        class_names,
        extractor.export_config(),
        {
            "sample_rate": int(config["project"]["sample_rate"]),
            "window_seconds": float(config["project"]["window_seconds"]),
            "hop_seconds": float(config["project"]["hop_seconds"]),
            "target_rms_dbfs": float(
                config.get("data", {}).get("target_rms_dbfs", -20.0)
            ),
        },
        {
            "seed": 13,
            "best_epoch": 1,
            "validation_mAP": 0.5,
        },
    )
    save_thresholds(
        thresholds,
        class_names,
        [0.5] * len(class_names),
    )
    return checkpoint, thresholds


def test_runtime_profiler_cpu_matches_canonical_inference_and_reports_contract(tmp_path):
    checkpoint, thresholds = _make_demo_checkpoint(tmp_path)
    profiler = RuntimeProfiler(
        checkpoint,
        thresholds,
        device="cpu",
    )

    t = np.arange(profiler.window_samples, dtype=np.float32)
    waveform = (0.1 * np.sin(2 * np.pi * 440.0 * t / profiler.sample_rate)).astype(
        np.float32
    )
    report = profiler.benchmark(waveform, warmup=0, iterations=2)

    assert report["batch_size"] == 1
    assert report["device"] == "cpu"
    assert report["parity_with_canonical_inference"]["passed"] is True
    assert report["parity_with_canonical_inference"]["max_abs_score_diff"] <= 1e-5
    assert report["canonical_total"]["n"] == 2
    assert report["stages"]["feature"]["n"] == 2
    assert report["stream_hop_ms"] == pytest.approx(profiler.hop_seconds * 1000.0)
    assert report["initial_prediction_latency_p95_ms"] == pytest.approx(
        report["window_ms"] + report["canonical_total"]["p95_ms"]
    )


def test_runtime_benchmark_script_does_not_use_heldout_accuracy_for_selection():
    source = (
        ROOT / "scripts" / "benchmark_frozen_runtime.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = (
        "known_test",
        "ood_test",
        "test_evaluation",
        "frozen_test_summary",
        "f1_micro",
        "f1_macro",
    )
    for token in forbidden:
        assert token not in source
