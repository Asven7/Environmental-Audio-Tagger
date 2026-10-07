from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_phase14_qa.py"
    spec = importlib.util.spec_from_file_location(
        "run_phase14_qa_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    # dataclasses resolves postponed string annotations through
    # sys.modules[cls.__module__].  A module loaded manually with
    # module_from_spec() is not inserted automatically, so register it
    # before exec_module().
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_constants_match_locked_phase12_deployment():
    module = _load_module()

    assert module.EXPECTED_MODEL == "crnn"
    assert module.EXPECTED_SEED == 23
    assert module.EXPECTED_CHECKPOINT_SHA256 == (
        "80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8"
    )
    assert module.EXPECTED_THRESHOLD_SHA256 == (
        "788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2"
    )


def test_prohibited_tracked_paths_detects_local_and_generated_content():
    module = _load_module()

    bad = module.prohibited_tracked_paths(
        [
            "README.md",
            "src/esaudio/inference.py",
            ".venv/Scripts/python.exe",
            "data/UrbanSound8K/audio/fold1/a.wav",
            "artifacts/experiments_phase11/crnn_seed23/best_model.pt",
            "src/esaudio/__pycache__/inference.cpython-311.pyc",
        ]
    )

    assert ".venv/Scripts/python.exe" in bad
    assert "data/UrbanSound8K/audio/fold1/a.wav" in bad
    assert (
        "artifacts/experiments_phase11/crnn_seed23/best_model.pt"
        in bad
    )
    assert "src/esaudio/__pycache__/inference.cpython-311.pyc" in bad
    assert "README.md" not in bad


def test_prohibited_tracked_paths_accepts_normal_repository_files():
    module = _load_module()

    assert module.prohibited_tracked_paths(
        [
            "README.md",
            "pyproject.toml",
            "src/esaudio/inference.py",
            "tests/test_streaming.py",
            "docs/PHASE_13_END_TO_END_DEMO.md",
        ]
    ) == []


def test_must_pass_raises_on_failed_command():
    module = _load_module()

    failed = module.CommandResult(
        name="example",
        command=["python", "-c", "raise SystemExit(1)"],
        returncode=1,
        elapsed_seconds=0.01,
        stdout_tail="",
        stderr_tail="failure",
    )
    with pytest.raises(RuntimeError, match="QA command failed"):
        module._must_pass(failed)


def test_phase14_source_preserves_frozen_scientific_boundary():
    source = (
        ROOT / "scripts" / "run_phase14_qa.py"
    ).read_text(encoding="utf-8").lower()

    forbidden = (
        "tune_per_class_thresholds",
        "save_thresholds(",
        "run_frozen_evaluation",
        "evaluate_checkpoint(",
        "train_experiment(",
    )
    for token in forbidden:
        assert token not in source
