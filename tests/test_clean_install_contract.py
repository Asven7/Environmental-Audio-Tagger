from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_clean_install.py"
DOC = ROOT / "docs" / "CLEAN_INSTALL.md"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_clean_install_test",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalized_doc_text() -> str:
    return " ".join(
        DOC.read_text(encoding="utf-8").lower().split()
    )


def test_clean_install_verifier_exists():
    assert SCRIPT.exists()


def test_clean_install_verifier_declares_all_dependency_groups():
    module = _load_module()

    expected = {
        "environmental-audio-tagger",
        "numpy",
        "pandas",
        "scipy",
        "soundfile",
        "torch",
        "torchaudio",
        "scikit-learn",
        "PyYAML",
        "matplotlib",
        "gradio",
        "sounddevice",
        "pytest",
    }
    assert set(module.REQUIRED_DISTRIBUTIONS) == expected


def test_clean_install_verifier_checks_installed_console_scripts():
    module = _load_module()

    assert module.CONSOLE_SCRIPTS == (
        "esaudio-train",
        "esaudio-evaluate",
        "esaudio-infer",
    )


def test_clean_install_verifier_is_dataset_and_frozen_artifact_independent():
    source = SCRIPT.read_text(encoding="utf-8").lower()

    forbidden = (
        "data/urbansound8k",
        "artifacts/experiments_phase11",
        "run_frozen_evaluation",
        "evaluate_checkpoint(",
        "train_model(",
        "tune_per_class_thresholds",
    )
    for token in forbidden:
        assert token not in source


def test_clean_install_verifier_uses_cpu_synthetic_smoke():
    module = _load_module()
    result = module.verify_synthetic_cpu_forward(ROOT)

    assert result["device"] == "cpu"
    assert result["models"]["cnn"]["model_spec_name"] == "cnn"
    assert result["models"]["crnn"]["model_spec_name"] == "crnn"
    assert result["models"]["cnn"]["output_shape"][0] == 2
    assert result["models"]["crnn"]["output_shape"][0] == 2


def test_clean_install_doc_uses_fresh_clone_and_fresh_virtual_environment():
    text = _normalized_doc_text()

    assert "git clone" in text
    assert "python -m venv .venv" in text
    assert 'python -m pip install ".[all]"' in text
    assert "python scripts\\verify_clean_install.py" in text


def test_clean_install_doc_does_not_require_scientific_re_evaluation():
    text = _normalized_doc_text()

    assert "do not rerun frozen held-out evaluation" in text
    assert "urbansound8k is not required" in text
