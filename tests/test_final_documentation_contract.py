from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOC_INDEX = ROOT / "docs" / "README.md"
CLEAN = ROOT / "docs" / "CLEAN_INSTALL.md"
COMPLETION = ROOT / "docs" / "PHASE_16_COMPLETION.md"


def _normalized(path: Path) -> str:
    text = path.read_text(encoding="utf-8").lower()
    # Documentation contracts should test prose meaning, not Markdown emphasis.
    for marker in ("**", "__", "`"):
        text = text.replace(marker, "")
    return " ".join(text.split())


def test_final_readme_uses_correct_scientific_scope():
    text = _normalized(README)

    assert "window-level multi-label environmental audio tagging" in text
    assert "does not estimate exact event onset/offset" in text
    assert "full sound event detection (sed)" in text


def test_final_readme_records_frozen_headline_results():
    text = README.read_text(encoding="utf-8")

    assert "0.618628 ± 0.008281" in text
    assert "0.727378 ± 0.007039" in text
    assert "0.592859 ± 0.010359" in text
    assert "0.617833 ± 0.013461" in text


def test_final_readme_records_validation_only_deployment_selection():
    text = _normalized(README)

    assert "crnn" in text
    assert "seed: 23" in text
    assert "validation map: 0.6757137110" in text
    assert "selected using validation performance only" in text


def test_final_readme_records_canonical_runtime():
    text = README.read_text(encoding="utf-8")

    assert "4.358 ms" in text
    assert "1.561 ms" in text
    assert "1 s hop" in text


def test_final_readme_does_not_overclaim_open_set_recognition():
    text = _normalized(README)

    assert "does not claim robust unknown-sound or general open-set recognition" in text
    assert "held-out rejection rate: ~4.69%" in text
    assert "held-out false acceptance remained very high" in text


def test_final_readme_contains_verified_clean_install_path():
    text = README.read_text(encoding="utf-8")

    assert "https://github.com/Asven7/Environmental-Audio-Tagger.git" in text
    assert "python -m venv .venv" in text
    assert 'python -m pip install ".[all]"' in text
    assert "full pytest: 173 passed" in text


def test_final_readme_preserves_frozen_test_boundary():
    text = _normalized(README)

    assert "do not rerun the frozen held-out evaluation" in text
    assert "thresholds must not be changed" in text
    assert "new experimental protocol" in text


def test_documentation_index_and_phase16_completion_exist():
    assert DOC_INDEX.exists()
    assert COMPLETION.exists()

    index = _normalized(DOC_INDEX)
    completion = _normalized(COMPLETION)
    clean = _normalized(CLEAN)

    assert "phase_11_final_results.md" in index
    assert "phase_12_runtime_results.md" in index
    assert "phase_16_completion.md" in index
    assert "fresh-clone acceptance" in completion
    assert "urbansound8k is not required" in clean
