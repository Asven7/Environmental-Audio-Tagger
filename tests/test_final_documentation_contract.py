from __future__ import annotations

import re
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


def _assert_relative_markdown_links_resolve() -> None:
    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for path in [README, *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for raw_target in pattern.findall(text):
            target = raw_target.strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"Broken relative Markdown link in {path}: {raw_target}"


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
    assert "16a/16b/16c user verified" in completion
    assert "waiting for user local verification of 16b" not in completion
    assert "urbansound8k is not required" in clean

    # Final/current-status documents must not retain the pre-real-data project state.
    implementation = _normalized(ROOT / "docs" / "IMPLEMENTATION_STATUS.md")
    traceability = _normalized(ROOT / "docs" / "REQUIREMENTS_TRACEABILITY.md")
    verification = _normalized(ROOT / "docs" / "VERIFICATION.md")
    report_fa = _normalized(ROOT / "docs" / "IMPLEMENTATION_REPORT_FA.md")

    assert "three-seed experiment matrix user verified" in implementation
    assert "physical sounddevice microphone capture verified" in implementation
    assert "final real-data comparison" in traceability
    assert "user verified" in traceability
    assert "full repository suite: 181 passed" in verification
    assert "0.727378 ± 0.007039" in (ROOT / "docs" / "IMPLEMENTATION_REPORT_FA.md").read_text(
        encoding="utf-8"
    )

    # Historical phase protocol files should show their eventual verified state.
    phase_status_paths = [
        "PHASE_11_MULTI_SEED_EXPERIMENTS.md",
        "PHASE_11_FROZEN_RESULTS_AGGREGATION.md",
        "PHASE_12_RUNTIME_PROTOCOL.md",
        "PHASE_13_STREAMING_CORE.md",
        "PHASE_13_UI_MICROPHONE.md",
        "PHASE_13_END_TO_END_DEMO.md",
        "PHASE_14_QA.md",
        "PHASE_15_GITHUB_CI.md",
    ]
    for name in phase_status_paths:
        text = _normalized(ROOT / "docs" / name)
        assert "user verified" in text
        assert "waiting for user" not in text
        assert "ready to commit" not in text

    # Developer convenience metadata should match the final verified environment/workflow.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    setup = _normalized(ROOT / "docs" / "SETUP_LOCAL.md")
    troubleshooting = _normalized(ROOT / "docs" / "TROUBLESHOOTING.md")
    architecture = _normalized(ROOT / "docs" / "ARCHITECTURE.md")

    # The old requirements-lock.txt was not a real transitive lock and had become
    # stale. pyproject.toml is the dependency contract; clean-install verification
    # is the reproducibility check.
    assert not (ROOT / "requirements-lock.txt").exists()
    assert 'gradio>=5,<7' in pyproject
    assert 'sounddevice>=0.5,<0.6' in pyproject
    assert 'pytest>=8,<10' in pyproject
    assert "python -m pytest" in makefile
    assert 'python -m pip install -e ".[all]"' in (ROOT / "docs" / "SETUP_LOCAL.md").read_text(
        encoding="utf-8"
    )
    assert "clean/non-editable install" in troubleshooting
    assert "plot score trajectories" not in architecture
    assert "phase 16c" in _normalized(ROOT / "docs" / "PROJECT_LOG.md")
    assert "phase 16c" in _normalized(ROOT / "CHANGELOG.md")
    roadmap = _normalized(ROOT / "docs" / "ROADMAP.md")
    implementation_status = _normalized(ROOT / "docs" / "IMPLEMENTATION_STATUS.md")
    assert "final repository documentation/metadata consistency cleanup | ✅ user verified" in roadmap
    assert "phase 16c local verification has passed" in implementation_status

    _assert_relative_markdown_links_resolve()
