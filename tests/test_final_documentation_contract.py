"""Checks that the public documentation stays coherent with project interfaces."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CURRENT_DOCS = {
    "README.md",
    "ARCHITECTURE.md",
    "DATASET.md",
    "METHODS.md",
    "RESULTS.md",
    "REPRODUCIBILITY.md",
    "TESTING.md",
    "USAGE.md",
    "SECURITY_PRIVACY.md",
    "REFERENCES.md",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_expected_documents_exist_and_obsolete_stage_documents_are_absent():
    actual = {path.name for path in DOCS.glob("*.md")}
    assert actual == CURRENT_DOCS
    assert (ROOT / "LICENSE").exists()


def test_all_local_markdown_links_resolve():
    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for document in [ROOT / "README.md", ROOT / "CHANGELOG.md", *DOCS.glob("*.md")]:
        for destination in pattern.findall(_read(document)):
            if destination.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = destination.split("#", 1)[0].split("?", 1)[0].strip()
            if relative:
                assert (document.parent / relative).resolve().exists(), (
                    f"Broken relative link in {document}: {destination}"
                )


def test_current_docs_have_no_oral_presentation_instructions():
    forbidden = re.compile(
        r"\b(defense demo|presentation slides|oral defense|"
        r"professor demonstration|speaking script)\b|"
        r"(?:جلسه\s*دفاع|سناریوی\s*ارائه|اسلایدهای\s*ارائه)",
        re.IGNORECASE,
    )
    for document in [ROOT / "README.md", *DOCS.glob("*.md")]:
        assert not forbidden.search(_read(document)), str(document)


def test_readme_declares_correct_task_and_limitations():
    text = _read(ROOT / "README.md").lower()
    assert "window-level" in text
    assert "multi-label" in text
    assert "onset/offset" in text
    assert "out-of-distribution" in text


def test_results_preserve_recorded_scientific_numbers():
    text = _read(DOCS / "RESULTS.md")
    for required in (
        "0.618628 ± 0.008281",
        "0.727378 ± 0.007039",
        "0.592859 ± 0.010359",
        "0.617833 ± 0.013461",
        "0.953105 ± 0.025317",
        "0.046895 ± 0.025317",
        "0305a2aa821ed16a5f485cde1951d3720c47fa65dcc9d0ec9befd2e07f63be68",
    ):
        assert required in text


def test_methods_preserve_split_and_threshold_validation_only():
    methods = _read(DOCS / "METHODS.md").lower()
    dataset = _read(DOCS / "DATASET.md").lower()
    assert "validation" in methods
    assert "13" in methods and "23" in methods and "37" in methods
    assert "folds 1–7" in dataset
    assert "fold 8" in dataset
    assert "folds 9–10" in dataset
    assert "before" in dataset and "mix" in dataset


def test_reproducibility_explains_clean_install_and_missing_research_files():
    text = _read(DOCS / "REPRODUCIBILITY.md").lower()
    assert "git clone" in text
    assert "python -m venv .venv" in text
    assert "verify_clean_install.py" in text
    assert "not" in text and "tracked" in text
    assert "80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8" in text
    assert "788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2" in text


def test_usage_describes_executable_entry_points():
    usage = _read(DOCS / "USAGE.md")
    for script in ("scripts\\infer_file.py", "scripts\\run_ui.py", "scripts\\live_microphone.py"):
        assert script in usage
    for path in (
        "scripts/infer_file.py",
        "scripts/run_ui.py",
        "scripts/live_microphone.py",
    ):
        assert (ROOT / path).exists()


def test_ci_and_project_metadata_are_available():
    assert (ROOT / ".github/workflows/ci.yml").exists()
    project = _read(ROOT / "pyproject.toml")
    assert "esaudio-train" in project
    assert "esaudio-evaluate" in project
    assert "esaudio-infer" in project
    assert (DOCS / "TESTING.md").exists()
