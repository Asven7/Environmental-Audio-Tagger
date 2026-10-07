from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_workflow_exists_and_has_expected_triggers():
    text = _text()

    assert "name: CI" in text
    assert "push:" in text
    assert "pull_request:" in text
    assert "workflow_dispatch:" in text


def test_ci_uses_current_major_official_actions_and_python311():
    text = _text()

    assert "actions/checkout@v7" in text
    assert "actions/setup-python@v7" in text
    assert 'python-version: "3.11"' in text
    assert "runs-on: ubuntu-latest" in text


def test_ci_has_read_only_repository_permissions():
    text = _text()

    assert "permissions:" in text
    assert "contents: read" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text


def test_ci_installs_cpu_torch_and_repository_test_dependencies():
    text = _text()

    assert "https://download.pytorch.org/whl/cpu" in text
    assert '"torch>=2.6,<2.11"' in text
    assert '"torchaudio>=2.6,<2.11"' in text
    assert 'python -m pip install -e ".[dev,ui]"' in text


def test_ci_runs_repository_quality_smokes_and_pytest():
    text = _text()

    assert "python -m pip check" in text
    assert "python -m compileall -q src scripts tests" in text
    assert "python scripts/run_ui.py --help" in text
    assert "python scripts/live_microphone.py --help" in text
    assert "python -m pytest" in text


def test_ci_does_not_require_private_local_data_or_frozen_artifacts():
    text = _text().lower()

    forbidden = (
        "data/urbansound8k",
        "artifacts/experiments_phase11",
        "run_phase14_qa.py",
        "pull_request_target",
        "secrets.",
    )
    for token in forbidden:
        assert token not in text
