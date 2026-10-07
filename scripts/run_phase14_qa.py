#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from esaudio.deployment import select_frozen_deployment


EXPECTED_MODEL = "crnn"
EXPECTED_SEED = 23
EXPECTED_CHECKPOINT_SHA256 = (
    "80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8"
)
EXPECTED_THRESHOLD_SHA256 = (
    "788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2"
)

PROHIBITED_TRACKED_PREFIXES = (
    ".venv/",
    ".pytest_tmp/",
    "data/UrbanSound8K/",
    "artifacts/experiments_phase11/",
)

PROHIBITED_TRACKED_SEGMENTS = (
    "/__pycache__/",
)


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _tail(text: str, *, max_lines: int = 30) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def run_command(
    name: str,
    command: list[str],
    *,
    cwd: Path,
) -> CommandResult:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return CommandResult(
        name=name,
        command=command,
        returncode=int(completed.returncode),
        elapsed_seconds=float(elapsed),
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def prohibited_tracked_paths(paths: Iterable[str]) -> list[str]:
    bad: list[str] = []
    for raw in paths:
        path = raw.replace("\\", "/")
        if any(path.startswith(prefix) for prefix in PROHIBITED_TRACKED_PREFIXES):
            bad.append(path)
            continue
        wrapped = f"/{path}/"
        if any(segment in wrapped for segment in PROHIBITED_TRACKED_SEGMENTS):
            bad.append(path)
    return sorted(set(bad))


def check_git_tracked_hygiene(project_root: Path) -> dict:
    result = run_command(
        "git_ls_files",
        ["git", "ls-files"],
        cwd=project_root,
    )
    if not result.passed:
        raise RuntimeError(
            "git ls-files failed:\n"
            f"{result.stderr_tail or result.stdout_tail}"
        )

    tracked = [line for line in result.stdout_tail.splitlines() if line.strip()]
    # stdout_tail is intentionally short for reports, so get the complete list
    # for the actual hygiene decision.
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Could not enumerate tracked Git files")
    all_tracked = [line for line in completed.stdout.splitlines() if line.strip()]
    bad = prohibited_tracked_paths(all_tracked)
    if bad:
        raise RuntimeError(
            "Prohibited generated/local paths are tracked by Git:\n"
            + "\n".join(bad)
        )

    return {
        "tracked_file_count": len(all_tracked),
        "prohibited_tracked_paths": bad,
    }


def check_frozen_deployment(experiment_root: Path) -> dict:
    selection = select_frozen_deployment(
        experiment_root,
        model_name=EXPECTED_MODEL,
    )

    problems: list[str] = []
    if selection.model_name != EXPECTED_MODEL:
        problems.append(
            f"model={selection.model_name!r}, expected {EXPECTED_MODEL!r}"
        )
    if int(selection.seed) != EXPECTED_SEED:
        problems.append(
            f"seed={selection.seed}, expected {EXPECTED_SEED}"
        )
    if selection.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        problems.append("checkpoint SHA-256 changed")
    if selection.threshold_artifact_sha256 != EXPECTED_THRESHOLD_SHA256:
        problems.append("threshold artifact SHA-256 changed")

    if problems:
        raise RuntimeError(
            "Frozen deployment integrity check failed: "
            + "; ".join(problems)
        )

    return {
        "model": selection.model_name,
        "seed": int(selection.seed),
        "best_validation_mAP": float(selection.best_validation_mAP),
        "checkpoint": str(selection.checkpoint),
        "thresholds": str(selection.thresholds),
        "checkpoint_sha256": selection.checkpoint_sha256,
        "threshold_sha256": selection.threshold_artifact_sha256,
    }


def _must_pass(result: CommandResult) -> None:
    if result.passed:
        return
    details = result.stderr_tail or result.stdout_tail or "(no output)"
    raise RuntimeError(
        f"QA command failed: {result.name}\n"
        f"command: {' '.join(result.command)}\n"
        f"{details}"
    )


def run_qa(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root).resolve()
    experiment_root = (project_root / args.experiment_root).resolve()
    audio_file = (project_root / args.audio_file).resolve()
    output = (project_root / args.output).resolve()

    if not (project_root / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"pyproject.toml not found under project root: {project_root}"
        )
    if not audio_file.exists():
        raise FileNotFoundError(
            f"Phase-14 QA demo audio not found: {audio_file}"
        )

    frozen = check_frozen_deployment(experiment_root)
    git_hygiene = check_git_tracked_hygiene(project_root)

    commands: list[CommandResult] = []

    checks = [
        (
            "git_diff_check",
            ["git", "diff", "--check"],
        ),
        (
            "pip_check",
            [sys.executable, "-m", "pip", "check"],
        ),
        (
            "compileall",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "src",
                "scripts",
                "tests",
            ],
        ),
        (
            "ui_help",
            [sys.executable, "scripts/run_ui.py", "--help"],
        ),
        (
            "microphone_help",
            [sys.executable, "scripts/live_microphone.py", "--help"],
        ),
        (
            "phase13_acceptance",
            [
                sys.executable,
                "scripts/verify_phase13_demo.py",
                "--project-root",
                ".",
                "--experiment-root",
                args.experiment_root,
                "--audio-file",
                args.audio_file,
                "--device",
                args.device,
                "--output",
                args.phase13_output,
            ],
        ),
        (
            "pytest_full",
            [sys.executable, "-m", "pytest"],
        ),
    ]

    for name, command in checks:
        result = run_command(name, command, cwd=project_root)
        commands.append(result)
        print(
            f"[{'PASS' if result.passed else 'FAIL'}] "
            f"{name} ({result.elapsed_seconds:.2f}s)"
        )
        _must_pass(result)

    report = {
        "scope": "phase14_quality_assurance",
        "status": "PASS",
        "scientific_boundary": {
            "retrained": False,
            "thresholds_retuned": False,
            "heldout_metrics_recomputed": False,
            "frozen_artifacts_modified": False,
        },
        "frozen_deployment": frozen,
        "git_hygiene": git_hygiene,
        "commands": [asdict(result) | {"passed": result.passed} for result in commands],
        "qa_claim": (
            "Repository engineering quality gates passed locally; this does "
            "not change or strengthen the frozen scientific accuracy claims."
        ),
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
            "Run Phase-14 repository QA gates without retraining, "
            "retuning thresholds, or recomputing held-out metrics."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--experiment-root",
        default="artifacts/experiments_phase11",
    )
    parser.add_argument(
        "--audio-file",
        default="data/UrbanSound8K/audio/fold8/103076-3-0-0.wav",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--phase13-output",
        default="artifacts/phase13c_acceptance.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/phase14_qa_report.json",
    )
    args = parser.parse_args()

    report = run_qa(args)
    print("=== Phase 14 QA Gate ===")
    print(f"status={report['status']}")
    frozen = report["frozen_deployment"]
    print(
        "frozen_deployment="
        f"{frozen['model']}_seed{frozen['seed']} "
        f"validation_mAP={frozen['best_validation_mAP']:.6f}"
    )
    print(
        "git_hygiene="
        f"{report['git_hygiene']['tracked_file_count']} tracked files, "
        "0 prohibited tracked paths"
    )
    print(f"quality_checks={len(report['commands'])} passed")
    print(f"report={args.output}")
    print("Frozen scientific metrics were NOT recomputed.")
    print("Thresholds were NOT changed.")


if __name__ == "__main__":
    main()
