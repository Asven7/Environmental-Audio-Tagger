#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from esaudio.config import load_config
from esaudio.features import feature_extractor_from_config
from esaudio.models import build_model, count_parameters


PROJECT_DISTRIBUTION = "environmental-audio-tagger"
REQUIRED_DISTRIBUTIONS = (
    PROJECT_DISTRIBUTION,
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
)
CONSOLE_SCRIPTS = (
    "esaudio-train",
    "esaudio-evaluate",
    "esaudio-infer",
)


def distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in REQUIRED_DISTRIBUTIONS:
        versions[name] = importlib.metadata.version(name)
    return versions


def run_command(command: list[str], *, cwd: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
    }


def require_command_pass(result: dict) -> None:
    if int(result["returncode"]) == 0:
        return
    details = result["stderr_tail"] or result["stdout_tail"] or "(no output)"
    raise RuntimeError(
        "Clean-install verification command failed:\n"
        f"{' '.join(result['command'])}\n{details}"
    )


def verify_console_scripts(project_root: Path) -> list[dict]:
    results: list[dict] = []
    for name in CONSOLE_SCRIPTS:
        executable = shutil.which(name)
        if executable is None:
            raise RuntimeError(
                f"Installed console script was not found on PATH: {name}"
            )
        result = run_command([executable, "--help"], cwd=project_root)
        require_command_pass(result)
        results.append(
            {
                "name": name,
                "executable": executable,
                "returncode": result["returncode"],
            }
        )
    return results


def verify_optional_interfaces(project_root: Path) -> list[dict]:
    importlib.import_module("gradio")
    importlib.import_module("sounddevice")

    commands = (
        [sys.executable, "scripts/run_ui.py", "--help"],
        [sys.executable, "scripts/live_microphone.py", "--help"],
    )
    results: list[dict] = []
    for command in commands:
        result = run_command(command, cwd=project_root)
        require_command_pass(result)
        results.append(
            {
                "command": command,
                "returncode": result["returncode"],
            }
        )
    return results


def verify_synthetic_cpu_forward(project_root: Path) -> dict:
    config = load_config(project_root / "config" / "demo.yaml")
    sample_rate = int(config["project"]["sample_rate"])
    window_seconds = float(config["project"]["window_seconds"])
    samples = int(round(sample_rate * window_seconds))
    class_names = list(config["project"]["target_classes"])

    waveform = torch.zeros(2, samples, dtype=torch.float32)
    extractor = feature_extractor_from_config(config).cpu().eval()

    with torch.inference_mode():
        features = extractor(waveform)

    models: dict[str, dict] = {}
    for model_name in ("cnn", "crnn"):
        model, spec = build_model(config, model_name, len(class_names))
        model = model.cpu().eval()
        with torch.inference_mode():
            logits = model(features)

        expected = (2, len(class_names))
        if tuple(logits.shape) != expected:
            raise RuntimeError(
                f"{model_name} output shape {tuple(logits.shape)} != {expected}"
            )
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{model_name} produced non-finite logits")

        models[model_name] = {
            "parameters": int(count_parameters(model)),
            "output_shape": list(logits.shape),
            "model_spec_name": str(spec.name),
        }

    return {
        "device": "cpu",
        "sample_rate": sample_rate,
        "window_seconds": window_seconds,
        "input_samples": samples,
        "feature_shape": list(features.shape),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a fresh source installation without requiring UrbanSound8K, "
            "frozen experiment artifacts, network access, or microphone capture."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/phase16_clean_install_report.json",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output = (project_root / args.output).resolve()

    if not (project_root / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"pyproject.toml not found under project root: {project_root}"
        )

    versions = distribution_versions()

    pip_check = run_command(
        [sys.executable, "-m", "pip", "check"],
        cwd=project_root,
    )
    require_command_pass(pip_check)

    console_scripts = verify_console_scripts(project_root)
    optional_interfaces = verify_optional_interfaces(project_root)
    synthetic_forward = verify_synthetic_cpu_forward(project_root)

    report = {
        "scope": "phase16_clean_install_verification",
        "status": "PASS",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "distributions": versions,
        "pip_check": pip_check,
        "console_scripts": console_scripts,
        "optional_interfaces": optional_interfaces,
        "synthetic_cpu_forward": synthetic_forward,
        "scientific_boundary": {
            "dataset_required": False,
            "frozen_experiment_artifacts_required": False,
            "retrained": False,
            "thresholds_retuned": False,
            "heldout_metrics_recomputed": False,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== Phase 16 Clean-Install Verification ===")
    print("status=PASS")
    print(f"python={sys.version.split()[0]}")
    print(f"package={versions[PROJECT_DISTRIBUTION]}")
    print(
        "optional="
        f"gradio {versions['gradio']}, "
        f"sounddevice {versions['sounddevice']}"
    )
    print(
        "console_scripts="
        + ",".join(item["name"] for item in console_scripts)
    )
    models = synthetic_forward["models"]
    print(
        "synthetic_cpu_forward="
        f"cnn{models['cnn']['output_shape']} "
        f"crnn{models['crnn']['output_shape']}"
    )
    print(f"report={args.output}")
    print("UrbanSound8K was NOT required.")
    print("Frozen experiment artifacts were NOT required.")
    print("Held-out scientific metrics were NOT recomputed.")


if __name__ == "__main__":
    main()
