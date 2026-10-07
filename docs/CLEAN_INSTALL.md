# Clean Installation — Windows / CPU Reproducibility

This document verifies that the repository can be cloned and installed in a fresh Python
environment without relying on the existing development virtual environment.

The clean-install acceptance is an engineering reproducibility check. UrbanSound8K is not
required, frozen Phase-11 experiment artifacts are not required, and the frozen held-out
scientific evaluation must not be rerun.

## Baseline

The primary clean-install baseline is:

```text
Windows 10/11 x64
Python 3.11
CPU execution
Git
PowerShell
```

CPU is intentional. The final system already has a verified CUDA path, while a CPU clean
install is more portable and directly verifies the project's required CPU fallback.

## 1. Clone into a new directory

Do not reuse the existing project directory or its `.venv`.

From a parent directory such as `C:\Projects`:

```powershell
git clone https://github.com/Asven7/Environmental-Audio-Tagger.git environmental-audio-tagger-clean
cd environmental-audio-tagger-clean
```

Verify the checkout:

```powershell
git status
git log --oneline --decorate -3
```

The working tree should be clean.

## 2. Create a fresh virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable); print(sys.version)"
```

The executable should point inside:

```text
environmental-audio-tagger-clean\.venv\
```

## 3. Upgrade packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install CPU PyTorch

Install CPU PyTorch explicitly so the clean-install acceptance does not depend on a local
CUDA toolkit or GPU driver:

```powershell
python -m pip install `
  --index-url https://download.pytorch.org/whl/cpu `
  "torch>=2.6,<2.11" `
  "torchaudio>=2.6,<2.11"
```

## 5. Install the project with all optional local-demo/test dependencies

Use a normal, non-editable source install:

```powershell
python -m pip install ".[all]"
```

This installs the package plus:

```text
Gradio
sounddevice
pytest
```

The dataset and frozen experiment outputs are intentionally not Python package
dependencies.

## 6. Check dependency consistency

```powershell
python -m pip check
```

Expected:

```text
No broken requirements found.
```

## 7. Run the clean-install acceptance

```powershell
python scripts\verify_clean_install.py `
  --project-root . `
  --output artifacts\phase16_clean_install_report.json
```

Expected high-level output:

```text
=== Phase 16 Clean-Install Verification ===
status=PASS
...
console_scripts=esaudio-train,esaudio-evaluate,esaudio-infer
synthetic_cpu_forward=...
UrbanSound8K was NOT required.
Frozen experiment artifacts were NOT required.
Held-out scientific metrics were NOT recomputed.
```

The exact package versions and tensor frame dimension can vary within the dependency
ranges declared by `pyproject.toml`.

## 8. Run the complete repository test suite

```powershell
python -m pytest
```

This is the strongest repository-contained regression check in the fresh environment.

## 9. Verify the installed command-line entry points

The acceptance script already checks these, but they can also be inspected manually:

```powershell
esaudio-train --help
esaudio-evaluate --help
esaudio-infer --help
```

## 10. Verify optional demo command surfaces

No microphone recording or web server launch is required for clean-install acceptance:

```powershell
python scripts\run_ui.py --help
python scripts\live_microphone.py --help
```

The real browser microphone and local sounddevice capture were verified separately in
Phase 13.

## 11. Verify Git cleanliness

The acceptance report is written under `artifacts/`, which is a local engineering-output
location.

Run:

```powershell
git status
```

The source checkout should remain clean when ignored local artifacts and the virtual
environment are excluded correctly.

## Optional CUDA environment

CUDA is not required to pass Phase 16. The project's CUDA path was already verified in the
main development environment.

If a separate CUDA environment is desired later, install the PyTorch build matching the
supported NVIDIA runtime for that machine, then rerun the same repository tests and
synthetic verification. Do not change the frozen model, thresholds, or evaluation protocol
to accommodate a runtime environment.

## Scientific boundary

Clean-install verification must not:

```text
download UrbanSound8K for acceptance
retrain CNN or CRNN models
retune validation thresholds
modify frozen checkpoints
inspect held-out labels for development
recompute frozen held-out accuracy/F1/mAP
```

In particular: **do not rerun frozen held-out evaluation** as part of Phase 16.

The purpose of this phase is installation/documentation reproducibility, not a new
scientific experiment.
