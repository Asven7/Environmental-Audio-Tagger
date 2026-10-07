# Local Setup Guide

This guide targets the verified primary development machine:

- Windows 10 Enterprise 64-bit
- Python 3.11.9
- Intel Core i7-12700H
- 16 GB RAM
- NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB VRAM
- Git for Windows

The project does **not** require Node.js, Java, Docker, a database server, or a separately installed CUDA Toolkit for normal local use.

For a fresh reproducibility install, prefer [`CLEAN_INSTALL.md`](CLEAN_INSTALL.md). This document focuses on an editable developer checkout.

## 1. Check the Python executable

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
```

Use a native Windows CPython installation. If the path resolves to `C:\msys64\...`, use native Windows Python 3.11 instead.

## 2. Create and activate `.venv`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python -c "import sys; print(sys.executable)"
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Upgrade packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 4. Choose PyTorch CPU or CUDA

### Verified CUDA path

The development laptop was verified with PyTorch 2.10.0 / torchaudio 2.10.0 using the CUDA 12.6 wheels:

```powershell
python -m pip install `
  --index-url https://download.pytorch.org/whl/cu126 `
  "torch>=2.6,<2.11" `
  "torchaudio>=2.6,<2.11"
```

A separate CUDA Toolkit installation is not required for this prebuilt-wheel workflow.

### CPU fallback

```powershell
python -m pip uninstall -y torch torchaudio

python -m pip install `
  --index-url https://download.pytorch.org/whl/cpu `
  "torch>=2.6,<2.11" `
  "torchaudio>=2.6,<2.11"
```

CPU inference is a fully supported path and was independently verified against the real-time no-backlog criterion.

## 5. Install the complete developer/demo environment

For an editable developer checkout:

```powershell
python -m pip install -e ".[all]"
```

The `all` extra includes:

```text
Gradio
sounddevice
pytest
```

Then:

```powershell
python -m pip check
```

For a clean, non-editable reproducibility install, use:

```powershell
python -m pip install ".[all]"
```

and follow [`CLEAN_INSTALL.md`](CLEAN_INSTALL.md).

## 6. Verify installed versions

Use distribution metadata instead of package-specific `.version` attributes:

```powershell
python -c "from importlib.metadata import version; print('gradio=', version('gradio')); print('sounddevice=', version('sounddevice')); print('pytest=', version('pytest'))"
```

The verified development environment used:

```text
Python 3.11.9
Gradio 6.29.1
sounddevice 0.5.6
pytest 9.1.1
```

## 7. Verify PyTorch and CUDA

```powershell
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda_runtime=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is available:

```powershell
python -c "import torch; x=torch.randn((1024,1024), device='cuda'); y=x@x; torch.cuda.synchronize(); print('GPU smoke test:', y.shape, 'PASS')"
```

## 8. Run environment and repository checks

```powershell
python scripts\check_environment.py
python -m pip check
python -m pytest
```

The Phase-16B verified repository baseline was:

```text
181 passed
```

The exact count may increase if later phases add tests; a clean all-pass run is the requirement.

The repository configures pytest to use `.pytest_tmp/` because the target Windows machine encountered an ACL `PermissionError` in the user temp directory.

## 9. Optional UI and microphone checks

```powershell
python scripts\run_ui.py --help
python scripts\live_microphone.py --help
python scripts\live_microphone.py --list-devices
```

The browser microphone and physical `sounddevice` microphone paths were both verified in Phase 13.

## 10. Resource monitoring

```powershell
nvidia-smi
nvidia-smi -l 2
```

The development GPU has 4096 MiB VRAM, so the project intentionally uses lightweight models and retains CPU fallback.

## 11. What not to install

Do not add these unless a newly declared future scope requires them:

- standalone CUDA Toolkit,
- cuDNN separately,
- Node.js/npm,
- Java,
- Docker Desktop,
- PostgreSQL/MySQL/MongoDB,
- WSL solely for this project.

## 12. Clean rebuild

```powershell
deactivate
Remove-Item -Recurse -Force .venv

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Then install the appropriate PyTorch build and rerun:

```powershell
python -m pip install -e ".[all]"
python -m pip check
python -m pytest
```

## Scientific boundary

Environment repair must not be used as a reason to change the frozen checkpoint, thresholds, preprocessing, split, or scientific results.
