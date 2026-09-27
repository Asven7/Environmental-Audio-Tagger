# Local Setup Guide

This guide targets the primary development machine currently used for the project:

- Windows 10 Enterprise 64-bit
- Python 3.11.9
- Intel Core i7-12700H
- 16 GB RAM
- NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB VRAM
- Git for Windows

The project does **not** require Node.js, Java, Docker, a database server, or a separately installed CUDA Toolkit for normal PyTorch development.

## 1. Important Python Path Check

Before creating the virtual environment, confirm which Python executable is actually being launched:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
```

A normal native Windows CPython installation is preferred. If the path resolves to `C:\msys64\...`, stop before installing PyTorch and use a native Windows Python 3.11 installation instead.

## 2. Create the Virtual Environment

Run these commands from the repository root:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify activation:

```powershell
python -c "import sys; print(sys.executable)"
```

The path should point inside the repository's `.venv` directory.

If PowerShell blocks activation, use the temporary per-process policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This does not permanently change the machine-wide execution policy.

## 3. Upgrade Packaging Tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Verify:

```powershell
python -m pip --version
```

## 4. Install PyTorch for the RTX 3050 Ti

The repository was previously verified with PyTorch 2.10.0 and torchaudio 2.10.0. To minimize moving parts during the undergraduate project, the Windows GPU setup uses the official CUDA 12.6 wheels for the same versions.

```powershell
python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

A separate CUDA Toolkit installation is not required for this prebuilt-wheel workflow. The NVIDIA display driver still needs to be compatible, which is checked using `nvidia-smi`.

### CPU fallback

If GPU installation or CUDA initialization is problematic, use the CPU build instead:

```powershell
python -m pip uninstall -y torch torchaudio
python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

The project remains functional on CPU; training will simply be slower.

## 5. Install the Project and Development Dependencies

For the current phase, install the core project plus testing tools. The UI and live microphone dependencies are intentionally deferred until their dedicated phase.

```powershell
python -m pip install -e ".[dev]"
```

Check dependency consistency:

```powershell
python -m pip check
```

## 6. Verify the Environment

Run the standard-library environment checker:

```powershell
python scripts/check_environment.py
```

Then verify PyTorch directly:

```powershell
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda_runtime=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is available, also perform a small real GPU operation:

```powershell
python -c "import torch; x=torch.randn((1024,1024), device='cuda'); y=x@x; torch.cuda.synchronize(); print('GPU smoke test:', y.shape, 'PASS')"
```

## 7. Run the Existing Test Suite

Only after editable installation:

```powershell
python -m pytest
```

Expected baseline: all existing tests should pass. The exact count may increase as the project evolves.

On the target Windows machine, pytest's default user-temp directory produced an ACL `PermissionError`. The repository therefore configures a project-local temporary directory (`.pytest_tmp/`) through `pyproject.toml`. No extra test flag should be required.

## 8. Resource Monitoring

GPU state and VRAM:

```powershell
nvidia-smi
```

Live refresh every two seconds:

```powershell
nvidia-smi -l 2
```

The current machine reports 4096 MiB total GPU memory, so later training phases will use a lightweight model and conservative batch sizes. CPU execution remains the fallback.

## 9. What Not to Install

Do not install these unless a later phase explicitly requires them:

- standalone CUDA Toolkit;
- cuDNN separately;
- Node.js/npm;
- Java;
- Docker Desktop;
- PostgreSQL/MySQL/MongoDB;
- WSL solely for this project.

Keeping the environment small reduces dependency and debugging risk.

## 10. Clean Rebuild of the Virtual Environment

If the environment becomes inconsistent:

```powershell
Deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Then repeat the PyTorch and editable project installation steps above.
