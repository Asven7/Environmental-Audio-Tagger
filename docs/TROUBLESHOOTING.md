# Troubleshooting

This document records recurring development problems and the smallest reliable fix.

## `py` command is not recognized

**Meaning:** the Python Launcher for Windows is not installed or not on `PATH`.

**Impact on this project:** none, as long as the `python` command resolves to a suitable native Windows CPython installation.

**Check:**

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
```

Use `python -m ...` commands throughout the project instead of depending on `py`.

## `where.exe python` shows `WindowsApps` and `msys64`

The `where.exe` output alone does not prove which interpreter is actually executing. Use:

```powershell
python -c "import sys; print(sys.executable)"
```

If it resolves to `C:\msys64\...`, do not install the Windows PyTorch wheels into that interpreter. Use a native Windows CPython 3.11 installation.

## PowerShell refuses to activate `.venv`

Typical message: script execution is disabled.

Use a temporary policy for the current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Avoid changing the machine-wide policy just for this project.

## `ModuleNotFoundError: No module named 'esaudio'`

The repository uses a `src/` package layout. Install it in editable mode before running tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## `torch.cuda.is_available()` returns `False`

First check the NVIDIA driver:

```powershell
nvidia-smi
```

Then inspect the installed PyTorch build:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.version.cuda` is `None`, a CPU-only wheel was installed. Reinstall the intended CUDA wheel inside the active virtual environment.

Do **not** install a full CUDA Toolkit as the first troubleshooting step.

## `CUDA out of memory`

This machine has 4 GB of VRAM. In later training phases, reduce `batch_size` before changing architecture or system software. Close other GPU-heavy applications when running experiments and monitor usage with:

```powershell
nvidia-smi -l 2
```

## `pip` installs packages outside `.venv`

Verify:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Both paths should refer to `.venv`. Prefer `python -m pip` rather than a bare `pip` command.

## `pytest` fails with `PermissionError` in the Windows user temp directory

Observed error:

```text
PermissionError: [WinError 5] Access is denied: C:\\Users\\<user>\\AppData\\Local\\Temp\\pytest-of-<user>
```

**Meaning:** pytest can run the project tests, but Windows denies access to pytest's default temporary-directory root. This is an environment/ACL issue, not an audio-model or CUDA failure.

The repository configures pytest to use a project-local temporary directory instead:

```text
.pytest_tmp/
```

This directory is ignored by Git and is recreated as needed. The normal test command remains:

```powershell
python -m pytest
```

For diagnosis, the equivalent one-off command is:

```powershell
python -m pytest --basetemp=.pytest_tmp
```

If the one-off command passes while the default command fails, verify that the repository contains the current `pyproject.toml` setting:

```toml
[tool.pytest.ini_options]
addopts = "-q --basetemp=.pytest_tmp"
```

Do not change CUDA, PyTorch, or project source code to solve this specific error.
