# Troubleshooting

This document records recurring development problems and the smallest reliable fix.

## `py` command is not recognized

The Windows Python Launcher is not installed or is not on `PATH`.

This project does not require it. Use:

```powershell
python -m ...
```

and verify:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
```

## `where.exe python` shows `WindowsApps` and `msys64`

Check the interpreter that is actually executing:

```powershell
python -c "import sys; print(sys.executable)"
```

If it resolves to `C:\msys64\...`, use a native Windows CPython 3.11 installation for the verified Windows workflow.

## PowerShell refuses to activate `.venv`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This changes only the current process policy.

## `ModuleNotFoundError: No module named 'esaudio'`

The repository uses a `src/` package layout.

For a developer checkout:

```powershell
python -m pip install -e ".[all]"
python -m pytest
```

For a clean/non-editable install:

```powershell
python -m pip install ".[all]"
python -m pytest
```

Do not manually edit `PYTHONPATH` as the first fix.

## `torch.cuda.is_available()` returns `False`

First:

```powershell
nvidia-smi
```

Then:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.version.cuda` is `None`, a CPU-only wheel is installed. Install the intended CUDA wheel inside the active `.venv`.

Do **not** install a standalone CUDA Toolkit as the first troubleshooting step.

## `CUDA out of memory`

The verified development GPU has 4 GB VRAM.

Reduce `training.batch_size` before changing architecture or system software, close other GPU-heavy applications, and monitor:

```powershell
nvidia-smi -l 2
```

Do not change the frozen deployed model after held-out evaluation merely to work around a local runtime environment.

## `pip` installs outside `.venv`

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Both should point into `.venv`.

Prefer `python -m pip` to a bare `pip`.

## `pytest` fails with Windows temp `PermissionError`

Observed class of error:

```text
PermissionError: [WinError 5] Access is denied:
C:\Users\<user>\AppData\Local\Temp\pytest-of-<user>
```

The repository uses:

```text
.pytest_tmp/
```

through:

```toml
[tool.pytest.ini_options]
addopts = "-q --basetemp=.pytest_tmp"
```

Diagnostic equivalent:

```powershell
python -m pytest --basetemp=.pytest_tmp
```

Do not change CUDA, the model, or signal-processing code for this filesystem ACL problem.

## Gradio or sounddevice version lookup fails

Do not rely on package-specific `.version` or `__version__` attributes.

Use:

```powershell
python -c "from importlib.metadata import version; print(version('gradio')); print(version('sounddevice'))"
```

The verified development versions are:

```text
Gradio 6.29.1
sounddevice 0.5.6
```

## `sounddevice` cannot open the microphone

List devices:

```powershell
python scripts\live_microphone.py --list-devices
```

Check Windows microphone privacy/permission settings and confirm the selected device supports input.

The local live CLI was verified with a C-Media microphone at:

```text
22050 Hz
1 channel
float32
```

A device-specific failure is not evidence that the streaming core is broken.

## Live microphone produces false positives

This is a known scientific limitation, not necessarily a microphone bug.

The frozen CRNN held-out rejection rate is only about 4.69%, and quiet-room/fan audio produced false positive known classes during Phase-13 verification.

Do **not** retune frozen thresholds on live/demo behavior.

Correct wording:

```text
No confident known class
```

is a weak threshold heuristic, not robust open-set recognition.

## Clean install fails

Follow [`CLEAN_INSTALL.md`](CLEAN_INSTALL.md) exactly:

```powershell
python -m pip check
python scripts\verify_clean_install.py --project-root . --output artifacts\phase16_clean_install_report.json
python -m pytest
```

The verified fresh-clone acceptance did not require UrbanSound8K or frozen Phase-11 research artifacts.

## Git shows generated artifacts

Check:

```powershell
git status --short
git check-ignore -v artifacts\phase16_clean_install_report.json
```

Real datasets, normal checkpoints, and experiment outputs must remain ignored. Only the deliberately small synthetic demo artifacts are tracked.

## Final rule

Troubleshooting installation, UI, microphone, or CI must not reopen the frozen scientific protocol. If a future model/preprocessing change is desired, declare a new experiment protocol first.
