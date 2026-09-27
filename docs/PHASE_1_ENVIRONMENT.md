# Phase 1 — Local Development Environment

## Target machine baseline

User-reported environment on 2026-09-27:

- Windows 10 Enterprise 64-bit
- Python 3.11.9
- Git 2.51.0.windows.1
- Intel Core i7-12700H
- 15.63 GB RAM
- 78.73 GB free space on `C:`
- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- 4096 MiB VRAM
- NVIDIA driver 566.07
- `nvidia-smi` CUDA compatibility report: 12.7

## Decisions

1. Use Python 3.11 for the project virtual environment.
2. Use `python -m pip` instead of bare `pip` to avoid interpreter ambiguity.
3. Confirm the actual executable behind `python` before installing dependencies because `where.exe python` also exposes an MSYS2 interpreter.
4. Retain PyTorch 2.10.0 and torchaudio 2.10.0 for parity with the repository's previously verified build.
5. Prefer the official CUDA 12.6 PyTorch wheel for the target laptop.
6. Keep CPU-only PyTorch as a fully supported fallback.
7. Do not install a standalone CUDA Toolkit, Node.js, Java, Docker, or database software for the core project.
8. Defer Gradio and `sounddevice` until the UI/live-audio phase.

## Phase gate

Local verification on 2026-09-27 confirmed Python 3.11.9 inside `.venv`, PyTorch/torchaudio 2.10.0+cu126, CUDA availability, a real CUDA tensor operation, and `pip check`. The first default pytest run exposed a Windows temp-directory ACL issue; the same suite passed (`13 passed`) when redirected to `.pytest_tmp`. The repository now makes that local temp directory the default for pytest.

Phase 1 is not fully USER VERIFIED until the updated repository also passes the normal command `python -m pytest` on the target machine.

Phase 1 gate items:

- [x] native Windows CPython path confirmed;
- [x] `.venv` created and activated;
- [x] PyTorch/torchaudio installed;
- [x] `torch.cuda.is_available()` checked;
- [x] real CUDA tensor operation passes;
- [x] editable project installation succeeds;
- [x] `python -m pip check` succeeds;
- [ ] updated repository: normal `python -m pytest` succeeds.
