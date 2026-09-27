# Testing Guide

## General rule

Run tests from the repository root with the project virtual environment active:

```powershell
python -m pytest
```

The repository configures a local `.pytest_tmp` base directory to avoid the Windows user-temp ACL problem encountered during Phase 1.

## Phase-focused tests

### Configuration tests

```powershell
python -m pytest tests/test_config.py -v
```

These tests prove that valid configurations load and invalid experimental settings fail before training.

### Audio signal tests

```powershell
python -m pytest tests/test_audio.py -v
```

These tests cover crop/pad semantics, RMS normalization, silence handling, clipping protection, and controlled source levels.

### Audio file I/O tests

```powershell
python -m pytest tests/test_audio_io.py -v
```

These tests create temporary WAV files, verify stereo-to-mono conversion, verify resampling, and check save/load behavior.

## Full regression test

After phase-specific tests pass:

```powershell
python -m pytest
```

A phase should not be committed if it breaks an already verified test from an earlier phase.

## Interpreting failures

- Failure in `test_config.py`: configuration validation or expected experiment constraints changed.
- Failure in `test_audio.py`: signal-processing semantics changed; do not proceed to dataset/model work until resolved.
- Failure in `test_audio_io.py`: file decoding, resampling, dtype, or temporary-file behavior is broken.
- `PermissionError` involving the Windows user temp directory: verify that the current `pyproject.toml` still contains the repository-local `--basetemp=.pytest_tmp` setting.

## Why these are unit tests

These tests intentionally avoid UrbanSound8K and the GPU. They verify deterministic low-level behavior in isolation. Dataset integration and learned-model behavior are tested in later phases.

## Dataset integration regression discovered in Phase 3

A full-suite run found that `dataset.py` still imports `apply_global_time_shift`.
The phase-specific audio tests had not exercised that import path, so the missing
compatibility helper was detected only by the regression suite. The helper now has
explicit unit tests for positive, negative, zero, oversized, and invalid shifts.
This is an example of why phase-specific tests and full regression tests are both
required.

## Phase 3 verified baseline

On the target Windows development laptop, Phase 3 ended with:

```text
python -m pytest tests/test_config.py -v       -> 8 passed
python -m pytest tests/test_audio.py -v        -> 13 passed
python -m pytest tests/test_audio_io.py -v     -> 4 passed
python -m pytest tests/test_demo_dataset.py -v -> 1 passed
python -m pytest                               -> 31 passed
```

This is the regression baseline that later phases must preserve or intentionally update.
