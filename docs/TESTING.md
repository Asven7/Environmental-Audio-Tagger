# Testing Guide

## General rule

Run tests from the repository root with the project virtual environment active:

```powershell
python -m pytest
```

The repository configures a local `.pytest_tmp` base directory to avoid the Windows user-temp ACL problem encountered in Phase 1.

## Current verified repository baseline

At the end of Phase 16B:

```text
documentation contract: 8 passed
full repository suite: 181 passed
```

Phase 16C changes documentation/metadata consistency only and must preserve a completely green full suite. Later phases may add tests, so the exact count can increase.

## Important focused suites

### Configuration / audio

```powershell
python -m pytest tests/test_config.py -v
python -m pytest tests/test_audio.py -v
python -m pytest tests/test_audio_io.py -v
```

These protect config validation, crop/pad semantics, normalization, mixing, file decoding, mono conversion, resampling, and save/load behavior.

### Features / models

```powershell
python -m pytest tests/test_features.py tests/test_models.py tests/test_crnn.py -v
```

These protect the shared Log-Mel representation and CNN/CRNN tensor contracts.

### Scientific protocol

```powershell
python -m pytest `
  tests/test_experiment_protocol.py `
  tests/test_evaluation_protocol.py `
  tests/test_frozen_results.py `
  -v
```

These protect validation-only selection, frozen evaluation provenance, and multi-seed result aggregation.

### Runtime / streaming / UI

```powershell
python -m pytest `
  tests/test_runtime_protocol.py `
  tests/test_streaming.py `
  tests/test_ui_support.py `
  tests/test_ui_stop_preservation.py `
  tests/test_phase13_demo_acceptance.py `
  -v
```

### Repository QA / CI / clean install / documentation

```powershell
python -m pytest `
  tests/test_phase14_qa.py `
  tests/test_ci_contract.py `
  tests/test_clean_install_contract.py `
  tests/test_final_documentation_contract.py `
  -v
```

## Full regression rule

After any focused suite:

```powershell
python -m pytest
```

Do not commit a phase if it breaks an earlier verified test.

## Historical baselines

Earlier phase counts remain useful as historical checkpoints, not current totals. For example Phase 3 ended with:

```text
test_config.py        8 passed
test_audio.py        13 passed
test_audio_io.py      4 passed
test_demo_dataset.py  1 passed
full suite            31 passed
```

Those numbers should not be confused with the current repository-wide regression count.

## Windows temp-directory failure

If pytest reports a `PermissionError` under:

```text
C:\Users\<user>\AppData\Local\Temp\pytest-of-<user>
```

verify that `pyproject.toml` still contains:

```toml
[tool.pytest.ini_options]
addopts = "-q --basetemp=.pytest_tmp"
```

The normal command should remain:

```powershell
python -m pytest
```

## Scientific-test boundary

Repository tests and CI must not silently rerun the frozen held-out scientific experiment. The held-out result is a frozen research artifact, not a development regression target.

Tests may verify protocol logic, artifact hashes/metadata, synthetic fixtures, and engineering paths without reopening the scientific test set.
