# Verification and tests

## Running the suite

After installation from the repository root:

```powershell
python -m pip check
python -m compileall -q src scripts tests
python -m pytest
```

The project uses pytest with its base temporary directory set to `.pytest_tmp` in `pyproject.toml`, reflecting a Windows ACL issue encountered in the recorded setup.

Useful focused checks:

```powershell
python -m pytest tests\test_audio.py tests\test_audio_io.py tests\test_config.py -v
python -m pytest tests\test_models.py tests\test_crnn.py tests\test_features.py -v
python -m pytest tests\test_experiment_protocol.py tests\test_evaluation_protocol.py tests\test_frozen_results.py -v
python -m pytest tests\test_streaming.py tests\test_ui_support.py tests\test_ui_stop_preservation.py -v
python -m pytest tests\test_final_documentation_contract.py -v
```

## Verification layers

| Layer | Verifies | Does not establish |
|---|---|---|
| Unit and integration tests | Audio conversions, window shapes, model forward passes, scoring and state handling | General real-world classification accuracy |
| Protocol tests | Deterministic mixture metadata, source isolation, freeze bookkeeping and aggregation | A rerun of the original training/evaluation |
| Synthetic example tests | End-to-end software plumbing with generated tones | UrbanSound8K or natural-sound scientific performance |
| GitHub Actions | CPU installation, imports, compilation, CLI surfaces and pytest | A working local microphone, CUDA or external data |
| Local recorded engineering checks | Device-specific runtime, UI/microphone acceptance and previously recorded outputs | Independent replication without original files |

## CI

The repository's `.github/workflows/ci.yml` runs on push to `main`, on pull requests and by manual dispatch. It uses an Ubuntu-hosted Python 3.11 CPU environment with read-only repository permissions. The job installs `.[dev,ui]`, runs `pip check`, compiles Python sources, executes the UI and microphone CLI help checks and runs pytest.

Four successful CI runs were observed on the referenced repository by **8 October 2026**, including the run of the audited commit `cc8837e...`. This historical status should not be represented as proof of successful checks on any future changes until those checks actually run.

## Recorded historical local results

A fresh clone with a fresh virtual environment reportedly passed 173 tests and `pip check`. A later local documentation/metadata verification reportedly passed 181 tests. These results were recorded in the earlier project documentation; this page does **not** claim to have independently rerun them.

## Scope of evaluation

The selected seed/checkpoints and the held-out test metrics are frozen. Routine unit tests, packaging fixes, and documentation changes do not justify rerunning the held-out test set or tuning against it. No test in public CI replaces the separate scientific evaluation performed with UrbanSound8K data and original experiment artifacts.

For a clear breakdown of what is and is not independently reproducible from GitHub, see [Reproducibility](REPRODUCIBILITY.md).
