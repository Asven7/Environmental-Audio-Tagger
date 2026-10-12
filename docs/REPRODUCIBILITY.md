# Installation and reproducibility

## Python and dependency contract

The project uses a `src/` Python package, configured by `pyproject.toml`. Supported Python range in the checked revision is `>=3.11,<3.14`. The documented development baseline used **Python 3.11.9 on Windows**. `torch`, `torchaudio`, NumPy, pandas, SciPy, soundfile, scikit-learn and PyYAML are core dependencies; Gradio, sounddevice and pytest are optional groups available under `.[all]`.

A historical local development environment included Windows 10 Enterprise, an Intel i7-12700H, 16 GB RAM and an RTX 3050 Ti Laptop GPU. This is a description of a recorded environment, not a minimum hardware requirement.

### Fresh Windows CPU setup

From a directory where Git and Python 3.11 are available:

```powershell
git clone https://github.com/Asven7/Environmental-Audio-Tagger.git
cd Environmental-Audio-Tagger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<2.11" "torchaudio>=2.6,<2.11"
python -m pip install ".[all]"
python -m pip check
python scripts\verify_clean_install.py --project-root . --output artifacts\clean_install_report.json
python -m pytest
```

The verifier checks dependencies, console scripts and synthetic CPU model forwarding. It does not require UrbanSound8K or the original research checkpoints and does not recompute held-out research accuracy. `--basetemp=.pytest_tmp` is configured in `pyproject.toml` to avoid a Windows temporary-directory permission issue observed during development.

### CUDA option

The recorded NVIDIA configuration used PyTorch/torchaudio 2.10.0 with CUDA 12.6 wheels. CUDA compatibility depends on the actual driver and platform. For a compatible Windows installation, the project previously used:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.6,<2.11" "torchaudio>=2.6,<2.11"
```

Check the installed versions and CUDA availability before use:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CPU installation is adequate for running the repository-contained synthetic example and its automated tests.

## Reproducibility tiers

| What is being checked | Required inputs | Supported by public checkout? |
|---|---|---|
| Python packaging and unit tests | Source, dependencies | Yes |
| Synthetic tone inference | Small sample WAVs and example checkpoints | Yes |
| UrbanSound8K manifest preparation | Separate UrbanSound8K download | No, raw dataset is external |
| Frozen six-run research evaluation | Raw dataset, original manifests, checkpoints, thresholds and result records | No, complete local artifacts are not tracked |
| Live device recording | Working local microphone and optional sounddevice | Device-dependent |

Synthetic example outputs are for engineering checks only. They must not be substituted for the documented real-data research results in [Results](RESULTS.md).

## Historical verification records

Repository records document a **173-pass** pytest suite from a fresh-clone/fresh-environment CPU check, and a later **181-pass** local suite following documentation updates. The GitHub Actions CI run for commit `cc8837e996598955cb2a728c73ce932281859a2b` was reported successful. These are **historical verification records**, not claims that the tests were rerun as part of later documentation editing. The latest full-suite result should always be obtained by running `python -m pytest` on the changed checkout.

A public CI job is defined in `.github/workflows/ci.yml`. It installs CPU PyTorch, verifies dependencies, compiles Python files, checks CLI `--help` surfaces and runs pytest. It does **not** access a dataset, checkpoint set or microphone and therefore is not an independent validation of the original scientific metrics.

## Research artifact location and integrity

For the recorded six-run study, files are locally stored beneath:

```text
artifacts/experiments_phase11/
```

They are intentionally Git-ignored. The recorded freeze version is `cnn_crnn_validation_freeze_v1`; its SHA-256 is:

```text
0305a2aa821ed16a5f485cde1951d3720c47fa65dcc9d0ec9befd2e07f63be68
```

The selected deployment was CRNN seed 23, based on validation performance. Recorded SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `crnn_seed23/best_model.pt` | `80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8` |
| `crnn_seed23/thresholds.json` | `788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2` |

Verify the local copies using PowerShell, e.g.:

```powershell
Get-FileHash artifacts\experiments_phase11\crnn_seed23\best_model.pt -Algorithm SHA256
Get-FileHash artifacts\experiments_phase11\crnn_seed23\thresholds.json -Algorithm SHA256
```

The local output directory also includes or has historically included `training_index.json`, `experiment_freeze.json`, and frozen aggregate result files (`frozen_test_summary.json`, `frozen_test_headline.csv`, `frozen_test_per_class.csv`, `frozen_test_groups.csv`). Full independent verification requires preserving these alongside associated manifests and environmental metadata. They are not currently provided through the public repository.

## Dataset retrieval and licenses

Obtain UrbanSound8K from the original source and follow its terms. Do not commit the raw clips or unrelated recordings. Generated manifests contain source paths and metadata, not duplicated corpus audio. See [Dataset](DATASET.md) and [References](REFERENCES.md).

## Practical limits

Dependency version ranges in `pyproject.toml` are not an exact transitive lockfile. Different permitted releases can produce different runtime characteristics and, where kernels vary, potentially different numerical behavior. For rigorous research repetition, preserve exact installed versions, source SHA, data identities and artifact hashes. A documentation cleanup must not alter frozen results, preprocessing, class definitions or historical Git commits.

If external development tools contributed to the implementation, describe that accurately wherever the university's reporting or authorship policy requires it; technical documentation and the Git commit history should not be rewritten to give a false account of contributions.
