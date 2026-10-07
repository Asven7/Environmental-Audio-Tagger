# Environmental Audio Tagger

[![CI](https://github.com/Asven7/Environmental-Audio-Tagger/actions/workflows/ci.yml/badge.svg)](https://github.com/Asven7/Environmental-Audio-Tagger/actions/workflows/ci.yml)

A reproducible undergraduate Computer Engineering project for **window-level multi-label environmental audio tagging / event-presence detection** using a CNN baseline and a lightweight CNN+GRU (CRNN) model.

> **Scope:** this project predicts which trained environmental-sound classes are present in each short analysis window. It does **not** estimate exact event onset/offset times and must not be described as full Sound Event Detection (SED).

## Project status

The research and engineering pipeline is frozen and verified through:

- leakage-safe UrbanSound8K split/manifests,
- controlled two-source multi-label mixtures,
- shared Log-Mel preprocessing,
- CNN and CRNN training,
- validation-only model selection and per-class threshold tuning,
- one-time frozen held-out evaluation,
- three-seed aggregation,
- frozen deployment selection,
- CPU/CUDA runtime benchmarking,
- file and microphone inference,
- Gradio demo UI,
- repository QA,
- GitHub Actions CI,
- fresh-clone / fresh-virtual-environment installation verification.

The final deployment family is **CRNN**, and deployment seed **23** was selected using validation mAP only.

## Scientific protocol

Default research protocol:

```text
Dataset: UrbanSound8K
Task: window-level multi-label audio tagging / event-presence detection
Sample rate: 22,050 Hz
Window: 2.0 s
Hop: 1.0 s
Features: Log-Mel spectrogram
n_fft: 1024
feature hop: 512
n_mels: 64

Train folds: 1-7
Validation fold: 8
Test folds: 9-10

Research seeds: 13, 23, 37
```

Target classes:

```text
air_conditioner
children_playing
dog_bark
drilling
engine_idling
jackhammer
siren
car_horn
```

Held-out classes used for a limited rejection analysis:

```text
gun_shot
street_music
```

The held-out classes are **not** evidence of general open-set recognition.

## Models

### CNN baseline

```text
Log-Mel
  -> convolutional frontend
  -> global pooling
  -> linear multi-label logits
```

### CRNN

```text
Log-Mel
  -> convolutional frontend
  -> frequency mean pooling
  -> unidirectional GRU
  -> temporal mean pooling
  -> linear multi-label logits
```

Training uses raw logits with `BCEWithLogitsLoss`. Sigmoid is applied only for scores at inference/evaluation time.

## Frozen held-out results

The final research comparison used three seeds per model. Thresholds were selected on validation data only and frozen before held-out test evaluation.

| Model | mAP | F1 micro | F1 macro | Precision micro | Recall micro | Hamming loss |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.618628 ± 0.008281 | 0.549715 ± 0.004061 | 0.564576 ± 0.003953 | 0.477551 ± 0.001776 | 0.647589 ± 0.008033 | 0.194428 ± 0.000791 |
| CRNN | **0.727378 ± 0.007039** | **0.592859 ± 0.010359** | **0.617833 ± 0.013461** | **0.520813 ± 0.018149** | **0.688571 ± 0.010842** | **0.173433 ± 0.008472** |

CRNN improved mean held-out mAP by about **0.10875** over the CNN baseline.

### Rejection limitation

The threshold-based `No confident known class` state is only a weak heuristic:

```text
CNN held-out rejection rate:  ~1.27%
CRNN held-out rejection rate: ~4.69%
```

Equivalently, held-out false acceptance remained very high. This project therefore does **not** claim robust unknown-sound or general open-set recognition.

## Frozen deployment

The deployment model was selected using validation performance only:

```text
model: CRNN
seed: 23
validation mAP: 0.6757137110
```

The research checkpoint and threshold artifacts are local experiment outputs and are not required for repository installation or CI.

When the frozen local artifacts are available, the UI resolves the deployment from:

```text
artifacts/experiments_phase11
```

The frozen per-class thresholds for the selected deployment are:

```text
[0.20, 0.70, 0.75, 0.80, 0.65, 0.35, 0.60, 0.85]
```

These thresholds must not be changed based on held-out test or live-demo behavior.

## Runtime

Canonical batch-1 frozen runtime measurements on the verified development machine:

| Device | Mean total compute | p95 total compute | 1 s hop backlog criterion |
|---|---:|---:|---:|
| CPU | 3.435 ms | **4.358 ms** | PASS |
| RTX 3050 Ti Laptop GPU | 1.424 ms | **1.561 ms** | PASS |

The first prediction still requires collecting the initial 2-second audio window. The runtime result shows that subsequent per-hop processing does not accumulate backlog; it does not mean zero end-to-end latency.

## Quick start: clean CPU installation on Windows

The clean-install baseline was verified with **Python 3.11.9** in a fresh clone and fresh virtual environment.

```powershell
git clone https://github.com/Asven7/Environmental-Audio-Tagger.git
cd Environmental-Audio-Tagger

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

python -m pip install `
  --index-url https://download.pytorch.org/whl/cpu `
  "torch>=2.6,<2.11" `
  "torchaudio>=2.6,<2.11"

python -m pip install ".[all]"
python -m pip check
python -m pytest
```

Phase-16 fresh-clone acceptance completed with:

```text
pip check: PASS
clean-install verifier: PASS
full pytest: 173 passed
working tree: clean
```

See [`docs/CLEAN_INSTALL.md`](docs/CLEAN_INSTALL.md) for the complete verified procedure.

## Verify an installation without the dataset

After installation:

```powershell
python scripts\verify_clean_install.py `
  --project-root . `
  --output artifacts\phase16_clean_install_report.json
```

The verifier checks installed dependencies, console entry points, optional UI/live imports, and synthetic CPU forward passes for both CNN and CRNN. It does not require UrbanSound8K or frozen experiment artifacts.

## Synthetic engineering demo

The repository contains a synthetic demo path for software verification. Synthetic demo outputs are **not research accuracy results**.

A file can be processed with the demo CRNN artifact when the demo files are present:

```powershell
python scripts\infer_file.py `
  --checkpoint artifacts\demo_crnn\best_model.pt `
  --thresholds artifacts\demo_crnn\thresholds.json `
  --audio sample_data\demo\audio\test\demo_tone_180\000.wav
```

The demo UI can be launched with explicit demo artifacts:

```powershell
python scripts\run_ui.py `
  --checkpoint artifacts\demo_crnn\best_model.pt `
  --thresholds artifacts\demo_crnn\thresholds.json `
  --device cpu
```

## Frozen file and microphone demo

With the local frozen research artifacts available:

```powershell
python scripts\run_ui.py `
  --experiment-root artifacts\experiments_phase11 `
  --device cpu
```

The UI supports:

- sequential file-window inference,
- browser microphone streaming,
- latest per-class scores and frozen thresholds,
- timestamped history,
- newest/oldest history ordering,
- stop-with-history-preservation,
- explicit clear-results behavior.

For the local `sounddevice` microphone CLI:

```powershell
python scripts\live_microphone.py `
  --experiment-root artifacts\experiments_phase11 `
  --device cpu
```

List available input devices with:

```powershell
python scripts\live_microphone.py --list-devices
```

Microphone predictions are qualitative demo evidence unless separately evaluated against annotated real recordings.

## Dataset and leakage policy

UrbanSound8K is not redistributed by this repository.

The research protocol freezes train/validation/test folds before controlled mixing. Two broad `fsID` groups were found to span configured research splits and were excluded from all research manifests to avoid source leakage.

Controlled mixtures:

```text
two different known target classes
same research split only
relative level: -6 / 0 / +6 dB
overlap ratio: 0.25 / 0.50 / 1.00
```

The split occurs **before mixing**.

## Testing and CI

Run the complete local suite with:

```powershell
python -m pytest
```

GitHub Actions runs a CPU repository CI job on pushes to `main` and pull requests. The hosted CI intentionally does not require:

```text
UrbanSound8K
local frozen experiment artifacts
microphone hardware
CUDA
repository secrets
```

CI is an engineering regression gate, not a replacement for the frozen scientific evaluation.

## Repository layout

```text
Environmental-Audio-Tagger/
|-- .github/workflows/       # GitHub Actions CI
|-- config/                  # default + demo configurations
|-- docs/                    # protocol, results, QA, reproducibility, defense docs
|-- scripts/                 # research, inference, demo, QA, install-verification scripts
|-- src/esaudio/             # core Python package
|-- tests/                   # unit/integration/contract tests
|-- sample_data/demo/        # synthetic engineering demo data
|-- artifacts/               # local/generated outputs; selected demo artifacts may exist
|-- pyproject.toml
`-- README.md
```

## Reproducibility boundary

The scientific test result is frozen.

Do **not** use the frozen held-out test results to:

```text
retune thresholds
select a new model or seed
change preprocessing
change class definitions
change the split/mixing protocol
claim a newly improved test score
```

Do **not rerun the frozen held-out evaluation** as part of ordinary installation, CI, UI work, documentation work, or defense preparation.

Any future model/preprocessing development must declare a new experimental protocol before another held-out evaluation.

## Known limitations

- UrbanSound8K is originally single-label; multi-label evidence is based largely on controlled synthetic mixtures.
- Synthetic mixtures do not fully represent natural soundscapes.
- Clip labels do not guarantee perfect event presence in every fixed crop.
- Scores are sigmoid outputs and are **not claimed to be calibrated probabilities**.
- Threshold-based rejection performs poorly on held-out classes and is not general open-set recognition.
- Live microphone audio can produce false positives, particularly under distribution shift.
- Exact onset/offset localization is outside project scope.
- Source separation, source counting, and localization are outside project scope.
- Quantitative live-microphone accuracy would require separately annotated real recordings.

## Documentation

Start with [`docs/README.md`](docs/README.md).

Key documents:

- [`docs/CLEAN_INSTALL.md`](docs/CLEAN_INSTALL.md) — verified fresh installation procedure.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture and data flow.
- [`docs/SCOPE_AND_ACCEPTANCE.md`](docs/SCOPE_AND_ACCEPTANCE.md) — scope and acceptance boundaries.
- [`docs/PHASE_11_FINAL_RESULTS.md`](docs/PHASE_11_FINAL_RESULTS.md) — frozen multi-seed research results.
- [`docs/PHASE_12_RUNTIME_RESULTS.md`](docs/PHASE_12_RUNTIME_RESULTS.md) — frozen runtime measurements.
- [`docs/PHASE_13_END_TO_END_DEMO.md`](docs/PHASE_13_END_TO_END_DEMO.md) — demo acceptance.
- [`docs/PHASE_14_QA.md`](docs/PHASE_14_QA.md) — repository QA gate.
- [`docs/PHASE_15_GITHUB_CI.md`](docs/PHASE_15_GITHUB_CI.md) — CI design.
- [`docs/PHASE_16_COMPLETION.md`](docs/PHASE_16_COMPLETION.md) — clean-install and documentation closure.
- [`docs/DEFENSE_DEMO.md`](docs/DEFENSE_DEMO.md) — defense/demo guidance.
- [`docs/IMPLEMENTATION_REPORT_FA.md`](docs/IMPLEMENTATION_REPORT_FA.md) — Persian implementation report.

## Security and privacy

- No external API keys are required.
- The application performs local inference.
- The local microphone CLI does not intentionally persist microphone audio.
- The Gradio demo defaults to local hosting rather than public sharing.
- Do not expose the local demo publicly without adding suitable authentication, upload limits, and deployment hardening.

See [`docs/SECURITY_PRIVACY.md`](docs/SECURITY_PRIVACY.md).

## License

MIT for the code in this repository. Third-party datasets and libraries retain their own licenses and terms.
