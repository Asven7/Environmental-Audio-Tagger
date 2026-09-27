# Environmental Audio Tagger

A reproducible undergraduate Computer Engineering project for **window-level multi-label environmental audio tagging** using a lightweight CNN baseline and a CNN+GRU (CRNN) model.

The repository turns a proposal about real-time environmental sound analysis into a runnable local system with:

- leakage-safe dataset manifests,
- controlled two-source synthetic mixtures,
- shared training/inference Log-Mel preprocessing,
- CNN and CRNN models,
- validation-only threshold tuning,
- multi-label evaluation,
- held-out-class rejection analysis,
- file and microphone-oriented inference paths,
- runtime benchmarking,
- a Gradio demonstration UI,
- automated tests,
- synthetic demo data and demo checkpoints for engineering verification.

> **Terminology:** the implemented task is window-level multi-label audio tagging / event-presence detection. It does **not** estimate exact onset/offset times and should not be described as full Sound Event Detection (SED).

## 1. Project Scope

### Core MVP

1. Read file-based environmental audio.
2. Convert audio to mono, resample, crop/pad to a fixed window, and apply bounded RMS normalization.
3. Extract Log-Mel spectrograms.
4. Train a CNN baseline.
5. Train a lightweight CRNN (CNN + unidirectional GRU).
6. Train with multi-label `BCEWithLogitsLoss`.
7. Tune one decision threshold per class using validation data only.
8. Evaluate with micro/macro Precision, Recall, F1, mAP, and Hamming Loss.
9. Evaluate controlled two-label mixtures under different relative levels and temporal-overlap ratios.
10. Measure batch=1 feature/inference/total compute time.
11. Run window-level inference over files and expose a local demonstration UI.

### Secondary Features

- Held-out-class rejection analysis (`no_confident_known_class`).
- Optional continuous microphone inference via `sounddevice`.
- Multi-seed experiment runner and mean/std aggregation.
- Synthetic engineering demo dataset.

### Explicitly Out of Scope

- exact event onset/offset estimation,
- source separation,
- source counting,
- sound localization,
- microphone arrays / beamforming,
- general open-set recognition,
- large AudioSet-scale training,
- cloud production deployment,
- authentication, database, REST API, microservices, message queues, Kubernetes.

These exclusions are intentional to keep the project technically meaningful and realistic for an undergraduate timeline.

## 2. Architecture

### Training

```text
UrbanSound8K metadata
        |
        v
fixed folds + target/held-out classes
        |
        v
single-source manifests -----> no-source-leakage check
        |
        +--> controlled two-source mixture manifests
        |
        v
waveform loading / resampling / fixed window
        |
        v
bounded RMS normalization
        |
        v
Log-Mel spectrogram
        |
        +--> CNN baseline
        |
        +--> CNN + GRU (CRNN)
        |
        v
BCEWithLogitsLoss
        |
        v
validation-only model selection + per-class thresholds
        |
        v
frozen test evaluation
```

### Inference

```text
File or microphone stream
        |
        v
fixed-size overlapping windows
        |
        v
same waveform normalization + Log-Mel transform used in training
        |
        v
trained CNN/CRNN
        |
        v
sigmoid scores
        |
        v
per-class thresholds
        |
        +--> active known labels
        +--> no_confident_known_class
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

## 3. Technology Stack

- **Python 3.11–3.13** — main implementation language.
- **PyTorch** — model definition, training, checkpointing, inference.
- **torchaudio** — Log-Mel feature extraction.
- **SoundFile + SciPy** — robust local file loading and resampling.
- **NumPy / pandas** — waveform operations and experiment manifests.
- **scikit-learn** — multi-label metrics and threshold evaluation.
- **Gradio** — simple local university-demo UI.
- **pytest** — automated tests.
- **sounddevice** — optional live microphone stream; not required for core training/evaluation.

No database is used because the project data is immutable experiment data and manifests; CSV/JSON files are simpler, inspectable, and sufficient. No backend/API layer is used because inference is local and adding one would not improve the research objective.

## 4. Repository Structure

```text
environmental-audio-tagger/
├── config/
│   ├── default.yaml          # UrbanSound8K experiment configuration
│   └── demo.yaml             # fast synthetic smoke-test configuration
├── src/esaudio/
│   ├── audio.py              # loading, resampling, cropping, normalization, mixing
│   ├── checkpoints.py        # checkpoint and threshold persistence
│   ├── cli.py                # train/evaluate/infer entry points
│   ├── config.py             # YAML loading and validation
│   ├── dataset.py            # manifest-backed PyTorch dataset + augmentation
│   ├── demo_data.py          # safe synthetic engineering demo data
│   ├── evaluate_runner.py    # frozen-checkpoint evaluation orchestration
│   ├── evaluation.py         # multi-label metrics, thresholds, rejection metrics
│   ├── features.py           # shared Log-Mel extractor
│   ├── inference.py          # file/window inference
│   ├── manifests.py          # UrbanSound8K split/mix manifest generation
│   ├── models.py             # CNN and CRNN
│   ├── runtime.py            # batch=1 runtime benchmark
│   ├── streaming.py          # tested overlapping stream buffer
│   └── training.py           # deterministic training pipeline
├── scripts/
│   ├── prepare_urbansound8k.py
│   ├── generate_demo_data.py
│   ├── run_demo_pipeline.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── infer_file.py
│   ├── benchmark_runtime.py
│   ├── run_experiments.py
│   ├── summarize_experiments.py
│   ├── run_ui.py
│   └── live_microphone.py
├── tests/                    # automated unit/integration tests
├── docs/                     # architecture, decisions, evaluation, traceability, defense guide
├── sample_data/demo/         # generated synthetic demo audio + manifests
├── artifacts/demo_cnn/       # verified synthetic CNN smoke artifact
├── artifacts/demo_crnn/      # verified synthetic CRNN smoke artifact
├── .gitattributes          # cross-platform line-ending policy
├── pyproject.toml
├── requirements-lock.txt
└── README.md
```

## 5. Prerequisites

Recommended:

- Python 3.11, 3.12, or 3.13
- 8+ GB RAM
- CPU is sufficient for the lightweight models
- CUDA GPU is optional for faster real-data training

The build environment used to verify this repository had Python 3.13.5, PyTorch 2.10.0 CPU, torchaudio 2.10.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, Gradio 6.5.1, and pytest 9.0.2.

## 6. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
```

Install the project:

```bash
pip install -e '.[ui,dev]'
```

For optional continuous microphone inference:

```bash
pip install -e '.[ui,live,dev]'
```

If you are in an offline environment where all dependencies are already installed, editable installation can be done without dependency resolution/build isolation:

```bash
pip install -e . --no-deps --no-build-isolation
```

## 7. Immediate Demo Without UrbanSound8K

This repository includes a **synthetic engineering smoke dataset** and tiny demo checkpoints. These are only for verifying the software pipeline; they are **not research results** and must not be reported as environmental-audio accuracy.

Run a file prediction:

```bash
python scripts/infer_file.py \
  --checkpoint artifacts/demo_crnn/best_model.pt \
  --thresholds artifacts/demo_crnn/thresholds.json \
  --audio sample_data/demo/audio/test/demo_tone_180/000.wav
```

Launch the local UI:

```bash
python scripts/run_ui.py \
  --checkpoint artifacts/demo_crnn/best_model.pt \
  --thresholds artifacts/demo_crnn/thresholds.json
```

Then open the local address printed by Gradio (normally `http://127.0.0.1:7860`).

To regenerate and re-run the complete engineering smoke pipeline:

```bash
python scripts/run_demo_pipeline.py
```

## 8. Preparing UrbanSound8K

The repository intentionally does not redistribute UrbanSound8K. Obtain the dataset from its official source and preserve its original directory layout:

```text
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

Generate fixed manifests:

```bash
python scripts/prepare_urbansound8k.py \
  --dataset-root /path/to/UrbanSound8K \
  --config config/default.yaml \
  --output-dir artifacts/manifests
```

Default research split:

- train folds: 1–7
- validation fold: 8
- test folds: 9–10

Default target classes:

- air_conditioner
- children_playing
- dog_bark
- drilling
- engine_idling
- jackhammer
- siren
- car_horn

Default held-out classes used only for limited rejection analysis:

- gun_shot
- street_music

All values are configurable, but once a real experiment begins, class choices and folds should be frozen and documented before examining test results.

## 9. Training

CNN baseline:

```bash
python scripts/train_model.py \
  --config config/default.yaml \
  --model cnn \
  --train-manifest artifacts/manifests/known_train.csv \
  --val-manifest artifacts/manifests/known_val.csv \
  --audio-root /path/to/UrbanSound8K \
  --output-dir artifacts/cnn_seed13 \
  --seed 13
```

CRNN:

```bash
python scripts/train_model.py \
  --config config/default.yaml \
  --model crnn \
  --train-manifest artifacts/manifests/known_train.csv \
  --val-manifest artifacts/manifests/known_val.csv \
  --audio-root /path/to/UrbanSound8K \
  --output-dir artifacts/crnn_seed13 \
  --seed 13
```

Each run produces:

- `best_model.pt`
- `thresholds.json`
- `history.csv`
- `training_summary.json`

Model selection uses validation mAP. Per-class thresholds are selected **after loading the best checkpoint** by maximizing per-class F1 on known validation samples only. The test split remains untouched until final evaluation.

## 10. Multi-Seed Research Experiment

The default configuration specifies three seeds: `13, 23, 37`.

```bash
python scripts/run_experiments.py \
  --config config/default.yaml \
  --manifests-dir artifacts/manifests \
  --audio-root /path/to/UrbanSound8K \
  --output-root artifacts/experiments
```

Aggregate mean and standard deviation:

```bash
python scripts/summarize_experiments.py \
  --experiment-index artifacts/experiments/experiment_index.json
```

## 11. Evaluation

```bash
python scripts/evaluate_model.py \
  --config config/default.yaml \
  --checkpoint artifacts/crnn_seed13/best_model.pt \
  --thresholds artifacts/crnn_seed13/thresholds.json \
  --known-manifest artifacts/manifests/known_test.csv \
  --ood-manifest artifacts/manifests/ood_test.csv \
  --audio-root /path/to/UrbanSound8K \
  --split test \
  --output artifacts/crnn_seed13/test_evaluation.json
```

The evaluator reports:

- micro Precision / Recall / F1
- macro Precision / Recall / F1
- mAP
- Hamming Loss
- per-class metrics
- single vs mixed sample metrics
- metrics grouped by relative dB level
- metrics grouped by overlap ratio
- held-out rejection recall
- false rejection rate on known inputs

`no_confident_known_class` means that none of the trained classes crossed its selected threshold. It is deliberately **not** called a complete unknown-sound detector.

## 12. Runtime Benchmark

```bash
python scripts/benchmark_runtime.py \
  --checkpoint artifacts/crnn_seed13/best_model.pt \
  --thresholds artifacts/crnn_seed13/thresholds.json \
  --manifest artifacts/manifests/known_test.csv \
  --audio-root /path/to/UrbanSound8K \
  --max-samples 100 \
  --output artifacts/crnn_seed13/runtime.json
```

Engineering acceptance criterion:

```text
p95 total compute time per window < stream hop duration
```

This criterion establishes that the processing chain does not accumulate backlog. It is distinct from initial latency, which necessarily includes collecting the first audio window.

## 13. Continuous Microphone Inference

Install the optional `live` dependency and ensure PortAudio/device permissions are available:

```bash
python scripts/live_microphone.py \
  --checkpoint artifacts/crnn_seed13/best_model.pt \
  --thresholds artifacts/crnn_seed13/thresholds.json
```

The tested `StreamingWindowBuffer` is independent of the microphone backend. Physical microphone operation depends on the local OS/audio device and therefore must be verified on the demonstration machine.

## 14. Testing

Run all automated tests:

```bash
pytest
```

The suite covers:

- configuration validation,
- audio normalization/cropping/mixing,
- feature extraction,
- CNN/CRNN forward passes,
- threshold tuning/metrics,
- overlapping stream-window generation,
- synthetic data/manifests and no-source-leakage logic,
- checkpoint round-trip and end-to-end inference.

## 15. Reproducibility Rules

1. Freeze target/held-out classes before final experiments.
2. Freeze fold assignments before final experiments.
3. Generate mixtures only from source files belonging to the same split.
4. Never tune on test data.
5. Store model checkpoint, threshold file, config, seed, and results together.
6. Run at least three seeds for the final CNN/CRNN comparison.
7. Report mean ± standard deviation across seeds.
8. Record the reference CPU/GPU/OS/Python/PyTorch versions used for runtime measurements.
9. Do not report synthetic demo metrics as UrbanSound8K performance.

## 16. Privacy and Security

- No credentials or external API keys are required.
- The application does not upload audio to an external service.
- Microphone audio is processed locally and is not saved by the live CLI.
- The Gradio server defaults to `127.0.0.1` and `share=False`.
- Do not expose the demo UI publicly without adding appropriate access controls and upload limits.

See [`docs/SECURITY_PRIVACY.md`](docs/SECURITY_PRIVACY.md).

## 17. Known Limitations

- UrbanSound8K is originally single-label; multi-label training/evaluation relies on controlled synthetic mixtures.
- Clip labels do not guarantee perfect event presence in every fixed crop.
- Synthetic mixing cannot fully represent real soundscapes.
- Held-out-class threshold rejection is not general open-set recognition.
- A known + unknown mixture can still be reported only as the known class.
- Continuous microphone accuracy requires separately annotated real recordings for quantitative claims.
- The default project does not estimate exact event boundaries.

## 18. Documentation

- [`docs/SCOPE_AND_ACCEPTANCE.md`](docs/SCOPE_AND_ACCEPTANCE.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/DECISIONS.md`](docs/DECISIONS.md)
- [`docs/EVALUATION.md`](docs/EVALUATION.md)
- [`docs/REQUIREMENTS_TRACEABILITY.md`](docs/REQUIREMENTS_TRACEABILITY.md)
- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)
- [`docs/SECURITY_PRIVACY.md`](docs/SECURITY_PRIVACY.md)
- [`docs/DEFENSE_DEMO.md`](docs/DEFENSE_DEMO.md)
- [`docs/IMPLEMENTATION_REPORT_FA.md`](docs/IMPLEMENTATION_REPORT_FA.md)
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md)

## 19. What Counts as Project Completion?

The real research project is complete when:

1. UrbanSound8K manifests are generated and the leakage check passes.
2. CNN and CRNN train successfully for the frozen protocol.
3. At least three seeds per model are completed.
4. Test metrics are generated only after tuning is finished.
5. Controlled relative-level and overlap groups are reported.
6. Runtime p95 is measured on a documented reference machine and is below the hop duration.
7. File-based end-to-end inference works.
8. Microphone streaming is demonstrated locally or explicitly documented as unavailable due to hardware/OS constraints.
9. Tests pass.
10. The final report clearly distinguishes controlled synthetic-mixture evidence from real-world claims.

## 20. License

MIT for the code in this repository. Third-party datasets and libraries retain their own licenses/terms.
