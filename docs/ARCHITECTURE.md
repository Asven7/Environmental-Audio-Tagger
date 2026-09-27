# Architecture

## Architectural Style

The project is intentionally a **single local Python application/research repository**, not a distributed system. Training, evaluation, inference, and UI reuse the same package modules.

No database, REST API, authentication service, queue, cloud service, or microservice is required.

## Components

### 1. Configuration (`config.py` + YAML)

Responsibilities:
- load project parameters,
- validate required values,
- keep dataset/model/training/evaluation choices explicit.

Input: YAML file.

Output: validated Python dictionary.

### 2. Manifest Builder (`manifests.py`)

Responsibilities:
- map official UrbanSound8K folds to train/validation/test,
- keep target and held-out classes separate,
- build single-sample records,
- build deterministic controlled two-source mixture records,
- enforce source-level split isolation.

No mixed audio is permanently materialized; manifests describe how to build mixtures on demand.

### 3. Waveform Processing (`audio.py`)

Responsibilities:
- read audio safely,
- downmix to mono,
- resample,
- crop/pad to fixed windows,
- bounded RMS normalization,
- controlled two-source mixing,
- peak safety.

### 4. Dataset (`dataset.py`)

Responsibilities:
- turn manifest rows into model-ready waveform/target pairs,
- generate mixtures on demand,
- use energy crop for training and fixed center crop for evaluation by default,
- apply deterministic train-only augmentation.

### 5. Feature Extraction (`features.py`)

`LogMelExtractor` is the only primary learned-model representation.

```text
waveform -> MelSpectrogram(power) -> log -> per-window feature standardization
```

The exact same class is serialized by configuration and reconstructed for inference.

### 6. Models (`models.py`)

#### CNN baseline

```text
Log-Mel
 -> ConvBlock(16)
 -> ConvBlock(32)
 -> ConvBlock(64)
 -> global mean over frequency/time
 -> dropout
 -> linear logits
```

Pooling only reduces frequency inside the convolutional frontend.

#### CRNN

```text
Log-Mel
 -> same convolutional frontend
 -> mean over frequency
 -> [time, channel] sequence
 -> unidirectional GRU
 -> temporal mean pooling
 -> dropout
 -> linear logits
```

The default recurrent unit is GRU because it is smaller than an equivalent LSTM and adequate for the undergraduate scope. The code supports LSTM as a configuration-level experiment without changing the architecture layer.

### 7. Training (`training.py`)

Responsibilities:
- deterministic seeding,
- optional positive-class weighting,
- Adam optimization,
- validation mAP model selection,
- early stopping,
- checkpoint persistence,
- post-selection threshold tuning using validation data only.

### 8. Evaluation (`evaluation.py`, `evaluate_runner.py`)

Responsibilities:
- thresholded multi-label metrics,
- per-class metrics,
- mAP,
- Hamming Loss,
- controlled-condition breakdowns,
- held-out rejection behavior.

### 9. Inference (`inference.py`)

Responsibilities:
- reconstruct a model from a checkpoint,
- reconstruct the exact feature extractor,
- load thresholds with class-order validation,
- predict a single window,
- predict overlapping windows across an entire file,
- report timings and scores.

### 10. Streaming (`streaming.py`)

`StreamingWindowBuffer` receives arbitrary chunks and emits fixed windows at a fixed hop. It contains no hardware dependency, which makes it fully unit-testable.

`scripts/live_microphone.py` provides the thin optional `sounddevice` adapter.

### 11. UI (`scripts/run_ui.py`)

Local-only Gradio interface:
- upload an audio file or record from the browser microphone,
- analyze successive windows,
- display active labels and class scores,
- plot score trajectories.

This UI is deliberately separate from the scientific evaluation pipeline.

## Communication Between Components

All communication is in-process Python calls and files:

- YAML configuration,
- CSV manifests,
- `.pt` model checkpoints,
- JSON threshold/evaluation files,
- CSV training history.

This makes the project easy to inspect, reproduce, and defend academically.

## Data Flow Invariants

1. Original source files do not cross train/validation/test splits.
2. Mixtures are created only from sources in the same split.
3. Test data is never passed to training or threshold tuning.
4. Model outputs are logits; sigmoid is applied for inference/evaluation.
5. Thresholds are saved separately and class-order checked when loaded.
6. “No confident known class” is not equated with universal unknown detection.

## Deployment

Primary deployment is local:

- training: CPU or optional CUDA GPU,
- inference: CPU target,
- UI: localhost,
- microphone: OS audio stack through optional `sounddevice`.

Docker is intentionally not the primary path because direct microphone/GUI/GPU device mapping would make a small academic project less portable rather than more portable.
