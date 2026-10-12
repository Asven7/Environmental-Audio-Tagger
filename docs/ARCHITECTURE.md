# Software architecture

Environmental Audio Tagger is a local Python project with a `src/` package layout. The training, evaluation and inference commands share implementations instead of maintaining separate model definitions. There is no application database or hosted inference API.

## Data path

```text
UrbanSound8K audio + metadata
    -> source/split checks and CSV manifest recipes (manifests.py)
    -> on-demand fixed audio windows / controlled mixtures (audio.py, dataset.py)
    -> Log-Mel spectrograms (features.py)
    -> CNNTagger or CRNNTagger (models.py)
    -> window-level logits
    -> sigmoid class scores and frozen class thresholds
```

The `config/default.yaml` file defines the research sample rate, window geometry, features, classes, model sizes and training/evaluation settings. `config/demo.yaml` defines a smaller synthetic workflow and must not be confused with the research protocol.

## Main modules

| Module | Responsibility |
|---|---|
| `config.py` | Read and validate YAML configuration |
| `manifests.py` | UrbanSound8K metadata validation, split isolation and balanced mixture recipes |
| `audio.py` | Audio I/O, downmix, resample, crop/pad, normalization and mixing |
| `dataset.py` | Load manifest examples, synthesize mixtures and apply training augmentation |
| `features.py` | `LogMelExtractor` based on `torchaudio` |
| `models.py` | Convolutional frontend, CNN and CNN+GRU architecture |
| `training.py` | Seeding, mini-batch optimization, early stopping and checkpoint selection |
| `checkpoints.py` | Model/threshold serialization and class-order checks |
| `evaluation.py`, `evaluate_runner.py` | mAP, multi-label F1, rejection and evaluation artifacts |
| `experiments.py`, `frozen_results.py` | Experiment index and six-run result validation/aggregation |
| `deployment.py` | Select a CRNN research run using validation mAP |
| `inference.py` | Single-window and whole-file model predictions |
| `streaming.py`, `ui_support.py` | Audio-window buffering and local interaction state |
| `runtime_profile.py` | CPU/CUDA processing-time measurement |

## Model structure

The CNN receives a Log-Mel tensor, passes it through a convolutional frontend and uses global pooling plus a linear classification head. The default frontend has channel widths 16, 32 and 64.

The CRNN uses the same convolutional frontend. It averages across frequency, feeds the remaining temporal sequence into a unidirectional GRU (64 hidden units, one recurrent layer in the default configuration), averages over time and emits one logit per target class.

Both model heads produce **eight independent logits**, not a mutually exclusive softmax distribution. Training uses `BCEWithLogitsLoss`; score reporting applies sigmoid and independently selected class thresholds. Details: [Methods](METHODS.md).

## Entry points and execution

Console entry points declared in `pyproject.toml` are `esaudio-train`, `esaudio-evaluate` and `esaudio-infer`. Research-specific scripts orchestrate manifest preparation, a predeclared seed/model matrix, threshold selection and frozen aggregation.

The local UI (`scripts/run_ui.py`) accepts file uploads and browser microphone chunks. `scripts/live_microphone.py` uses the operating system's audio devices through `sounddevice`. These use the same inference model and thresholds. No network service is needed for the model itself; the Gradio interface listens on localhost by default.

## Scope boundaries

The output is **window-level presence**, not the onset and offset of individual sounds. The threshold-only "no confident known class" state is not a reliable unknown-class detector. Long-file inference exists through multiple code paths; their treatment of a final incomplete window is being reviewed separately and should not be assumed identical in every entry point.

See [Dataset](DATASET.md), [Usage](USAGE.md), and [Results](RESULTS.md) for assumptions and recorded performance.
