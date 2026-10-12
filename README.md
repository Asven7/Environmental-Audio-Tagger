# Environmental Audio Tagger

Environmental Audio Tagger is a Python application for **window-level, multi-label tagging of environmental sounds**. It processes short audio windows and assigns scores to eight trained sound categories. The system implements a convolutional baseline (CNN) and a convolutional-recurrent model (CRNN, using a GRU); it also supports analysis of audio files and local microphone input.

**Task boundary:** The model estimates the presence of classes in a window. It does **not** locate exact event onset/offset times, separate audio sources, count sources, or provide reliable general-purpose recognition of unknown sounds.

## Research configuration

| Setting | Value |
|---|---|
| Dataset | UrbanSound8K (obtained separately) |
| Sample rate | 22,050 Hz |
| Analysis window / hop | 2.0 s / 1.0 s |
| Features | Log-Mel; FFT 1024, hop 512, 64 Mel bands |
| Split | folds 1–7 train; fold 8 validation; folds 9–10 test |
| Training seeds | 13, 23, 37 |
| Models | CNN; CRNN with unidirectional GRU |
| Training loss | `BCEWithLogitsLoss` |
| Threshold selection | Per-class F1 on validation data only |

The eight trained categories are `air_conditioner`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `jackhammer`, `siren`, and `car_horn`. `gun_shot` and `street_music` are reserved for a limited held-out rejection analysis, **not** trained as known categories.

Multi-label examples are largely created by mixing two different UrbanSound8K sound classes using a fixed, documented protocol. Source splits are established before mixing, and recordings that cross the configured splits are excluded. This construction does not fully represent uncontrolled acoustic environments.

## Reported held-out results

The following figures come from the project's **recorded three-seed research evaluation**, not from the synthetic audio shipped in GitHub. Values are mean ± sample standard deviation over seeds 13, 23 and 37.

| Metric | CNN | CRNN |
|---|---:|---:|
| mAP | 0.618628 ± 0.008281 | **0.727378 ± 0.007039** |
| Micro-F1 | 0.549715 ± 0.004061 | **0.592859 ± 0.010359** |
| Macro-F1 | 0.564576 ± 0.003953 | **0.617833 ± 0.013461** |
| Hamming loss (lower is better) | 0.194428 ± 0.000791 | **0.173433 ± 0.008472** |

The CRNN obtained better aggregate results on this fixed evaluation. However, **its mean rejection rate for held-out classes was only 4.6895%**, so the threshold-only rejection rule should not be treated as a reliable out-of-distribution (OOD) detector. Scores are not calibrated probabilities.

Full per-class, subset and runtime summaries appear in [Results](docs/RESULTS.md). The original full experimental checkpoints and raw reports are **not** part of the public GitHub checkout; see [Reproducibility](docs/REPRODUCIBILITY.md) for the evidence boundary.

## Installation — Windows, CPU

Use Python 3.11 in a fresh terminal:

```powershell
git clone https://github.com/Asven7/Environmental-Audio-Tagger.git
cd Environmental-Audio-Tagger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<2.11" "torchaudio>=2.6,<2.11"
python -m pip install ".[all]"
python -m pip check
python -m pytest
```

For CUDA installation and troubleshooting, see [Reproducibility](docs/REPRODUCIBILITY.md).

## Try the repository-contained example

The repository contains small **synthetic tone** examples and two small example checkpoints. These test the software workflow; they are **not evidence of environmental-sound classification accuracy**.

```powershell
python scripts\infer_file.py --checkpoint artifacts\demo_crnn\best_model.pt --thresholds artifacts\demo_crnn\thresholds.json --audio sample_data\demo\audio\test\demo_tone_180\000.wav
```

Launch the local file/microphone interface with the example checkpoint:

```powershell
python scripts\run_ui.py --checkpoint artifacts\demo_crnn\best_model.pt --thresholds artifacts\demo_crnn\thresholds.json --device cpu
```

To use the selected **research** checkpoint instead, obtain the corresponding locally stored frozen experiment directory and run:

```powershell
python scripts\run_ui.py --experiment-root artifacts\experiments_phase11 --device cpu
```

The research weights are not downloaded automatically. Further commands: [Usage](docs/USAGE.md).

## Project organization

- `src/esaudio/`: audio loading, manifests, feature extraction, models, training, metrics, inference and streaming.
- `scripts/`: research pipeline, CLI utilities, local UI, microphone input and verification.
- `config/`: research and synthetic-example configuration.
- `tests/`: unit, integration, protocol and documentation checks.
- `sample_data/demo/` and `artifacts/demo_*/`: small synthetic fixtures for local software verification.
- `docs/`: methods, results, installation, operation and limitations.

Start with the [documentation index](docs/README.md) for detailed explanations.

## Reproducibility and limitations

The public CI executes repository-contained software checks but does not download UrbanSound8K, access private local research checkpoints, or rerun the held-out scientific evaluation. The full evaluation therefore cannot be independently reconstructed from the public checkout alone.

The data splits, model selection, thresholds and held-out results are treated as **frozen**. Work that changes the method requires a separately defined experimental protocol rather than retuning against the already observed test set. A separately annotated real-world multi-label test set would be needed to quantify live-microphone accuracy.

## License

Source code in this repository is provided under [MIT](LICENSE). UrbanSound8K and third-party packages remain subject to their own terms; the dataset is not redistributed here.
