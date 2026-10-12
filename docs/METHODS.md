# Model and evaluation methods

This file describes the implementation represented by `config/default.yaml` and `src/esaudio/`. It does not replace the complete research records or introduce additional experiments.

## Audio and features

The research path converts decoded audio to mono, resamples to **22,050 Hz**, extracts **2.0-second windows**, and uses a **1.0-second hop** for sequential inference. Training selects a crop using an energy-based strategy; validation/test crops use a deterministic center strategy by default. Short waveforms are right-padded to the required window size.

Preprocessing uses bounded RMS normalization; training can add deterministic gain, temporal shift and noise augmentation. Synthetic two-source combinations are normalized and mixed according to their manifest parameters. The relative decibel sign indicates the first source's level minus the second source's level.

The feature extractor computes power Mel spectrograms, takes a log transform and optionally standardizes features per window. Research parameters:

| Parameter | Value |
|---|---:|
| Sample rate | 22,050 Hz |
| FFT length (`n_fft`) | 1,024 |
| Feature hop | 512 samples |
| Mel bands | 64 |
| Minimum/maximum frequency | 20 / 10,000 Hz |
| Normalization | Enabled |

The same saved feature settings are reconstructed when loading a model checkpoint.

## Architectures and training

The **CNN** uses convolutional blocks with channel widths `[16, 32, 64]`, frequency reduction in the frontend, global pooling and a linear classification layer.

The **CRNN** reuses the frontend, averages over frequency and applies a single-layer unidirectional **GRU with 64 hidden units** before temporal pooling and classification. Default dropout is `0.2`. Both models output independent class logits.

The shared objective is binary cross-entropy with logits (`BCEWithLogitsLoss`); optional class-positive weights address target imbalance. Training uses Adam with learning rate `0.001`, weight decay `0.0001`, batch size `32`, at most `30` epochs, and early-stopping patience `5`. The best epoch for each run is selected using **validation mAP**, never a held-out test score.

The official comparison consists of six predeclared runs: `cnn` and `crnn`, each with experiment seeds `13`, `23`, and `37`. A separate `data_seed=1234` is recorded in the configuration. Train-source membership and mixture recipe generation are fixed before experiments. Training/validation manifests are hashed for provenance.

## Thresholds, selection, and freezing

Sigmoid converts logits into class scores. A configured grid of possible thresholds (`0.10` through `0.90` in `0.05` increments) is searched **independently for each known class on validation data** to maximize binary F1. In a tie, the threshold nearest `0.5` wins; equal-distance ties choose the smaller threshold. The selected threshold file records associated manifest/checkpoint hashes and class order.

After training and threshold selection, the six checkpoint/threshold pairs are frozen before running the held-out evaluation. The final **deployment seed** within the CRNN family is `23`, selected using its highest validation mAP within the specified family. The previously reported held-out outcomes must not be used to change thresholds or select a different seed.

## Metrics and interpretation

The primary ranking metric is **mean average precision (mAP)** across known classes. Threshold-dependent outputs include micro/macro precision, recall, F1, Hamming loss and per-class values. Scores are not established as calibrated probabilities.

For limited rejection analysis, a window is marked `no_confident_known_class` when none of the eight known scores crosses its class-specific threshold. The reported held-out rejection rate measures that rule on samples from two withheld UrbanSound8K categories. A high false acceptance rate prevents any claim of robust OOD detection.

The recorded comparisons distinguish single-source clips from controlled mixtures, and group mixture performance by overlap ratio and relative source level. These are observations about this constructed dataset, not universal properties of environmental sounds.

## Reproduction boundary

Research processing is implemented in `scripts/run_official_experiments.py`, `scripts/select_experiment_thresholds.py`, `scripts/run_frozen_evaluation.py` and `scripts/summarize_frozen_experiments.py`. Ordinary installation and CI should **not** rerun the frozen held-out evaluation. Any new model variant or revised preprocessing needs a newly planned study with a fresh evaluation boundary.
