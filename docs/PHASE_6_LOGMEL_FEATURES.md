# Phase 6 — Log-Mel Feature Extraction

## Scope

This phase freezes the feature-extraction contract used by training, evaluation,
file inference, and later live inference. It does **not** train a CNN/CRNN.

## Baseline representation

The project uses a power Mel spectrogram followed by a natural logarithm:

1. fixed-length mono waveform;
2. STFT-backed Mel power spectrogram (`power=2.0`);
3. `log(max(mel, 1e-6))`;
4. optional per-example standardization over all Mel/time bins;
5. output shape `[batch, 1, n_mels, frames]`.

The main configuration starts with:

- sample rate: 22050 Hz;
- input window: 2 s = 44100 samples;
- `n_fft = 1024`;
- `hop_length = 512`;
- `n_mels = 64` baseline;
- `f_min = 20 Hz`;
- `f_max = 10000 Hz`;
- `center = True`;
- HTK Mel scale, no Mel filter normalization;
- per-example feature standardization enabled.

With `center=True`, 44100 samples, `n_fft=1024`, and `hop_length=512`, the
feature time axis has **87 frames**. Therefore the baseline tensor is
`[B, 1, 64, 87]`.

## Why center=True is explicit

The previous implementation already used `center=True`, but it was implicit and
was not exported in checkpoint feature configuration. Phase 6 preserves that
behavior while making it part of the reproducibility contract. Changing it later
would change the number and alignment of time frames and therefore affect the
CNN/CRNN input geometry.

## Checkpoint compatibility

`LogMelExtractor` keeps the original constructor arguments and adds only optional
arguments whose defaults match the earlier behavior. Old checkpoint feature
configuration dictionaries remain loadable. New checkpoints export all choices
needed to reconstruct the feature transform (`center`, `power`, `log_floor`,
Mel scale and Mel normalization included).

## 64 vs 128 Mel bands

64 Mel bands are the current baseline, not a claimed final winner. The code and
tests support 128 bands without changing the time axis. Any later comparison
between 64 and 128 must be selected using validation data only; the frozen test
set must not choose this hyperparameter.

## Normalization and silence

When feature normalization is enabled, mean and standard deviation are computed
per example over the Mel/time plane. Silence maps to the log floor; numerically flat feature planes are explicitly
mapped to zeros after normalization so no NaN/Inf or amplified round-off values are produced.

## Acceptance criteria

- exact baseline geometry `[B, 1, 64, 87]` for a 2-second 22.05-kHz window;
- deterministic repeated extraction;
- finite silence handling;
- explicit reproducibility settings in exported feature config;
- 64/128 Mel compatibility;
- no trainable parameters inside the extractor;
- existing training/evaluation/inference API remains compatible;
- real UrbanSound8K single and mixture examples produce finite features;
- optional CPU/CUDA feature sanity check passes on a CUDA-capable machine;
- full project regression suite passes.

## Status

USER VERIFIED.


## User-local verification record

Verified on the project Windows environment with Python 3.11.9:

- `tests/test_features.py`: 11 passed;
- existing feature/model compatibility test: 1 passed;
- existing checkpoint/inference compatibility test: 1 passed;
- full project regression: 59 passed;
- real UrbanSound8K Log-Mel smoke test: passed;
- real batch shape: `(4, 1, 64, 87)`;
- CPU repeat extraction: deterministic;
- CUDA shape/finite sanity check: passed;
- observed CPU/CUDA maximum absolute difference: `0.000010`;
- no model training and no feature files were written during the smoke test.
