# Phase 8 — CRNN Architecture

## Status

**USER VERIFIED**

Phase 8 freezes and verifies the lightweight convolutional-recurrent architecture.
No scientific training was performed in this phase.

## Frozen default configuration

```text
cnn_channels       = [16, 32, 64]
recurrent_type     = gru
recurrent_hidden   = 64
recurrent_layers   = 1
dropout            = 0.2
num_classes        = 8
bidirectional      = False
```

## Frozen tensor geometry

```text
Log-Mel input              [B, 1, 64, 87]
shared CNN frontend        [B, 64, 8, 87]
frequency mean             [B, 64, 87]
recurrent sequence input   [B, 87, 64]
GRU output                 [B, 87, 64]
temporal mean              [B, 64]
classifier output          [B, 8] raw logits
```

The 87-frame time axis is preserved by the convolutional frontend and passed to the
one-directional recurrent module.

## Output semantics

The model returns raw logits and contains no Sigmoid layer.

Training uses:

```text
BCEWithLogitsLoss
```

Inference applies Sigmoid externally before thresholding.

## Parameter counts

Verified default GRU CRNN:

```text
48,888 trainable parameters
```

Optional one-layer LSTM variant:

```text
57,208 trainable parameters
```

The CNN baseline from Phase 7 has 23,928 parameters.

## User verification

Dedicated CRNN tests:

```text
python -m pytest tests/test_crnn.py -v
→ 14 passed
```

Compatibility regression:

```text
python -m pytest tests/test_models.py tests/test_features_models.py tests/test_checkpoint_inference.py -v
→ 13 passed
```

Full repository regression:

```text
python -m pytest
→ 84 passed
```

Architecture smoke test:

```text
sample_rate=22050
samples=44100
batch_size=4
feature_shape=(4, 1, 64, 87)
sequence_input_shape=(4, 87, 64)
recurrent_output_shape=(4, 87, 64)
logit_shape=(4, 8)
recurrent=gru hidden=64 layers=1 bidirectional=False
parameter_count=48888
raw_logit_range=(-0.068107, 0.122735)
external_sigmoid_score_range=(0.482980, 0.530645)
CPU forward/sequence/determinism: PASSED
CUDA forward/finite check: PASSED
CUDA max_abs_diff=0.000011
CRNN architecture smoke test: PASSED
```

No training was performed and no checkpoint was written by the smoke test.

## Scientific boundary

This phase verifies architecture correctness only. It does not establish accuracy, F1,
mAP, optimal recurrent type, optimal hidden size, or CRNN superiority over CNN. Those
claims require the later reproducible training and frozen evaluation phases.
