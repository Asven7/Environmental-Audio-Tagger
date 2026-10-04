# Phase 7 — CNN Baseline

## Status

**USER VERIFIED**

Phase 7 freezes and verifies the lightweight CNN baseline architecture. No scientific
training was performed in this phase.

## Frozen input contract

The CNN consumes the Phase-6 Log-Mel tensor:

```text
[B, 1, 64, 87]
```

The implementation also accepts 128 Mel bins without changing the classifier structure.

## Frozen default architecture

```text
[B,1,64,87]
   ↓ Conv3x3 1→16 + BatchNorm + ReLU + MaxPool(2,1) + Dropout2d
[B,16,32,87]
   ↓ Conv3x3 16→32 + BatchNorm + ReLU + MaxPool(2,1) + Dropout2d
[B,32,16,87]
   ↓ Conv3x3 32→64 + BatchNorm + ReLU + MaxPool(2,1) + Dropout2d
[B,64,8,87]
   ↓ Global mean over frequency and time
[B,64]
   ↓ Dropout
[B,64]
   ↓ Linear 64→8
[B,8] raw logits
```

The shared convolutional frontend reduces only the frequency axis. For the default
64-Mel / 87-frame input, the frontend output is:

```text
[B, 64, 8, 87]
```

The time axis remains at 87 frames throughout the shared frontend.

## Output semantics

The CNN returns raw logits. There is no Sigmoid module inside the model.

Training will use:

```text
BCEWithLogitsLoss
```

Inference applies Sigmoid outside the model before thresholding.

## Frozen default parameter count

For:

```text
cnn_channels = [16, 32, 64]
dropout = 0.2
num_classes = 8
```

the verified trainable parameter count is:

```text
23,928
```

## User verification

Verified locally on the project Windows environment:

```text
python -m pytest tests/test_models.py -v
→ 11 passed

python -m pytest tests/test_features_models.py tests/test_checkpoint_inference.py -v
→ 2 passed

python -m pytest
→ 70 passed
```

CNN smoke test:

```text
sample_rate=22050
samples=44100
batch_size=4
feature_shape=(4, 1, 64, 87)
frontend_shape=(4, 64, 8, 87)
logit_shape=(4, 8)
parameter_count=23928
raw_logit_range=(-0.148485, 0.084709)
external_sigmoid_score_range=(0.462947, 0.521165)
CPU forward/determinism: PASSED
CUDA forward/finite check: PASSED
CUDA max_abs_diff=0.000008
CNN baseline smoke test: PASSED
```

No model training and no checkpoint writing occurred in the smoke test.

## Scientific boundary

This phase verifies engineering correctness only. It does not establish CNN accuracy,
F1, mAP, calibration, or superiority/inferiority relative to CRNN. Those claims require
the later reproducible training and frozen evaluation phases.

## Defense rationale

The CNN is intentionally simple and shares the same convolutional frontend that the CRNN
will reuse. This supports a controlled comparison:

```text
same data
same mixtures
same Log-Mel representation
same convolutional frontend
same evaluation protocol
        ↓
CNN
vs
CNN + recurrent temporal modeling
```

This isolates the effect of adding recurrent temporal modeling more cleanly than comparing
two unrelated architectures.
