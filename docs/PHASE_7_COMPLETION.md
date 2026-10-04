# Phase 7 Completion Record — CNN Baseline

## Final status

**USER VERIFIED**

Phase 7 is technically complete and ready for its Git milestone.

## Verified checks

- Dedicated CNN tests: **11 passed**
- Existing feature/model and checkpoint compatibility tests: **2 passed**
- Full project regression: **70 passed**
- Default CNN trainable parameters: **23,928**
- Phase-6 feature input: **(4, 1, 64, 87)**
- CNN shared frontend output: **(4, 64, 8, 87)**
- CNN logits output: **(4, 8)**
- CPU eval forward determinism: **PASSED**
- CUDA forward/finite check: **PASSED**
- CPU/CUDA maximum absolute logit difference: **0.000008**
- Model training performed: **No**
- Checkpoint written by smoke test: **No**

## Frozen architecture contract

- Three convolutional blocks with channels `[16, 32, 64]`
- Each block uses Conv2d, BatchNorm2d, ReLU, frequency-only MaxPool `(2,1)`, Dropout2d
- Shared frontend preserves the time axis
- CNN baseline uses global average pooling after the shared frontend
- Final layer outputs eight raw logits
- No Sigmoid is embedded inside the model
- BCEWithLogitsLoss is the intended training loss
- Sigmoid is applied only in inference before thresholding

## Compatibility

The existing public model APIs remain available:

```text
CNNTagger
CRNNTagger
build_model
model_from_spec
count_parameters
ModelSpec
```

The existing feature/model and checkpoint/inference tests passed after the Phase-7 changes.

## Next phase boundary

Phase 8 may harden and verify the CRNN architecture. Phase 7 itself does not start CRNN
training, hyperparameter selection, threshold tuning, or scientific evaluation.
