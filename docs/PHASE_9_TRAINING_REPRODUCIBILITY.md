# Phase 9 — Training and Reproducibility

## Status

**USER VERIFIED**

Phase 9 hardens and verifies the train/validation pipeline. It intentionally does not
perform frozen test/OOD evaluation. Research threshold selection remains deferred to
Phase 10.

## Frozen training configuration

```text
batch_size              = 32
max_epochs              = 30
learning_rate           = 0.001
weight_decay            = 0.0001
early_stopping_patience = 5
use_pos_weight          = true
data_seed               = 1234
experiment_seeds        = [13, 23, 37]
device                  = auto
optimizer               = Adam
loss                    = BCEWithLogitsLoss
```

## Reproducibility contract

Phase 9 controls:

```text
Python random
NumPy RNG
PyTorch CPU RNG
PyTorch CUDA RNG
deterministic algorithm request
cuDNN deterministic mode
cuDNN benchmark disabled
CUBLAS workspace configuration
dedicated DataLoader shuffle generator
```

The training DataLoader uses an experiment-seed-specific generator so sample order is
independent of architecture-specific random-number consumption during model
initialization.

`num_workers=0` remains fixed in this phase.

## Data / augmentation boundary

Training augmentation is deterministic per:

```text
(data_seed, sample_id, epoch)
```

and is applied only to the training split.

Validation uses deterministic evaluation preprocessing and no train-time augmentation.

`pos_weight` is computed only from the training target matrix:

```text
negative_count / positive_count
```

and training fails clearly if a target class has no positive training examples.

## Early stopping and model selection

Model selection uses validation mAP only.

```text
train epoch
→ validation inference
→ validation mAP
→ save if improved
→ early stop after configured patience
```

The selected checkpoint is therefore the best validation checkpoint, not necessarily the
last epoch.

## Frozen-test discipline

The research experiment runner is now train/validation only.

```text
Phase 9 research runner:
known_train   → used
known_val     → used
known_test    → not read
ood_test      → not read
threshold tuning → off
```

Threshold selection and frozen test/OOD evaluation are intentionally deferred to Phase 10.

## User verification

Dedicated training protocol tests:

```text
python -m pytest tests/test_training_protocol.py -v
→ 11 passed
```

Full repository regression:

```text
python -m pytest
→ 95 passed
```

Real tiny training / reproducibility smoke test:

```text
tiny_train_samples=24
tiny_val_samples=12
seed=13
test_manifests_used=False
threshold_tuning=False

CPU same-seed reproducibility: PASSED
best_epoch=2
best_validation_mAP=0.32705965909090906

Threshold isolation: PASSED
Manifest fingerprints: PASSED
train fingerprint prefix=11e42e5abb53
validation fingerprint prefix=a16909606d31

CUDA tiny CRNN training: PASSED
device=NVIDIA GeForce RTX 3050 Ti Laptop GPU
best_epoch=1

cpu_epochs_ran=2
cpu_stop_reason=max_epochs
thresholds_tuned=false

Phase 9 training protocol smoke test: PASSED
No test/OOD manifest was read.
No threshold tuning or frozen test evaluation was performed.
Temporary smoke-training artifacts were deleted.
```

## Interpretation

The smoke-test mAP value is not a scientific performance result. The smoke training uses
only a tiny subset and two CPU epochs, so its purpose is engineering verification only.

The validated claims from this phase are:

```text
same-seed CPU training is reproducible
train/validation separation is respected
threshold tuning can be disabled for research runs
test/OOD manifests are excluded from the Phase-9 research runner
CUDA backward/optimizer/checkpoint path works
```

## Scientific boundary

Phase 9 does not report final CNN/CRNN performance, choose final thresholds, evaluate the
test split, evaluate OOD rejection, or run the complete three-seed experiment campaign.

Those activities remain downstream of the frozen evaluation protocol.
