# Phase 10 — Threshold Selection and Frozen Evaluation Protocol

## Status

**USER VERIFIED**

Phase 10 freezes how validation-selected thresholds are created and how final known-test
and OOD-test evaluation must be executed.

No real test-set evaluation was performed while verifying this phase.

## Frozen threshold-selection rule

For each target class independently:

```text
threshold[c] = argmax F1[c] on KNOWN VALIDATION
```

using the configured grid:

```text
0.10, 0.15, 0.20, ..., 0.90
```

Validation threshold tuning requires at least one positive and one negative example for
every class.

Tie-breaking is frozen as:

```text
1. threshold closest to 0.5
2. if equally distant, lower threshold
```

The grid is sorted and deduplicated internally so caller ordering cannot alter the result.

## Metric boundary

Best training epoch:

```text
validation mAP
```

Threshold selection:

```text
per-class validation F1
```

Frozen test reporting:

```text
precision micro/macro
recall micro/macro
F1 micro/macro
mAP
Hamming loss
per-class metrics
single-vs-mixture metrics
relative-dB groups
overlap-ratio groups
joint relative-dB × overlap groups
OOD rejection metrics
```

## Threshold artifact provenance

Scientific threshold artifacts are bound to:

```text
exact checkpoint SHA-256
exact validation-manifest SHA-256
class order
threshold grid
selection split = val
selection metric = per_class_f1
protocol version
validation sample count
model name
experiment seed
best epoch
```

Strict evaluation rejects mismatched checkpoint/threshold or validation/threshold pairs.

## OOD interpretation

The project uses the transparent rule:

```text
no known class crosses its own threshold
→ reject as "no confident known class"
```

Reported metrics include:

```text
known_false_rejection_rate
known_acceptance_rate
ood_rejection_rate
ood_false_acceptance_rate
```

`ood_recall_rejected` remains as a backward-compatible alias for
`ood_rejection_rate`.

This is not a claim of general open-set recognition.

## Frozen-evaluation provenance

Every strict evaluation result records hashes for:

```text
checkpoint
threshold artifact
validation manifest
known evaluation manifest
OOD evaluation manifest
```

plus model/seed, class order, thresholds, and threshold-selection metadata.

## Test protection

The canonical final-test command is:

```text
scripts/run_frozen_evaluation.py
```

It requires strict threshold provenance, refuses to overwrite an existing result, and
writes:

```text
.frozen_evaluation.lock.json
```

beside the checkpoint after a successful frozen evaluation.

The generic evaluation CLI also protects test evaluation by requiring frozen mode and a
saved output artifact.

## User verification

Dedicated evaluation-protocol tests:

```text
python -m pytest tests/test_evaluation.py tests/test_evaluation_protocol.py -v
→ 15 passed
```

Full repository regression:

```text
python -m pytest
→ 109 passed
```

Validation-safe protocol smoke test:

```text
tiny_train_samples=12
tiny_val_samples=12
tiny_ood_val_samples=8

known_test_read=False
ood_test_read=False

Validation threshold selection: PASSED
classes=8

Threshold provenance binding: PASSED
Strict validation-only evaluation: PASSED
OOD validation rejection plumbing: PASSED
Checkpoint/threshold mismatch rejection: PASSED

best_epoch=1
validation_samples=12

Phase 10 protocol smoke test: PASSED
Known test manifest was NOT read.
OOD test manifest was NOT read.
Temporary smoke artifacts were deleted.
```

## Interpretation

The tiny smoke run is an engineering verification only. Its thresholds and validation
metrics are not scientific project results.

The verified claims are:

```text
validation-only threshold selection works
threshold artifacts bind to exact checkpoint and validation manifest
strict evaluation rejects mismatched provenance
OOD rejection plumbing works on validation-safe data
test/OOD-test files remain untouched
full repository regression remains healthy
```

## Scientific boundary

Phase 10 does not run the final multi-seed experiment campaign and does not expose real
known-test or OOD-test results.

The next scientific steps must preserve this protocol unchanged.
