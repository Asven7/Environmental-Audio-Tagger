# Phase 11F — Frozen Held-Out Results Aggregation

## Status

**USER VERIFIED — FROZEN SIX-RUN AGGREGATION COMPLETED**

This step aggregates the six frozen evaluation artifacts. It performs no inference, threshold tuning, model selection, or change to the Phase-10 evaluation definitions.

## Frozen input matrix

```text
CNN  × seeds 13, 23, 37
CRNN × seeds 13, 23, 37
```

All six runs are required.

## Integrity checks

For every run the aggregator verifies:

```text
checkpoint SHA-256 against experiment_freeze.json
threshold SHA-256 against experiment_freeze.json
evaluation SHA-256 against .frozen_evaluation.lock.json
evaluation protocol version
checkpoint/threshold provenance
validation-manifest provenance
known-test manifest provenance
OOD-test manifest provenance
model name
experiment seed
class order
known/OOD sample counts
```

Any mismatch aborts aggregation.

## Aggregation rule

For each model, the three predeclared seeds are summarized using:

```text
n
mean
sample standard deviation (ddof = 1)
minimum
maximum
```

No best-seed substitution is performed.

## Headline known-test metrics

```text
precision_micro
recall_micro
f1_micro
precision_macro
recall_macro
f1_macro
hamming_loss
mAP
```

## OOD/rejection metrics

```text
known_false_rejection_rate
known_acceptance_rate
ood_rejection_rate
ood_false_acceptance_rate
```

## Additional summaries

The same three-seed aggregation is produced for:

```text
per-class precision / recall / F1 / average precision
single-vs-mixture groups
relative-dB groups
overlap-ratio groups
joint relative-dB × overlap groups
```

## Outputs

Git-ignored experiment outputs:

```text
frozen_test_summary.json
frozen_test_headline.csv
frozen_test_per_class.csv
frozen_test_groups.csv
```

## Verified final headline

```text
CNN:
  mAP       0.618628 ± 0.008281
  F1 micro  0.549715 ± 0.004061
  F1 macro  0.564576 ± 0.003953

CRNN:
  mAP       0.727378 ± 0.007039
  F1 micro  0.592859 ± 0.010359
  F1 macro  0.617833 ± 0.013461
```

The full final interpretation is recorded in [`PHASE_11_FINAL_RESULTS.md`](PHASE_11_FINAL_RESULTS.md).

## Scientific interpretation

These are final held-out results for the frozen protocol and must be reported across all three predeclared seeds.

After observing them:

```text
no threshold retuning
no architecture change based on held-out performance
no preprocessing change based on held-out performance
no metric-definition change based on held-out performance
```

Any later scientific development requires a newly declared evaluation protocol.
