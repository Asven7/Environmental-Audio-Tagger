# Phase 11F — Frozen Held-Out Results Aggregation

## Status

**IMPLEMENTED — WAITING FOR USER LOCAL VERIFICATION**

This step aggregates the six already-generated frozen evaluation artifacts. It performs
no inference, no threshold tuning, no model selection, and no change to the Phase-10
evaluation definitions.

## Frozen input matrix

```text
CNN  × seeds 13, 23, 37
CRNN × seeds 13, 23, 37
```

The aggregator requires all six runs.

## Integrity checks before aggregation

For every run it verifies:

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

No best-seed selection is performed.

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

```text
frozen_test_summary.json
frozen_test_headline.csv
frozen_test_per_class.csv
frozen_test_groups.csv
```

All outputs live under the already Git-ignored experiment artifact directory.

## Scientific interpretation

These are final held-out results for the frozen protocol. They must be reported across
all three predeclared seeds. Individual seed results may be inspected diagnostically, but
must not replace the multi-seed mean ± standard deviation as the primary result.

After these results exist:

```text
no threshold retuning
no architecture change based on held-out performance
no preprocessing change based on held-out performance
no metric-definition change based on held-out performance
```

Any new development cycle after inspecting held-out results would require a newly
declared evaluation protocol and should not be described as the same untouched final
test.
