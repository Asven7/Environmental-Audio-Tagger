# Phase 11 — Official Multi-Seed Experiments

## Status

**USER VERIFIED — OFFICIAL SIX-RUN EXPERIMENT COMPLETED AND FROZEN**

The official matrix was frozen as:

```text
CNN  × seeds 13, 23, 37
CRNN × seeds 13, 23, 37
= 6 training runs
```

All six runs, validation-only threshold selections, the pre-test freeze, frozen held-out evaluations, and final aggregation were completed.

The authoritative final scientific results are in [`PHASE_11_FINAL_RESULTS.md`](PHASE_11_FINAL_RESULTS.md).

## Why a dedicated official runner exists

The generic Phase-9 runner is safe for train/validation experimentation, but Phase 11 required stronger safeguards around the final scientific campaign:

- exact six-run matrix;
- clean Git requirement;
- recorded Git commit;
- config and manifest fingerprints;
- Git-ignored artifact root;
- resumable completed runs without silently overwriting them;
- incremental training index after each completed run;
- threshold selection only after all six training runs were complete;
- a pre-test freeze artifact.

## Protocol-lock requirement

The official training runner refuses to start with a dirty Git working tree.

This makes the exact code revision part of the experiment plan and prevents scientific training from silently running against uncommitted protocol changes.

## Official training artifacts

Before the first run, the runner writes:

```text
artifacts/experiments_phase11/experiment_plan.json
```

containing:

```text
protocol version
Git commit
config SHA-256
train-manifest SHA-256
validation-manifest SHA-256
models
seeds
device
run matrix
```

Each run directory contains:

```text
best_model.pt
history.csv
training_summary.json
```

No threshold tuning occurs during training.

`training_index.json` is updated after each completed run.

## Resume behavior

`--resume` skips only runs that have both a checkpoint and training summary and whose model/seed/manifest hashes satisfy the frozen plan.

A partially written run directory is not guessed or silently overwritten.

## Validation-only threshold pass

After all six training runs completed:

```text
scripts/select_experiment_thresholds.py
```

selected one threshold vector per checkpoint from validation data only.

Per run:

```text
thresholds.json
threshold_selection.json
```

Matrix-level:

```text
threshold_index.json
experiment_freeze.json
```

The freeze artifact records hashes of all six checkpoints and all six threshold artifacts.

## Validation-only aggregation

`scripts/summarize_validation_experiments.py` reports across the three seeds per model:

```text
best validation mAP
best epoch
epochs run
training elapsed seconds
validation F1 micro
validation F1 macro
per-class selected thresholds
```

These are validation/development statistics, not held-out test performance.

## Completed validation results

Validation-only mean ± sample standard deviation:

```text
CNN:
  mAP       0.615137 ± 0.008819
  F1 micro  0.572758 ± 0.005473
  F1 macro  0.598761 ± 0.008029

CRNN:
  mAP       0.669377 ± 0.006753
  F1 micro  0.592500 ± 0.013290
  F1 macro  0.629206 ± 0.010054
```

CRNN seed 23 achieved the highest CRNN validation mAP and was later selected for deployment using validation information only.

## Held-out boundary

Training, threshold selection, and validation aggregation do not use the known-test or OOD-test sets for development.

Only after `experiment_freeze.json` existed was the frozen Phase-10 held-out evaluation workflow used.

## Frozen scientific result

Final three-seed held-out headline:

```text
CNN mAP  = 0.618628 ± 0.008281
CRNN mAP = 0.727378 ± 0.007039
```

The held-out result is final for this protocol.

No post-test:

```text
threshold retuning
architecture change
preprocessing change
split/class change
metric-definition change
best-seed substitution
```

is permitted while claiming the same untouched final test protocol.
