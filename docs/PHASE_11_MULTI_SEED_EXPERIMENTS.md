# Phase 11 — Official Multi-Seed Experiments

## Status

**IMPLEMENTED — WAITING FOR USER LOCAL VERIFICATION OF THE EXPERIMENT PROTOCOL**

Phase 11 is the first scientific experiment campaign. The official matrix is frozen as:

```text
CNN  × seeds 13, 23, 37
CRNN × seeds 13, 23, 37
= 6 training runs
```

The held-out known-test and OOD-test sets remain locked until all six training runs and
all six validation-selected threshold artifacts have been completed and frozen.

## Why a dedicated official runner exists

The generic Phase-9 runner is safe for train/validation experimentation, but Phase 11
needs stronger safeguards around the final scientific campaign:

- exact six-run matrix;
- clean Git requirement;
- recorded Git commit;
- config and manifest fingerprints;
- Git-ignored artifact root;
- resumable completed runs without silently overwriting them;
- incremental training index after each completed run;
- threshold selection only after all six training runs are complete;
- a pre-test freeze artifact.

## Protocol-lock commit

The official training runner refuses to start with a dirty Git working tree.

Therefore the Phase-11 code itself must be tested and committed before the six scientific
training runs are launched. This is intentional: the exact code revision is part of the
experiment plan.

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

Each run directory contains the normal Phase-9 artifacts, including:

```text
best_model.pt
history.csv
training_summary.json
```

No threshold tuning occurs during training.

`training_index.json` is updated after every completed run, so a later failure does not
erase the record of earlier completed runs.

## Resume behavior

`--resume` skips only runs that have both a checkpoint and training summary and whose
model/seed/manifest hashes satisfy the frozen plan.

A partially written run directory is not guessed or silently overwritten. The runner
fails and asks for inspection/removal of only that incomplete generated run directory.

## Threshold pass

After all six training runs complete:

```text
scripts/select_experiment_thresholds.py
```

selects one validation-only threshold vector for every checkpoint using the Phase-10
protocol.

For each run it writes:

```text
thresholds.json
threshold_selection.json
```

and the matrix-level:

```text
threshold_index.json
experiment_freeze.json
```

The freeze artifact records the hashes of all six checkpoints and all six threshold
artifacts. Once this artifact exists, the pre-test model/threshold matrix is considered
frozen.

## Validation-only aggregation

`scripts/summarize_validation_experiments.py` reports mean, sample standard deviation,
minimum, and maximum across the three seeds for each model for:

```text
best validation mAP
best epoch
epochs run
training elapsed seconds
validation F1 micro after validation threshold selection
validation F1 macro after validation threshold selection
per-class selected thresholds
```

These are development/validation statistics, not final held-out performance.

## Held-out boundary

Phase-11 training, threshold selection, and validation aggregation contain no call to the
held-out evaluator and no reference to known-test or OOD-test manifests.

The final held-out pass must use the already frozen Phase-10 evaluation workflow only
after `experiment_freeze.json` exists.

## Verification before scientific training

Run:

```powershell
python -m pytest tests/test_experiment_protocol.py -v
python -m pytest
git check-ignore -v artifacts/experiments_phase11
```

Do not start the six training runs until these checks pass and the Phase-11 protocol code
has been committed to Git.
