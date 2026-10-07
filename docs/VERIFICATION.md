# Verification Record

This document separates historical engineering smoke checks from the final user-verified scientific and repository state.

## Final repository regression

Phase 16B local verification:

```text
tests/test_final_documentation_contract.py: 8 passed
full repository suite: 181 passed
git diff --check: PASS
working tree after commit: clean
```

The exact future test count may increase; the acceptance rule is a fully passing suite.

## Fresh-clone installation verification

A separate clone and fresh `.venv` were used with Python 3.11.9.

Verified:

```text
python -m pip check
  -> No broken requirements found.

python scripts\verify_clean_install.py ...
  -> status=PASS

python -m pytest
  -> 173 passed

git status
  -> clean
```

The clean-install verifier confirmed:

```text
UrbanSound8K was NOT required.
Frozen experiment artifacts were NOT required.
Held-out scientific metrics were NOT recomputed.
```

## GitHub CI

The repository contains `.github/workflows/ci.yml`.

The Phase-16B commit:

```text
539418a docs: finalize project documentation and clean-install record
```

was pushed to `main`, and the associated GitHub Actions `CI` run completed successfully.

CI checks repository-contained package/test surfaces only; it does not pretend to execute dataset/frozen-artifact-dependent scientific acceptance.

## Frozen scientific verification

Official experiment matrix:

```text
CNN  × seeds 13, 23, 37
CRNN × seeds 13, 23, 37
```

Final held-out headline:

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

See [`PHASE_11_FINAL_RESULTS.md`](PHASE_11_FINAL_RESULTS.md).

No threshold, architecture, preprocessing, class, split, or metric definition was changed after observing the frozen held-out result.

## OOD/rejection verification

Frozen CRNN held-out rejection:

```text
OOD rejection rate         0.046895 ± 0.025317
OOD false acceptance rate  0.953105 ± 0.025317
```

Therefore the repository does **not** claim robust unknown/open-set recognition.

## Frozen deployment verification

Deployment:

```text
CRNN seed 23
selected using validation mAP only
validation mAP = 0.6757137110147023
```

Frozen hashes:

```text
checkpoint:
80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8

threshold artifact:
788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2
```

## Runtime verification

Canonical batch-1 p95 total compute:

```text
CPU  = 4.358 ms
CUDA = 1.561 ms
hop  = 1000 ms
```

Both satisfy the no-backlog criterion on the verified laptop.

See [`PHASE_12_RUNTIME_RESULTS.md`](PHASE_12_RUNTIME_RESULTS.md).

## UI / microphone verification

Verified in Phase 13:

```text
sequential file inference
Gradio UI construction and local use
browser microphone streaming
history ordering
Stop preserves results/history
explicit Clear results
sounddevice device listing
physical sounddevice microphone capture
CPU live inference
clean shutdown
```

Live false positives were retained as a documented limitation and did not trigger post-test threshold changes.

## Repository QA verification

Phase 14 verified:

```text
git diff --check
pip check
compileall
UI CLI surface
microphone CLI surface
Phase-13 acceptance
full pytest
frozen deployment integrity
tracked-file hygiene
```

## Historical synthetic demo verification

The tracked synthetic demo artifacts remain useful for software smoke testing without UrbanSound8K. Their metrics are **not** environmental-audio research results.

## Not quantitatively verified

The repository does not contain a separately annotated real-world multi-label field dataset. Therefore quantitative claims about unconstrained live-microphone accuracy are intentionally not made.

## Scientific boundary

The frozen held-out scientific result must not be rerun as ordinary CI, installation verification, documentation cleanup, UI work, or defense preparation.

Future model/preprocessing development requires a newly declared protocol before another held-out evaluation.
