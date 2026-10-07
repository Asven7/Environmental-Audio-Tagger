# Phase 14 — Quality Assurance

## Status

**USER VERIFIED — READY TO COMMIT**

## Purpose

Phase 14 adds a repository-level engineering quality gate after the complete Phase-13 demo.

This phase does not improve model accuracy and does not reopen the frozen scientific
protocol. Its purpose is to detect accidental breakage before GitHub/CI, clean-install,
documentation, and final defense packaging.

## Verified local results

The dedicated Phase-14 QA tests were run locally:

```powershell
python -m pytest tests/test_phase14_qa.py -v
```

Result:

```text
5 passed
```

The complete project regression suite was then run:

```powershell
python -m pytest
```

Result:

```text
160 passed
```

The repository QA gate was run with:

```powershell
python scripts\run_phase14_qa.py `
  --project-root . `
  --experiment-root artifacts\experiments_phase11 `
  --audio-file data\UrbanSound8K\audio\fold8\103076-3-0-0.wav `
  --device cpu `
  --output artifacts\phase14_qa_report.json
```

Verified high-level result:

```text
[PASS] git_diff_check
[PASS] pip_check
[PASS] compileall
[PASS] ui_help
[PASS] microphone_help
[PASS] phase13_acceptance
[PASS] pytest_full

status=PASS
frozen_deployment=crnn_seed23 validation_mAP=0.675714
git_hygiene=216 tracked files, 0 prohibited tracked paths
quality_checks=7 passed
```

## Frozen deployment integrity

The QA gate verified the already-selected Phase-12 deployment:

```text
model = CRNN
seed = 23
best validation mAP = 0.6757137110147023
```

Frozen artifact hashes remained:

```text
checkpoint SHA-256 =
80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8

threshold artifact SHA-256 =
788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2
```

No frozen artifact was modified.

## Repository hygiene

The QA report found:

```text
tracked files = 216
prohibited tracked paths = 0
```

The tracked-file hygiene check covers the project-local/generated paths:

```text
.venv/
.pytest_tmp/
data/UrbanSound8K/
artifacts/experiments_phase11/
__pycache__/
```

This is a focused repository hygiene check and is not presented as a complete secret
scanner or security audit.

## Dependency integrity

`python -m pip check` passed:

```text
No broken requirements found.
```

This confirms that the currently installed Python environment has no dependency conflicts
reported by pip.

## Python and CLI checks

Compilation succeeded for:

```text
src/
scripts/
tests/
```

Both user-facing demo CLIs also exposed their argument surfaces successfully:

```text
scripts/run_ui.py --help
scripts/live_microphone.py --help
```

The microphone CLI still exposes device selection/listing and callback block-duration
controls while retaining the frozen inference hop.

## Phase-13 regression inside QA

The Phase-13C acceptance was rerun automatically as part of the QA gate and passed.

Observed engineering smoke result:

```text
deployment = crnn_seed23
protocol = 22050 Hz / 2.0 s window / 1.0 s hop
file mode = 3 windows
maximum processing time observed = 10.814 ms
below 1000 ms hop = true
Gradio = 6.29.1
default input = Microphone (C-Media(R) Audio)
```

This timing is an engineering smoke observation and does not replace the frozen Phase-12
runtime benchmark.

## Full regression

The final Phase-14 QA run also invoked the full pytest suite internally.

Result:

```text
160 passed
```

Therefore both the explicit user-run regression and the QA-gate-internal regression passed.

## Generated report

The local engineering report is:

```text
artifacts\phase14_qa_report.json
```

The report recorded:

```text
status = PASS
retrained = false
thresholds_retuned = false
heldout_metrics_recomputed = false
frozen_artifacts_modified = false
```

This report is a local engineering artifact. It is not required to be committed to Git.

## Scientific boundary

Phase 14 does not:

```text
train or fine-tune a model
change model architecture
change preprocessing
retune thresholds
select a new deployment using held-out test results
recompute frozen held-out accuracy/F1/mAP
claim calibrated probabilities
claim robust open-set recognition
```

Passing Phase 14 means the repository and demo pipeline passed the defined engineering
quality gates. It does not strengthen the frozen scientific accuracy claims.

## Phase checkpoint

```text
Dedicated Phase-14 tests        USER VERIFIED
Full pytest regression          USER VERIFIED
git diff --check                USER VERIFIED
pip check                       USER VERIFIED
compileall                      USER VERIFIED
UI CLI surface                  USER VERIFIED
Microphone CLI surface          USER VERIFIED
Phase-13C regression            USER VERIFIED
Frozen deployment integrity     USER VERIFIED
Git tracked-file hygiene        USER VERIFIED
QA report                       USER VERIFIED
Scientific freeze boundary      VERIFIED
```

**Phase 14 is complete and ready for Git commit.**
