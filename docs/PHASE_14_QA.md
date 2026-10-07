# Phase 14 — Quality Assurance

## Status

**USER VERIFIED — COMMITTED**

## Purpose

Phase 14 adds a repository-level engineering QA gate after the complete Phase-13 demo.

It does not improve model accuracy or reopen the frozen scientific protocol.

## Verified local results

Dedicated QA tests:

```text
5 passed
```

Phase-14 full regression checkpoint:

```text
160 passed
```

QA command:

```powershell
python scripts\run_phase14_qa.py `
  --project-root . `
  --experiment-root artifacts\experiments_phase11 `
  --audio-file data\UrbanSound8K\audio\fold8\103076-3-0-0.wav `
  --device cpu `
  --output artifacts\phase14_qa_report.json
```

Verified:

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
quality_checks=7 passed
```

## Frozen deployment integrity

```text
model = CRNN
seed = 23
best validation mAP = 0.6757137110147023
```

Hashes remained:

```text
checkpoint:
80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8

threshold:
788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2
```

No frozen artifact was modified.

## Repository hygiene

The Phase-14 checkpoint reported:

```text
tracked files = 216
prohibited tracked paths = 0
```

That count is historical to Phase 14; later documentation/CI files legitimately increased the tracked-file count.

The hygiene rule protects against accidentally tracking:

```text
.venv/
.pytest_tmp/
data/UrbanSound8K/
artifacts/experiments_phase11/
__pycache__/
```

## Dependency / Python / CLI checks

Verified:

```text
python -m pip check
python -m compileall -q src scripts tests
python scripts/run_ui.py --help
python scripts/live_microphone.py --help
```

## Phase-13 regression inside QA

Phase-13C acceptance was rerun and passed.

Its observed processing time is an engineering smoke observation, not a replacement for the frozen Phase-12 runtime benchmark.

## Generated report

```text
artifacts\phase14_qa_report.json
```

Recorded:

```text
status = PASS
retrained = false
thresholds_retuned = false
heldout_metrics_recomputed = false
frozen_artifacts_modified = false
```

## Scientific boundary

Phase 14 does not:

```text
train/fine-tune
change architecture
change preprocessing
retune thresholds
select deployment using held-out test
recompute frozen held-out metrics
claim calibration
claim robust open-set recognition
```

## Final checkpoint

```text
Dedicated QA tests            USER VERIFIED
Full regression               USER VERIFIED
git diff --check              USER VERIFIED
pip check                     USER VERIFIED
compileall                    USER VERIFIED
UI CLI surface                USER VERIFIED
Microphone CLI surface        USER VERIFIED
Phase-13C regression          USER VERIFIED
Frozen deployment integrity   USER VERIFIED
Git tracked-file hygiene      USER VERIFIED
QA report                     USER VERIFIED
Scientific freeze boundary    VERIFIED
commit                        COMPLETED
```
