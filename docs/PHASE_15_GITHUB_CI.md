# Phase 15 — GitHub and Continuous Integration

## Status

**IMPLEMENTED — WAITING FOR USER LOCAL AND GITHUB VERIFICATION**

## Purpose

Phase 15 adds a minimal GitHub Actions continuous-integration gate for the repository.

The CI workflow is intentionally an engineering check. It does not download UrbanSound8K,
does not use frozen experiment artifacts, does not retrain a model, and does not rerun the
held-out scientific evaluation.

## Workflow

The workflow is stored at:

```text
.github/workflows/ci.yml
```

It runs on:

```text
push to main
pull requests
manual workflow_dispatch
```

The job uses:

```text
ubuntu-latest
Python 3.11
CPU PyTorch
read-only repository contents permission
```

Python 3.11 is used as the CI baseline because it matches the locally verified project
environment. Broader clean-install/package compatibility belongs to Phase 16.

## Dependency strategy

The workflow installs CPU-only PyTorch first:

```text
torch>=2.6,<2.11
torchaudio>=2.6,<2.11
```

from the official PyTorch CPU wheel index.

It then installs:

```text
.[dev,ui]
```

The `live` extra is deliberately not required by CI. A hosted CI runner has no meaningful
physical microphone for the project's sounddevice acceptance path. The real microphone
path remains covered by the Phase-13 local acceptance.

## CI checks

The workflow runs:

```text
python -m pip check
python -m compileall -q src scripts tests
python scripts/run_ui.py --help
python scripts/live_microphone.py --help
python -m pytest
```

The full pytest suite is expected to remain repository-contained and synthetic/fixture based.
Dataset-dependent scientific experiments and frozen deployment acceptance are not CI jobs.

## Security boundary

The workflow uses:

```yaml
permissions:
  contents: read
```

It does not use:

```text
pull_request_target
repository secrets
write permissions
deployment credentials
```

This keeps the pull-request CI surface minimal.

## Why Phase-14 QA is not run in GitHub Actions

`scripts/run_phase14_qa.py` verifies the frozen CRNN deployment and Phase-13 acceptance
using local files under:

```text
artifacts/experiments_phase11
data/UrbanSound8K
```

Those files are intentionally not repository content.

Therefore GitHub CI runs the full repository test suite but does not pretend that the
dataset/artifact-dependent local QA gate can run on a clean hosted runner.

This split is intentional:

```text
GitHub CI
  -> repository-contained code/tests/package checks

Local Phase-14 QA
  -> frozen artifact integrity + real dataset/demo acceptance
```

## Local verification before commit

Run:

```powershell
python -m pytest tests/test_ci_contract.py -v
python -m pytest
```

Then re-run the existing local QA gate:

```powershell
python scripts\run_phase14_qa.py `
  --project-root . `
  --experiment-root artifacts\experiments_phase11 `
  --audio-file data\UrbanSound8K\audio\fold8\103076-3-0-0.wav `
  --device cpu `
  --output artifacts\phase14_qa_report.json
```

No scientific result should change.

## GitHub verification after commit

Before pushing, inspect:

```powershell
git status
git remote -v
git log --oneline --decorate -5
```

Pushing is an external action and should only be performed intentionally after the local
Phase-15 checks pass.

After the branch is pushed to GitHub, verify in the repository's Actions page that:

```text
workflow name = CI
job = Python 3.11 / CPU
result = green / passed
```

A pull request should also trigger the same CI job.

## Optional branch protection

After CI is verified on GitHub, `main` can optionally be protected so that the CI check is
required before merging pull requests.

This is a repository-administration choice and is not automatically changed by Phase 15.

## Scientific boundary

Phase 15 does not:

```text
download or modify the dataset
train or fine-tune models
retune thresholds
modify frozen checkpoints
recompute held-out accuracy/F1/mAP
upload local experiment artifacts
claim improved model performance
```

## Phase checkpoint

Phase 15 is complete only after:

```text
CI contract tests pass locally
full pytest passes locally
Phase-14 QA still passes locally
Phase-15 files are committed
repository is pushed intentionally
GitHub Actions CI run is green
```

Suggested local commit after verification:

```text
ci: add GitHub Actions test workflow
```
