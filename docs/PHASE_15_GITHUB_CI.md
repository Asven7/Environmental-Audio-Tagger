# Phase 15 — GitHub and Continuous Integration

## Status

**USER VERIFIED — COMMITTED — PUSHED — CI GREEN**

## Purpose

Phase 15 adds a minimal GitHub Actions continuous-integration gate for the repository.

The workflow is an engineering check. It does not download UrbanSound8K, use frozen research artifacts, retrain a model, or rerun the held-out scientific evaluation.

## Workflow

Path:

```text
.github/workflows/ci.yml
```

Triggers:

```text
push to main
pull requests
manual workflow_dispatch
```

Job baseline:

```text
ubuntu-latest
Python 3.11
CPU PyTorch
read-only repository contents permission
```

## Dependency strategy

CPU PyTorch is installed first:

```text
torch>=2.6,<2.11
torchaudio>=2.6,<2.11
```

Then the hosted runner installs:

```text
.[dev,ui]
```

The `live` extra is deliberately not required by hosted CI because a hosted runner has no meaningful physical microphone acceptance path.

Physical `sounddevice` capture is covered by Phase 13 user-local verification.

## CI checks

```text
python -m pip check
python -m compileall -q src scripts tests
python scripts/run_ui.py --help
python scripts/live_microphone.py --help
python -m pytest
```

Dataset-dependent scientific experiments and frozen deployment acceptance are intentionally not hosted CI jobs.

## Security boundary

Workflow permission:

```yaml
permissions:
  contents: read
```

Not used:

```text
pull_request_target
repository secrets
write permissions
deployment credentials
```

## Why Phase-14 QA is not run on GitHub

`scripts/run_phase14_qa.py` requires local files under:

```text
artifacts/experiments_phase11
data/UrbanSound8K
```

Those are intentionally not tracked.

Therefore:

```text
GitHub CI
  -> repository-contained package/code/test checks

Local Phase-14 QA
  -> frozen-artifact integrity + real dataset/demo acceptance
```

## Verified Phase-15 acceptance

Local verification completed:

```text
CI contract tests: PASS
full pytest: PASS
Phase-14 QA: 7/7 PASS
```

Repository milestone:

```text
b759d5a ci: add GitHub Actions test workflow
```

The repository was pushed to GitHub and the corresponding Actions run completed successfully.

Subsequent Phase-16 pushes also triggered the same CI successfully, confirming that the workflow remained operational after the final documentation work.

## Optional branch protection

Requiring the CI check before merging pull requests remains an optional repository-administration choice. It is not required for the scientific or engineering claims of this project.

## Scientific boundary

Phase 15 did not:

```text
download/modify dataset
train/fine-tune
retune thresholds
modify frozen checkpoints
recompute held-out metrics
upload local experiment artifacts
claim improved model performance
```

## Final checkpoint

```text
CI contract tests          USER VERIFIED
full pytest                USER VERIFIED
Phase-14 QA                USER VERIFIED
workflow committed         YES
repository pushed          YES
GitHub Actions             GREEN
scientific freeze          PRESERVED
```

Phase 15 is complete.
