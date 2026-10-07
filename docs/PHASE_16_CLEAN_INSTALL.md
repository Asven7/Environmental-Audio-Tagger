# Phase 16 — Documentation and Clean-Install Reproducibility

## Status

**16A USER VERIFIED — FRESH CLONE / FRESH VENV ACCEPTANCE PASSED**

Phase 16B was also subsequently user verified, committed, pushed, and CI verified. Phase 16C is a documentation/metadata consistency cleanup and does not change the clean-install result.

## Scope

Phase 16 was split into:

```text
16A — fresh-clone / fresh-venv installation verification
16B — final README + user-facing documentation consolidation
16C — final repository documentation/metadata consistency cleanup
```

Phase 16A was verified before final documentation consolidation so the installation instructions are based on a real clean-environment run.

## Phase 16A implementation

Added:

```text
scripts/verify_clean_install.py
tests/test_clean_install_contract.py
docs/CLEAN_INSTALL.md
docs/PHASE_16_CLEAN_INSTALL.md
```

The verifier checks:

```text
installed distribution metadata
core + UI + live + test dependency presence
pip dependency consistency
installed console entry points
Gradio command surface
sounddevice command surface
CNN synthetic CPU forward
CRNN synthetic CPU forward
finite model logits
```

It deliberately does not require:

```text
UrbanSound8K
Phase-11 frozen experiment artifacts
microphone capture
a GPU
held-out scientific evaluation
```

## Development-checkout verification

Before the fresh clone:

```text
clean-install contract tests: 7 passed
full repository suite: 173 passed
verify_clean_install.py: PASS
```

Verifier environment:

```text
Python 3.11.9
package 0.1.0
Gradio 6.29.1
sounddevice 0.5.6
console scripts: esaudio-train, esaudio-evaluate, esaudio-infer
CNN synthetic CPU forward: PASS
CRNN synthetic CPU forward: PASS
```

## Fresh-clone acceptance

The project was cloned to a separate directory and installed into a new `.venv`.

Verified path:

```text
Windows
Python 3.11.9
CPU PyTorch
non-editable python -m pip install ".[all]"
```

Final acceptance:

```text
python -m pip check
  -> No broken requirements found.

python scripts\verify_clean_install.py ...
  -> status=PASS

python -m pytest
  -> 173 passed in 8.28s

git status
  -> branch up to date with origin/main
  -> nothing to commit, working tree clean
```

Explicit verifier boundary:

```text
UrbanSound8K was NOT required.
Frozen experiment artifacts were NOT required.
Held-out scientific metrics were NOT recomputed.
```

## Phase 16B

Final documentation consolidation updated:

```text
README.md
docs/README.md
docs/PHASE_16_CLEAN_INSTALL.md
docs/PHASE_16_COMPLETION.md
tests/test_final_documentation_contract.py
```

User-local verification:

```text
documentation contract: 8 passed
full pytest: 181 passed
git diff --check: PASS
```

Milestone:

```text
539418a docs: finalize project documentation and clean-install record
```

The commit was pushed to `main` and GitHub Actions completed successfully.

## Phase 16C

A repository-wide audit found stale status/history inconsistencies in older user-facing documents. Phase 16C updates only documentation, metadata convenience files, and documentation contract assertions.

It does **not** change:

```text
scientific config
model code
checkpoints
thresholds
frozen held-out results
runtime results
deployment selection
```

## Scientific boundary

Phase 16 did not:

```text
retrain a model
retune thresholds
modify frozen checkpoints
change preprocessing
reselect deployment using held-out results
recompute held-out accuracy/F1/mAP
```

Clean-install and documentation verification are engineering reproducibility work, not new scientific experiments.
