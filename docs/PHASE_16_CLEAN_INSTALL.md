# Phase 16 — Documentation and Clean-Install Reproducibility

## Status

**16A IMPLEMENTED — WAITING FOR USER FRESH-ENVIRONMENT VERIFICATION**

## Scope

Phase 16 has two controlled parts:

```text
16A — fresh-clone / fresh-venv installation verification
16B — final user-facing documentation consolidation
```

Phase 16A is implemented first because final documentation should record commands that
have actually been verified in a clean environment rather than assuming they work.

## Added in 16A

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
network access after installation
held-out scientific evaluation
```

## Why a new clone is required

Running `pip install` inside the existing development environment would not prove that the
repository is self-contained.

The Phase-16 acceptance therefore uses:

```text
new clone
new .venv
Python 3.11
CPU PyTorch
non-editable project install
```

The existing development checkout remains untouched except for the Phase-16 source/docs
changes.

## Local source verification before fresh-clone acceptance

In the current development checkout:

```powershell
python -m pytest tests/test_clean_install_contract.py -v
python -m pytest
```

After those pass, commit/push Phase-16A so that the clean clone contains the new verifier
and documentation.

The clean-clone procedure is then executed exactly as documented in:

```text
docs/CLEAN_INSTALL.md
```

## Completion rule

Phase 16 is not complete after the current-environment tests alone.

16A becomes **USER VERIFIED** only when the user reports a fresh clone and fresh virtual
environment with:

```text
pip check = PASS
verify_clean_install.py = PASS
full pytest = PASS
working tree = clean
```

After that evidence is recorded, 16B will update/consolidate the final README and
user-facing documentation based on verified commands.

No Phase 17 work begins before Phase 16 is fully verified and committed.
