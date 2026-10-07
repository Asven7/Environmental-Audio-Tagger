# Phase 16 Completion — Documentation and Clean Installation

## Status

**IMPLEMENTED — WAITING FOR USER LOCAL VERIFICATION OF 16B**

Phase 16A is already **USER VERIFIED** through a fresh clone and fresh virtual environment.

## Verified Phase 16A evidence

Fresh-clone acceptance:

```text
Python 3.11.9
pip check: PASS
clean-install verifier: PASS
full pytest: 173 passed
working tree: clean
```

The clean-install verifier required neither UrbanSound8K nor frozen Phase-11 research artifacts and did not recompute held-out scientific metrics.

## Phase 16B documentation consolidation

The final user-facing documentation is consolidated around:

```text
README.md
docs/README.md
docs/CLEAN_INSTALL.md
```

The top-level README now distinguishes:

```text
scientific scope vs full SED
frozen research results vs synthetic demo outputs
validation-only selection vs held-out evaluation
weak held-out rejection vs general open-set recognition
canonical runtime vs initial window collection latency
public-repository installability vs local frozen experiment artifacts
```

It also records the verified clean-install procedure and links the phase-specific evidence.

## Frozen research headline

Three-seed held-out comparison:

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

Selected deployment:

```text
CRNN seed 23
selected using validation mAP only
validation mAP = 0.6757137110
```

Canonical p95 runtime:

```text
CPU:  4.358 ms
CUDA: 1.561 ms
stream hop: 1000 ms
```

## Important limitation retained in final documentation

The selected threshold-based rejection heuristic is weak:

```text
CRNN held-out rejection rate ~4.69%
CRNN held-out false acceptance rate ~95.31%
```

Therefore the project does not claim robust open-set or unknown-sound recognition.

## Completion checks for 16B

Run:

```powershell
python -m pytest tests/test_final_documentation_contract.py -v
python -m pytest
git diff --check
git diff --stat
```

Phase 16 can be marked **USER VERIFIED + COMMITTED** after these checks pass and the final documentation commit is recorded.

## Scientific boundary

Phase 16B is documentation-only. It must not change:

```text
models
checkpoints
thresholds
preprocessing
split protocol
frozen held-out results
runtime benchmark artifacts
deployment selection
```
