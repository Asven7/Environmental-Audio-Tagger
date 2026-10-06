# Phase 10 Completion Record — Threshold / Frozen Evaluation

## Final status

**USER VERIFIED**

Phase 10 is technically complete and ready for its Git milestone.

## Verification record

```text
Evaluation protocol tests:           15 passed
Full project regression:             109 passed

Tiny train samples:                  12
Tiny validation samples:             12
Tiny OOD-validation samples:         8

Validation threshold selection:      PASSED
Threshold classes:                   8
Threshold provenance binding:        PASSED
Strict validation-only evaluation:   PASSED
OOD validation rejection plumbing:   PASSED
Checkpoint mismatch rejection:       PASSED

Best smoke-training epoch:           1
Threshold-selection val samples:     12

Known test manifest read:            false
OOD test manifest read:              false
Temporary smoke artifacts:           deleted
```

## Protocol decisions frozen in this phase

- Thresholds are selected per class using known-validation F1 only.
- Validation classes must have both positive and negative examples for threshold tuning.
- Threshold tie-breaking is deterministic.
- Threshold artifacts are bound to exact checkpoint and validation-manifest SHA-256.
- Strict evaluation checks threshold provenance before scoring.
- Test evaluation artifacts are protected from accidental overwrite.
- Frozen evaluation writes a lock artifact after successful final evaluation.
- OOD rejection means no known class exceeded its validation-selected threshold.
- Known false rejection and OOD rejection are reported together.
- Controlled-mixture analysis includes joint relative-dB × overlap groups.
- Real known-test and OOD-test data remained untouched during Phase 10 verification.

## Reproducibility / research rule

After the first real frozen-test result is observed, the threshold protocol, class set,
data split, architecture, and evaluation definitions must not be changed based on those
results.

Any later protocol change would require treating the old test result as development
feedback rather than as a final untouched test result.

## Next phase boundary

Phase 11 may run the predeclared multi-seed training campaign under the already frozen
training and evaluation protocols.

Do not run the final frozen test evaluation until all predeclared model/seed runs and
their validation-selected threshold artifacts are finalized.
