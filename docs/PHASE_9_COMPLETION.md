# Phase 9 Completion Record — Training / Reproducibility

## Final status

**USER VERIFIED**

Phase 9 is technically complete and ready for its Git milestone.

## Verification record

```text
Dedicated training-protocol tests: 11 passed
Full project regression:           95 passed

Tiny training train samples:       24
Tiny training validation samples:  12
Experiment seed:                   13

CPU same-seed reproducibility:     PASSED
CPU best epoch:                    2
CPU best validation mAP:           0.32705965909090906
CPU epochs run:                    2
CPU stop reason:                   max_epochs

Threshold isolation:               PASSED
Thresholds tuned:                  false
Frozen test manifests used:        false
OOD test manifests used:           false

CUDA tiny CRNN training:           PASSED
CUDA device:                       NVIDIA GeForce RTX 3050 Ti Laptop GPU
CUDA best epoch:                   1

Temporary smoke artifacts:         deleted
```

## Protocol decisions frozen in this phase

- Research training uses train + validation only.
- Test and OOD-test evaluation are not part of the Phase-9 experiment runner.
- Threshold tuning is disabled for research training runs.
- Best checkpoint selection uses validation mAP only.
- Early stopping uses validation behavior only.
- `pos_weight` uses training labels only.
- Training augmentation is train-only.
- Train DataLoader shuffling has a dedicated experiment-seeded RNG.
- Global RNGs are seeded.
- Deterministic PyTorch/cuDNN settings are requested.
- Experiment seed and data seed remain separate concepts.
- Training provenance includes manifest fingerprints and environment metadata.

## Reproducibility note

Passing the same-seed CPU reproducibility smoke test verifies exact equality for the
tested tiny run, including training history and checkpoint tensors.

This does not imply that every CUDA execution on every hardware/software stack will
necessarily be bit-for-bit identical. The project therefore still uses the planned
multi-seed experiment design `[13, 23, 37]` for scientific reporting.

## Next phase boundary

Phase 10 will define and freeze threshold selection and test/OOD evaluation discipline.
Do not run the full scientific experiment campaign until that protocol is closed.
