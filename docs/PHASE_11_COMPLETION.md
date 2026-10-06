# Phase 11 Completion Record

## Final status
**USER VERIFIED**

## Git / protocol record
- `c9da33d feat: lock multi-seed experiment protocol`
- `bf230ac feat: add frozen multi-seed result aggregation`

## Official training
CNN:
- seed 13: best epoch 27, validation mAP 0.623933
- seed 23: best epoch 23, validation mAP 0.606294
- seed 37: best epoch 30, validation mAP 0.615183

CRNN:
- seed 13: best epoch 27, validation mAP 0.662274
- seed 23: best epoch 29, validation mAP 0.675714
- seed 37: best epoch 30, validation mAP 0.670142

Validation-only mean ± std:
- CNN: mAP 0.615137 ± 0.008819, F1 micro 0.572758 ± 0.005473, F1 macro 0.598761 ± 0.008029
- CRNN: mAP 0.669377 ± 0.006753, F1 micro 0.592500 ± 0.013290, F1 macro 0.629206 ± 0.010054

## Pre-test freeze
- freeze version: `cnn_crnn_validation_freeze_v1`
- freeze SHA-256: `0305a2aa821ed16a5f485cde1951d3720c47fa65dcc9d0ec9befd2e07f63be68`
- frozen model/threshold pairs: 6

## Frozen held-out results
Known samples: 2574
OOD samples: 263
Retuning after held-out evaluation: false

CNN:
- mAP 0.618628 ± 0.008281
- F1 micro 0.549715 ± 0.004061
- F1 macro 0.564576 ± 0.003953
- Hamming loss 0.194428 ± 0.000791
- OOD rejection 0.012674 ± 0.007915

CRNN:
- mAP 0.727378 ± 0.007039
- F1 micro 0.592859 ± 0.010359
- F1 macro 0.617833 ± 0.013461
- Hamming loss 0.173433 ± 0.008472
- OOD rejection 0.046895 ± 0.025317

## Final model-family decision
CRNN is the model family carried forward into runtime/demo work because it is stronger on the frozen primary known-test metrics. This is a deployment choice only; held-out results are already final and cannot be used to retune the CRNN.

## Important limitation
The threshold-based OOD rule has very low rejection of held-out classes and must be described as a confidence heuristic rather than robust open-set recognition.

## Verification
- Phase-11 protocol tests: 8 passed
- repository regression after protocol work: 117 passed
- frozen aggregation tests: 5 passed
- repository regression after aggregation work: 122 passed
- final Git status before completion docs: clean

## Final generated result artifacts
Stored under Git-ignored `artifacts/experiments_phase11/`:
- `frozen_test_summary.json`
- `frozen_test_headline.csv`
- `frozen_test_per_class.csv`
- `frozen_test_groups.csv`

## Next phase
Phase 12 may implement inference/runtime benchmarking and deployment plumbing using the frozen CRNN artifacts. It must not alter the Phase-11 scientific results.
