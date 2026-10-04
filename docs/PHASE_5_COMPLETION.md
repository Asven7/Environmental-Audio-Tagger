# Phase 5 Completion — Controlled Multi-Label Mixtures

## Status

**USER VERIFIED**

## Scope completed

Phase 5 freezes and verifies the controlled two-label mixture protocol. It does not perform feature extraction or model training.

Completed controls:

- same-split source selection only;
- exactly two different known classes per mixture;
- held-out OOD classes excluded from known-class mixtures;
- relative dB grid: `-6`, `0`, `+6`;
- temporal-overlap grid: `0.25`, `0.50`, `1.00`;
- balanced unordered class-pair schedule;
- balanced dB/overlap condition schedule;
- balanced source-A/source-B orientation;
- exact research occurrence-pair reuse prohibited;
- same broader UrbanSound8K `fsID` pairing avoided when identifiable;
- deterministic generation from the frozen data seed;
- mixed audio synthesized on demand rather than stored as duplicate WAV files.

## Verified test baseline

```text
Phase 5 protocol tests: 9 passed
Demo dataset regression: 1 passed
Full regression suite: 48 passed
```

## Verified real-manifest baseline

```text
train mixtures:      4000
validation mixtures:  600
test mixtures:       1200
```

Balance ranges:

| Split | Class-pair count | dB/overlap condition count | Unique mixture pairs |
|---|---:|---:|---:|
| train | 142–143 | 444–445 | 4000 |
| validation | 21–22 | 66–67 | 600 |
| test | 42–43 | 133–134 | 1200 |

The metadata audit reported:

```text
Source-group leakage across splits: PASSED
Controlled mixture protocol validation: PASSED
```

## Verified real-audio smoke test

Configuration:

```text
split=train
sample_rate=22050
target_samples=44100
crop_strategy=energy
count=3
```

Observed mixtures:

```text
mix-train-000000: relative_db=+6, overlap=1.00, peak=0.5528
mix-train-000001: relative_db=+6, overlap=0.50, peak=0.5728
mix-train-000002: relative_db=+6, overlap=0.25, peak=0.5067
```

Result:

```text
Real mixture audio smoke test: PASSED
No files were written and no model training was performed.
```

## Demo compatibility note

The strict no-repeat occurrence-pair rule remains enabled for UrbanSound8K research manifests. Tiny synthetic demo manifests may reuse path-fallback pairs only when their finite pair pool is exhausted. This exception is restricted to engineering smoke tests and is not used for research results.

## Phase boundary

Phase 5 ends here. Phase 6 begins Log-Mel feature extraction and validation. No Phase 6 implementation belongs in this milestone commit.
