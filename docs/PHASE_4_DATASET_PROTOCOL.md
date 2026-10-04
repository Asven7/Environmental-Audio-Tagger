# Phase 4 — UrbanSound8K Dataset Protocol

## Goal

Freeze and verify the dataset protocol before any real model training.

## Frozen research protocol

### Classes

Known/target classes (8):

1. `air_conditioner`
2. `children_playing`
3. `dog_bark`
4. `drilling`
5. `engine_idling`
6. `jackhammer`
7. `siren`
8. `car_horn`

Held-out OOD classes (2):

1. `gun_shot`
2. `street_music`

The choice is now frozen for the main experiment. The known set retains broad acoustic diversity. `car_horn` is the smallest known class (429 clips in the raw metadata) and is intentionally retained rather than removed to preserve a realistic class-imbalance challenge. The two held-out classes provide both a relatively sparse transient class (`gun_shot`, 374 clips) and a well-represented sustained/complex class (`street_music`, 1000 clips) for limited rejection evaluation.

### Official-fold partition

- train: folds 1–7
- validation: fold 8
- test: folds 9–10

These fold identities are never reassigned.

### Source independence

Two levels are tracked:

1. Hard occurrence isolation: all slices of `(fsID, classID, occurrenceID)` must remain in one official fold.
2. Stricter research-manifest isolation: if one broader `fsID` appears in more than one configured train/validation/test split, **all clips from that fsID are excluded from every research manifest**.

The second rule avoids train/evaluation correlation from the same original Freesound recording without moving any clip to a different official fold.

The real metadata audit found two such recordings: `106905` and `180937`. The manifest builder derives this exclusion deterministically from the configured split protocol rather than hard-coding the IDs.

## Real-data audit received

- metadata rows: 8732
- folds present: 1–10
- annotated-occurrence isolation: PASSED
- broader fsIDs spanning multiple official folds: 5
- broader fsIDs spanning configured train/val/test splits: 2
- configured classes: all 10 official classes accounted for
- raw split totals: train 6273, validation 806, test 1653
- cross-split `fsID` exclusions: 2 recordings / 102 clips
- post-exclusion split totals: train 6190, validation 803, test 1637
- real single-source manifests: train 5209 known / 0 OOD; validation 673 known / 130 OOD; test 1374 known / 263 OOD
- mixtures in Phase 4 manifests: 0

The audit is metadata-only; no model training or mixture generation is performed in Phase 4.

## Acceptance criteria

| ID | Requirement | Verification | Status |
|---|---|---|---|
| D-01 | Official metadata file is readable | real audit | USER VERIFIED |
| D-02 | Metadata contains official schema/class mapping | automated tests + real audit | USER VERIFIED |
| D-03 | Only official folds 1..10 appear | real audit | USER VERIFIED |
| D-04 | One annotated occurrence `(fsID, classID, occurrenceID)` never spans multiple folds | occurrence-group check | USER VERIFIED |
| D-05 | Configured target/held-out classes exist | real audit | USER VERIFIED |
| D-06 | Class/fold distribution reviewed and classes frozen | real audit + documented decision | USER VERIFIED |
| D-07 | Held-out classes are absent from classifier training manifest | automated test + real manifest summary (`ood_singles=0`) | USER VERIFIED |
| D-08 | Real dataset remains outside Git | repository dataset policy; final Git cleanliness checked at milestone commit | USER VERIFIED |
| D-09 | Single-source manifests can be generated without mixtures | real prepare run; `include_mixtures=false` | USER VERIFIED |
| D-10 | No annotated occurrence crosses train/val/test manifests | real source-group leakage assertion | USER VERIFIED |
| D-11 | Broader fsIDs crossing configured splits are excluded from all research manifests | real audit + manifest summary (`106905`, `180937`; 102 clips) | USER VERIFIED |

## Benchmark comparability note

UrbanSound8K's standard classification guidance uses predefined folds with 10-fold cross-validation. This project instead needs a validation set for threshold/hyperparameter tuning, a frozen test set, synthetic multi-label mixtures, and held-out OOD classes. Results are therefore not presented as directly equivalent to the standard single-label 10-fold UrbanSound8K benchmark.

## Phase boundary

Phase 4 ends after leakage-safe **single-source** manifests are generated and verified. Synthetic two-source mixtures remain deferred to Phase 5.
