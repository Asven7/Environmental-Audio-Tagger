# Phase 4 Real UrbanSound8K Audit and Manifest Verification

## Local verification on the target Windows machine

- `tests/test_urbansound_manifest.py`: **9 passed**
- full regression suite: **39 passed**
- UrbanSound8K metadata rows: **8732**
- official folds present: **1–10**
- annotated-occurrence isolation `(fsID, classID, occurrenceID)`: **PASSED**
- broader `fsID` values spanning multiple official folds: **5**
- broader `fsID` values spanning configured train/validation/test splits: **2**

## Frozen class protocol

Known target classes:

- `air_conditioner`
- `children_playing`
- `dog_bark`
- `drilling`
- `engine_idling`
- `jackhammer`
- `siren`
- `car_horn`

Held-out OOD classes:

- `gun_shot`
- `street_music`

The held-out classes are not used to train the eight-output classifier.

## Frozen fold protocol

- train: folds **1–7**
- validation: fold **8**
- test: folds **9–10**

Official fold assignments are not rewritten.

## Cross-split broader-recording exclusion

The real audit detected two broader Freesound recordings spanning configured splits:

- `fsID=106905`: 7 clips across train/validation; classes `engine_idling`, `siren`
- `fsID=180937`: 95 clips across train/test; classes `drilling`, `jackhammer`

Policy: exclude all clips from any cross-split `fsID` from every research manifest. The implementation detects these dynamically; the IDs are not hard-coded.

Total excluded clips: **102**.

Raw split totals:

- train: 6273 clips / 924 broader source recordings
- validation: 806 clips / 126 broader source recordings
- test: 1653 clips / 249 broader source recordings

Post-exclusion split totals:

- train: **6190 clips / 922 source recordings**
- validation: **803 clips / 125 source recordings**
- test: **1637 clips / 248 source recordings**

## Real Phase 4 manifests

The single-source manifest preparation completed successfully with:

```text
Source-group leakage check: PASSED
Mixtures included: False
```

Manifest counts:

| Split | Known singles | Known mixtures | OOD singles |
|---|---:|---:|---:|
| train | 5209 | 0 | 0 |
| validation | 673 | 0 | 130 |
| test | 1374 | 0 | 263 |

The generated summary records:

- `source_group_unit = fsID:classID:occurrenceID`
- `cross_split_recording_policy = exclude_from_all_research_manifests`
- `excluded_cross_split_fsids = [106905, 180937]`
- `excluded_cross_split_clips = 102`
- `include_mixtures = false`

## Phase result

**Phase 4: USER VERIFIED.**

The leakage-safe single-source dataset protocol is frozen. Controlled two-source mixtures remain deferred to Phase 5.
