# Phase 4 Completion Checkpoint

Status: **USER VERIFIED**

## Verified commands

```text
python -m pytest tests/test_urbansound_manifest.py -v -> 9 passed
python -m pytest -> 39 passed
python scripts/inspect_urbansound8k.py --dataset-root data\UrbanSound8K -> audit passed
python scripts/prepare_urbansound8k.py --dataset-root data\UrbanSound8K --output-dir artifacts\manifests -> leakage check passed; mixtures false
```

## Frozen research protocol

- target classes: 8
- held-out OOD classes: 2
- train folds: 1–7
- validation fold: 8
- test folds: 9–10
- occurrence source-group unit: `fsID:classID:occurrenceID`
- cross-split broader-recording policy: exclude from all research manifests
- excluded `fsID` values: `106905`, `180937`
- excluded clips: 102
- Phase 4 manifests: single-source only

## Manifest counts

- train: 5209 known singles; 0 OOD singles
- validation: 673 known singles; 130 OOD singles
- test: 1374 known singles; 263 OOD singles

Phase 5 may now implement deterministic, controlled two-source mixtures without changing this frozen source split.
