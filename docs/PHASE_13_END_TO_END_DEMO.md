# Phase 13C — End-to-End Local Demo Verification

## Status

**USER VERIFIED — COMMITTED**

## Goal

Final acceptance of the already-frozen deployment through:

```text
deployment selection
-> checkpoint + thresholds
-> file decode / sequential windows
-> canonical inference
-> Gradio UI
-> browser microphone
-> local sounddevice microphone
```

No new model, threshold, or scientific protocol was introduced.

## Automated acceptance

Dedicated:

```powershell
python -m pytest tests/test_phase13_demo_acceptance.py -v
```

Verified:

```text
4 passed
```

Phase-13 full regression at that checkpoint:

```text
155 passed
```

Automated acceptance command:

```powershell
python scripts\verify_phase13_demo.py `
  --project-root . `
  --experiment-root artifacts\experiments_phase11 `
  --audio-file data\UrbanSound8K\audio\fold8\103076-3-0-0.wav `
  --device cpu `
  --output artifacts\phase13c_acceptance.json
```

Verified high-level output:

```text
status=PASS
deployment=crnn_seed23 validation_mAP=0.675714
protocol=22050Hz 2.0s-window 1.0s-hop
file_mode=3 windows
max_processing_ms=8.869
below_hop=True
ui=Blocks
gradio=6.29.1
Held-out accuracy metrics were NOT recomputed.
Thresholds were NOT changed.
```

## Frozen deployment

```text
model = CRNN
seed = 23
best validation mAP = 0.6757137110147023
sample rate = 22050 Hz
window = 2.0 s
hop = 1.0 s
device = CPU for acceptance
```

Hashes:

```text
checkpoint SHA-256 =
80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8

threshold SHA-256 =
788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2
```

## Sequential file acceptance

The 2.527 s validation example produced:

```text
window 0: 0.0 -> 2.0 s
window 1: 1.0 -> 3.0 s
window 2: 2.0 -> 4.0 s
```

Observed max processing in that acceptance was about 8.869 ms, below the 1000 ms hop.

This smoke timing does not replace the canonical Phase-12 benchmark.

## Manual Gradio acceptance

Verified:

```text
File tab                    PASS
Browser microphone          PASS
Newest-first history        PASS
Oldest-first history        PASS
Stop preserves history      PASS
Clear results               PASS
Clean UI shutdown           PASS
```

## Manual sounddevice acceptance

Verified local input:

```text
Microphone (C-Media(R) Audio)
```

The CLI produced sequential rolling windows and shut down cleanly.

## Scientific boundary

Acceptance explicitly preserved:

```text
retrained = false
thresholds_retuned = false
heldout_test_read_for_development = false
accuracy_metrics_recomputed = false
```

## Live/OOD limitation

Quiet-room/fan audio often produced false positive known classes. This is consistent with the frozen weak OOD result.

Correct project claim:

```text
real-time window-level multi-label tagging / event-presence detection is operational
```

Not:

```text
robust open-set environmental sound recognition
```

Scores are not claimed to be calibrated probabilities.

## Acceptance artifact

Local engineering artifact:

```text
artifacts\phase13c_acceptance.json
```

It is not a replacement for Phase-11 scientific results or Phase-12 runtime results.

## Final checkpoint

```text
Dedicated tests              USER VERIFIED
Full regression              USER VERIFIED
Automated E2E acceptance     USER VERIFIED
Sequential file demo         USER VERIFIED
Browser microphone           USER VERIFIED
History ordering             USER VERIFIED
Stop / Clear lifecycle       USER VERIFIED
sounddevice live capture     USER VERIFIED
CPU end-to-end path          USER VERIFIED
Scientific freeze boundary   VERIFIED
commit                       COMPLETED
```
