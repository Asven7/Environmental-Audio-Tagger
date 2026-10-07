# Phase 13C — End-to-End Local Demo Verification

## Status

**USER VERIFIED — READY TO COMMIT**

## Goal

Phase 13C is the final acceptance pass for the Phase-13 demo surface. It does not add a
new model, change thresholds, or reopen the frozen test protocol.

The purpose is to verify that the already-frozen deployment works end-to-end through:

```text
frozen deployment selection
-> checkpoint + threshold loading
-> file decoding / sequential windows
-> canonical inference
-> Gradio UI construction
-> browser microphone demo
-> local sounddevice microphone path
```

## Automated acceptance

The dedicated acceptance test suite was run locally:

```powershell
python -m pytest tests/test_phase13_demo_acceptance.py -v
```

Result:

```text
4 passed
```

The full project regression suite was then run:

```powershell
python -m pytest
```

Result:

```text
155 passed
```

The automated end-to-end acceptance command was:

```powershell
python scripts\verify_phase13_demo.py `
  --project-root . `
  --experiment-root artifacts\experiments_phase11 `
  --audio-file data\UrbanSound8K\audio\fold8\103076-3-0-0.wav `
  --device cpu `
  --output artifacts\phase13c_acceptance.json
```

Verified output:

```text
status=PASS
deployment=crnn_seed23 validation_mAP=0.675714
protocol=22050Hz 2.0s-window 1.0s-hop
file_mode=3 windows
max_processing_ms=8.869
below_hop=True
ui=Blocks
gradio=6.29.1
default_input=1 Microphone (C-Media(R) Audio)
Held-out accuracy metrics were NOT recomputed.
Thresholds were NOT changed.
```

## Frozen deployment verified

The acceptance report confirmed:

```text
model = CRNN
seed = 23
best validation mAP = 0.6757137110147023
sample rate = 22050 Hz
window = 2.0 s
hop = 1.0 s
device = CPU
```

Frozen artifact hashes:

```text
checkpoint SHA-256 =
80079291a52bca7cadd25e39bf05761e378bf619d812c532fa1b2fafc9e615b8

threshold SHA-256 =
788d5722162b66b75a59bfead064faa5513a5cdcf312e64eca7d0dd95ec022d2
```

No model weights or thresholds were modified during Phase 13C.

## Sequential file acceptance

The known demo/validation audio:

```text
data\UrbanSound8K\audio\fold8\103076-3-0-0.wav
```

produced three sequential windows under the frozen 2-second window / 1-second hop protocol:

```text
window 0: 0.0 -> 2.0 s
window 1: 1.0 -> 3.0 s
window 2: 2.0 -> 4.0 s
```

The automated acceptance observed a maximum processing time of approximately:

```text
8.869 ms
```

which was below the 1000 ms hop.

This is an engineering smoke observation, not a replacement for the Phase-12 runtime
benchmark.

## Manual Gradio acceptance

A final manual end-to-end UI smoke test was completed successfully.

Verified:

```text
File tab: PASS
Browser microphone: PASS
Newest first history order: PASS
Oldest first history order: PASS
Stop preserves history: PASS
Clear results: PASS
Clean UI shutdown: PASS
```

The File tab displayed the expected sequential windows, scores, active labels, timestamps,
padding metadata, and processing time.

The browser microphone produced rolling predictions after the first complete 2-second
window and continued on the 1-second hop.

The complete recording-session history remained available after Stop.

## Manual sounddevice acceptance

The local sounddevice path was also verified successfully.

Default input device:

```text
index = 1
name = Microphone (C-Media(R) Audio)
```

The CLI:

```powershell
python scripts\live_microphone.py `
  --experiment-root artifacts\experiments_phase11 `
  --device cpu
```

produced sequential windows and shut down cleanly with `Ctrl+C`.

Verified:

```text
sounddevice live capture: PASS
sequential rolling windows: PASS
clean shutdown: PASS
```

## Scientific boundary

The acceptance report explicitly confirmed:

```text
retrained = false
thresholds_retuned = false
heldout_test_read_for_development = false
accuracy_metrics_recomputed = false
```

Phase 13C therefore does not alter the frozen scientific result.

## Live / OOD limitation

The final demo is operational, but live microphone predictions are not reliable open-set
recognition.

During live testing, quiet-room / laptop-fan audio frequently produced false-positive known
classes such as:

```text
siren
air_conditioner
jackhammer
```

This behavior is consistent with the already-frozen weak OOD rejection result.

The correct project claim remains:

```text
real-time window-level multi-label audio tagging / event presence detection is operational
```

not:

```text
robust open-set environmental sound recognition
```

Scores are sigmoid model outputs and are not calibrated probabilities.

No post-test threshold change, smoothing rule, or model modification was introduced to hide
this limitation.

## Acceptance artifact

The local engineering acceptance report is written to:

```text
artifacts\phase13c_acceptance.json
```

It records engineering verification only. It is not a new test-set evaluation artifact and
does not replace the frozen Phase-11 or Phase-12 results.

## Phase checkpoint

```text
Dedicated Phase-13C tests      USER VERIFIED
Full regression suite          USER VERIFIED
Automated E2E acceptance       USER VERIFIED
Sequential file demo           USER VERIFIED
Browser microphone             USER VERIFIED
History ordering               USER VERIFIED
Stop / Clear lifecycle         USER VERIFIED
sounddevice live capture       USER VERIFIED
CPU end-to-end path            USER VERIFIED
Scientific freeze boundary     VERIFIED
```

**Phase 13C is complete and ready for Git commit.**
