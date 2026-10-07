# Phase 13B — File UI and Live Microphone Adapter

## Status

**USER VERIFIED — COMMITTED**

## Scope

Phase 13B connects the tested Phase-13A core to:

```text
1. Gradio file + browser-microphone UI
2. local sounddevice microphone CLI
```

Neither path changes the frozen CRNN, thresholds, feature configuration, 2-second window, or 1-second hop.

## Verified local environment

```text
Python 3.11.9
Gradio 6.29.1
sounddevice 0.5.6
CPU inference path
```

Optional dependencies:

```text
ui   -> gradio>=5,<7
live -> sounddevice>=0.5,<0.6
all  -> UI + live + dev dependencies
```

## Gradio UI

`scripts/run_ui.py` provides File, Microphone, and Interpretation surfaces.

### File

The complete file is analyzed as sequential overlapping windows.

Each row includes:

```text
window index
start time
nominal end time
valid audio end time
padding flag
active labels
display status
processing time
one score column per target class
```

A verified 2.527 s example produced nominal windows:

```text
0.0 -> 2.0 s
1.0 -> 3.0 s
2.0 -> 4.0 s
```

with expected right padding near the end.

### Microphone

Browser microphone chunks are:

```text
downmixed to mono
converted to float32
resampled when required
fed into the rolling target-rate buffer
```

No prediction is produced before the first complete 2-second window. Later predictions follow the 1-second hop.

The UI shows:

```text
current status
active labels
latest processing time
per-class score
per-class frozen threshold
per-class active flag
complete timestamped live-window history
```

`No confident known class` remains a threshold heuristic, not an unknown detector.

## Session lifecycle

Stop:

```text
marks session stopped
preserves latest scores
preserves complete live history
```

Verified message:

```text
Recording stopped. Results preserved.
```

Only `Clear results` removes the visible results/history.

History ordering:

```text
Newest first
Oldest first
```

was user verified.

## sounddevice CLI

`scripts/live_microphone.py` uses `StreamingInferenceEngine`.

Capture occurs in the PortAudio callback while model inference stays on the main Python thread; chunks are copied through a bounded queue.

Verified input:

```text
Microphone (C-Media(R) Audio), MME
22050 Hz
1 channel
float32
```

Sequential windows and clean `Ctrl+C` shutdown were verified.

## Deployment artifact

Default demo paths resolve the frozen deployment from:

```text
artifacts/experiments_phase11
```

Verified deployment:

```text
model = CRNN
seed = 23
validation mAP = 0.675714
```

Explicit `--checkpoint` + `--thresholds` remain available together.

Default device policy:

```text
auto -> CUDA when available, otherwise CPU
```

CPU remains fully supported.

## Installation

Developer checkout:

```powershell
python -m pip install -e ".[all]"
```

Clean/non-editable installation:

```powershell
python -m pip install ".[all]"
```

Version inspection:

```powershell
python -c "from importlib.metadata import version; print('gradio=', version('gradio')); print('sounddevice=', version('sounddevice'))"
```

## Live/OOD limitation

Quiet-room / laptop-fan audio produced false positives such as:

```text
siren
air_conditioner
jackhammer
```

This is consistent with the frozen weak OOD rejection result.

Correct interpretation:

```text
live pipeline works
false positives expose a model/rejection limitation
no threshold/model changes were made after held-out evaluation
```

## Scientific boundary

Phase 13B did not:

```text
retrain
retune thresholds
read held-out labels for development
recompute held-out metrics
add smoothing
change frozen scientific results
```

## Final checkpoint

```text
File UI                  USER VERIFIED
Browser microphone       USER VERIFIED
Full history retention   USER VERIFIED
History ordering         USER VERIFIED
Stop / Clear lifecycle   USER VERIFIED
sounddevice device list  USER VERIFIED
sounddevice live capture USER VERIFIED
CPU live inference       USER VERIFIED
commit                   COMPLETED
```
