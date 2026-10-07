# Phase 13B — File UI and Live Microphone Adapter

## Status

**USER VERIFIED — READY TO COMMIT**

## Scope

Phase 13B connects the Phase-13A streaming core to two demo interfaces:

```text
1. Gradio file + browser-microphone UI
2. local sounddevice microphone CLI
```

Neither path changes the frozen CRNN, thresholds, feature configuration, 2-second window,
or 1-second hop.

## Verified local environment

User-local verification was completed on Windows with:

```text
Python 3.11.9
Gradio 6.29.1
sounddevice 0.5.6
CPU inference path
```

The project optional dependencies already define the UI/live packages in `pyproject.toml`:

```text
ui   -> gradio>=5,<7
live -> sounddevice>=0.5,<0.6
all  -> UI + live + dev dependencies
```

## Gradio UI

`scripts/run_ui.py` provides the File, Microphone, and Interpretation tabs.

### File

The complete audio file is analyzed as sequential overlapping windows.

Each row contains:

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

This is the proposal's window-level file demo rather than a one-file/one-window shortcut.

User-local verification confirmed a 2.527 s UrbanSound8K example produced three windows:

```text
0.0 -> 2.0 s
1.0 -> 3.0 s
2.0 -> 4.0 s
```

with right padding on the final two nominal windows as expected.

### Microphone

The Gradio `Audio` component uses browser microphone capture and streaming events.

The adapter:

```text
downmixes to mono
converts integer PCM to float32 [-1, 1]
resamples to the checkpoint sample rate when necessary
maintains cumulative sample counts
feeds the Phase-13A rolling buffer
```

No prediction is produced until the first complete two-second target-rate window exists.
After that, predictions follow the frozen one-second hop.

The UI shows:

```text
current status
active labels
latest processing time
per-class score
per-class frozen threshold
per-class active flag
complete timestamped live-window history for the current recording session
```

`No confident known class` is intentionally used instead of claiming generic unknown
recognition.

## Microphone session lifecycle

Starting a new recording creates a fresh session state.

Stopping recording:

```text
marks the backend session as stopped
updates the visible status
preserves the score table
preserves the complete live-window history
```

The final verified stop message is:

```text
Recording stopped. Results preserved.
```

Results are cleared only through the explicit `Clear results` control.

### History ordering

The UI provides a dedicated server-side `History order` control:

```text
Newest first
Oldest first
```

This avoids depending on transient browser-side Dataframe column sorting while streaming
updates are still arriving.

User-local verification confirmed both orderings remain usable and that stopping a recording
does not remove the current session history.

## sounddevice CLI

`scripts/live_microphone.py` is the local sounddevice implementation and uses
`StreamingInferenceEngine`.

Audio capture runs in PortAudio's callback while model inference runs on the main Python
thread. The callback copies chunks into a bounded queue rather than running PyTorch inside
the real-time audio callback.

The verified default microphone path used:

```text
Microphone (C-Media(R) Audio), MME
```

The InputStream successfully operated at the frozen checkpoint sample rate:

```text
sample rate = 22050 Hz
channels = 1
dtype = float32
```

User-local verification produced sequential predictions with:

```text
2.0 s analysis window
1.0 s hop
window indices 0, 1, 2, ...
clean Ctrl+C shutdown
```

and ended with:

```text
Microphone capture stopped.
```

## Deployment artifact

By default both demo paths resolve the frozen CRNN deployment run from:

```text
artifacts/experiments_phase11
```

using the Phase-12 validation-only deployment-selection rule.

The verified deployment was:

```text
model = CRNN
seed = 23
validation mAP = 0.675714
```

Explicit `--checkpoint` and `--thresholds` remain available together for debugging and
backward compatibility.

Default runtime device policy:

```text
auto -> CUDA when available, otherwise CPU
```

CPU remains a fully supported path and was explicitly verified for this phase.

## Optional dependencies

The dependencies are already declared in `pyproject.toml`.

Recommended project installation forms are:

```powershell
python -m pip install -e ".[ui]"
python -m pip install -e ".[live]"
```

or for the complete development/demo environment:

```powershell
python -m pip install -e ".[all]"
```

To inspect installed versions without relying on package-specific version attributes:

```powershell
python -c "from importlib.metadata import version; print('gradio=', version('gradio')); print('sounddevice=', version('sounddevice'))"
```

## Verified automated tests

During Phase 13B, the user locally verified the dedicated UI/streaming tests and repeated
full-project regression tests after the UI lifecycle fixes.

The final regression suite passed completely before this documentation-only update.

## Live / OOD limitation observed during verification

Live microphone verification exposed an important deployment limitation.

In a quiet environment with laptop-fan/background sound, the frozen model frequently
activated known classes such as:

```text
siren
air_conditioner
jackhammer
```

This is consistent with the already-frozen evaluation result showing weak rejection of
out-of-distribution audio. Therefore:

```text
the live microphone pipeline is functioning
the observed false positives are a model/rejection limitation
the UI must not present this as robust unknown/open-set recognition
```

No thresholds, model weights, feature parameters, or frozen evaluation artifacts were
changed in response to this observation.

This limitation must be retained in the final report and defense discussion.

## Scientific boundary

Phase 13B does not:

```text
retrain
retune thresholds
read held-out test labels for development
recompute accuracy/F1/mAP
smooth scores across windows
change the frozen scientific result
```

The UI is a demonstration/deployment layer over the frozen inference system.

## Phase checkpoint

```text
File UI                  USER VERIFIED
Browser microphone       USER VERIFIED
Full history retention   USER VERIFIED
History ordering         USER VERIFIED
Stop / Clear lifecycle   USER VERIFIED
sounddevice device list  USER VERIFIED
sounddevice live capture USER VERIFIED
CPU live inference       USER VERIFIED
```

**Phase 13B is functionally complete and ready for Git commit.**
