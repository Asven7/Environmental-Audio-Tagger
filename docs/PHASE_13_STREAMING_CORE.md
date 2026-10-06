# Phase 13A — Streaming and Sequential File Core

## Status

**IMPLEMENTED — WAITING FOR USER LOCAL VERIFICATION**

## Goal

Phase 13A implements the UI-independent logic required by the proposal:

```text
file -> sequential 2 s windows / 1 s hop
microphone chunks -> continuous rolling buffer
first live result after a complete 2 s window
subsequent live result every 1 s hop
per-window timestamps
per-window scores / active labels / transparent rejection status
per-window processing time
```

Gradio and `sounddevice` are deliberately deferred to Phase 13B so this signal-flow logic
can be unit tested without optional GUI/audio-device dependencies.

## Sequential file behavior

`iter_waveform_windows(...)` starts at time zero and advances by the frozen one-second
hop. Every model input has exactly the frozen two-second window length.

For a file whose final window is incomplete, the final window is right-padded with zeros.
The output retains both:

```text
end_seconds       = nominal two-second window boundary
valid_end_seconds = end of actual file audio
padded            = whether zero padding was required
```

This makes UI timestamps explicit rather than silently treating padded audio as recorded
content.

## Live rolling buffer

`RollingWindowBuffer` accepts arbitrary chunk lengths. It supports mono `[T]` and
sounddevice-style `[T, C]` arrays. Multichannel input is downmixed by arithmetic mean,
consistent with project file loading.

Example with the frozen protocol:

```text
0.0 s ---------------- 2.0 s
       first window -> inference

1.0 s ---------------- 3.0 s
       second window -> inference

2.0 s ---------------- 4.0 s
       third window -> inference
```

No output is emitted before the first full two-second window exists.

The buffer discards audio that can no longer participate in a future window, so memory
usage remains bounded during long microphone sessions.

## Streaming inference

`StreamingInferenceEngine` wraps a frozen `AudioTagger`.

For every emitted window it records:

```text
window index
window start time
window end time
scores for all classes
active labels
raw inference status
display status
processing time in milliseconds
```

The rejection display wording is deliberately:

```text
No confident known class
```

This does not claim robust unknown/open-set recognition.

## File analysis

`analyze_file(...)`:

1. loads and resamples the file to the checkpoint sample rate;
2. splits the complete file into sequential overlapping windows;
3. calls the same frozen `AudioTagger.predict_waveform(...)` path for every window;
4. returns timestamped predictions suitable for the Phase-13B UI.

This is different from the old one-file/one-window convenience inference path.

## Scope boundary

Phase 13A does not:

```text
retrain
retune thresholds
change window/hop
smooth predictions between windows
open a microphone device
launch Gradio
change Phase-11 or Phase-12 results
```

Optional multi-window smoothing mentioned in the proposal remains disabled unless a
separate, explicitly declared postprocessing experiment is later justified.
