# Phase 13A — Streaming and Sequential File Core

## Status

**USER VERIFIED — COMPLETED**

Phase 13A implements the UI-independent streaming/file logic later verified end-to-end in Phases 13B/13C.

## Goal

```text
file -> sequential 2 s windows / 1 s hop
microphone chunks -> continuous rolling buffer
first live result after a complete 2 s window
subsequent live result every 1 s hop
per-window timestamps
per-window scores / active labels / transparent rejection status
per-window processing time
```

Gradio and `sounddevice` are adapters around this tested core.

## Sequential file behavior

`iter_waveform_windows(...)` starts at time zero and advances by the frozen one-second hop.

Every model input has exactly the frozen two-second length.

For an incomplete final window, right padding is used while retaining:

```text
end_seconds       = nominal two-second window boundary
valid_end_seconds = end of actual file audio
padded            = whether zero padding was required
```

## Live rolling buffer

`RollingWindowBuffer` accepts arbitrary chunk lengths, supports mono `[T]` and sounddevice-style `[T, C]`, and downmixes multichannel audio by arithmetic mean.

Frozen example:

```text
0.0 ---------------- 2.0 s  -> window 0
1.0 ---------------- 3.0 s  -> window 1
2.0 ---------------- 4.0 s  -> window 2
```

No output is emitted before the first full two-second window exists.

Audio that can no longer participate in a future window is discarded, keeping memory bounded.

## Streaming inference

`StreamingInferenceEngine` wraps the frozen `AudioTagger`.

Each emitted window records:

```text
window index
start/end time
all class scores
active labels
raw inference status
display status
processing time
```

Rejection display wording:

```text
No confident known class
```

This does not claim robust unknown/open-set recognition.

## File analysis

`analyze_file(...)`:

1. loads/resamples audio to the checkpoint sample rate;
2. splits it into sequential overlapping windows;
3. calls the same frozen `AudioTagger.predict_waveform(...)` path per window;
4. returns timestamped predictions for the UI.

## Completed verification

Phase 13B/13C subsequently verified:

```text
sequential file path
browser microphone path
sounddevice microphone path
rolling history
CPU frozen inference
Stop/Clear lifecycle
```

## Scope boundary

Phase 13A does not:

```text
retrain
retune thresholds
change window/hop
smooth predictions
change Phase-11/12 results
```

Optional multi-window smoothing remains disabled.
