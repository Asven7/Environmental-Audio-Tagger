from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class StreamWindow:
    index: int
    start_sample: int
    end_sample: int
    waveform: np.ndarray


class StreamingWindowBuffer:
    """Ring-like streaming buffer yielding overlapping fixed windows at a fixed hop.

    This class is independent of microphone libraries and is therefore testable with
    prerecorded or synthetic chunks. It is used by the optional live microphone CLI.
    """

    def __init__(self, window_samples: int, hop_samples: int) -> None:
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window_samples and hop_samples must be positive")
        if hop_samples > window_samples:
            raise ValueError("hop_samples must not exceed window_samples")
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self._buffer = np.empty(0, dtype=np.float32)
        self._samples_seen = 0
        self._next_emit_at = self.window_samples
        self._window_index = 0

    def push(self, chunk: np.ndarray) -> list[StreamWindow]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []
        self._buffer = np.concatenate([self._buffer, chunk])
        self._samples_seen += chunk.size
        emitted: list[StreamWindow] = []

        while self._samples_seen >= self._next_emit_at:
            # The current buffer contains all untrimmed samples. Compute absolute
            # window coordinates, then map to buffer-relative coordinates.
            end_sample = self._next_emit_at
            start_sample = end_sample - self.window_samples
            buffer_absolute_start = self._samples_seen - self._buffer.size
            local_start = start_sample - buffer_absolute_start
            local_end = end_sample - buffer_absolute_start
            if local_start < 0 or local_end > self._buffer.size:
                raise RuntimeError("Streaming buffer bookkeeping error")
            waveform = self._buffer[local_start:local_end].copy()
            emitted.append(
                StreamWindow(
                    index=self._window_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    waveform=waveform,
                )
            )
            self._window_index += 1
            self._next_emit_at += self.hop_samples

        # Keep enough history for the next overlapping window plus a small margin.
        earliest_needed = max(0, self._next_emit_at - self.window_samples)
        buffer_absolute_start = self._samples_seen - self._buffer.size
        trim = max(0, earliest_needed - buffer_absolute_start)
        if trim > 0:
            self._buffer = self._buffer[trim:]
        return emitted

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._samples_seen = 0
        self._next_emit_at = self.window_samples
        self._window_index = 0
