import numpy as np

from esaudio.streaming import StreamingWindowBuffer


def test_streaming_window_buffer_overlap():
    buffer = StreamingWindowBuffer(window_samples=8, hop_samples=4)
    out = []
    out.extend(buffer.push(np.arange(0, 3, dtype=np.float32)))
    assert out == []
    out.extend(buffer.push(np.arange(3, 10, dtype=np.float32)))
    assert len(out) == 1
    np.testing.assert_array_equal(out[0].waveform, np.arange(0, 8, dtype=np.float32))
    out.extend(buffer.push(np.arange(10, 14, dtype=np.float32)))
    assert len(out) == 2
    np.testing.assert_array_equal(out[1].waveform, np.arange(4, 12, dtype=np.float32))
