import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_run_ui():
    spec = importlib.util.spec_from_file_location(
        "run_ui_test_module",
        ROOT / "scripts" / "run_ui.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_history_order_is_server_side_and_deterministic():
    module = _load_run_ui()
    rows = [
        [0, 0.0, 2.0],
        [1, 1.0, 3.0],
        [2, 2.0, 4.0],
    ]

    assert module._ordered_history(
        rows,
        module.HISTORY_OLDEST_FIRST,
    ) == rows
    assert module._ordered_history(
        rows,
        module.HISTORY_NEWEST_FIRST,
    ) == list(reversed(rows))

    assert rows[0][0] == 0
    assert rows[-1][0] == 2


def test_stop_event_updates_only_state_and_status():
    source = (ROOT / "scripts" / "run_ui.py").read_text(encoding="utf-8")

    assert "def stop_microphone(state):" in source
    assert "return stopped_state, status" in source
    assert "outputs=[mic_state, mic_status]" in source

    # Stop must not resend/re-render the score/history Dataframes.
    stop_block = source.split(
        "microphone.stop_recording(",
        1,
    )[1].split(")", 1)[0]
    assert "mic_scores" not in stop_block
    assert "mic_history" not in stop_block


def test_clear_is_explicit_button_not_audio_clear_lifecycle():
    source = (ROOT / "scripts" / "run_ui.py").read_text(encoding="utf-8")

    assert 'gr.Button("Clear results")' in source
    assert "clear_results.click(" in source
    assert "microphone.clear(" not in source
