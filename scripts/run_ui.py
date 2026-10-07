#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from esaudio.deployment import select_frozen_deployment
from esaudio.inference import AudioTagger
from esaudio.streaming import analyze_file
from esaudio.ui_support import (
    clear_microphone_state,
    file_results_headers,
    file_results_rows,
    mark_microphone_stopped,
    microphone_display_tuple,
    new_microphone_state,
    process_microphone_chunk,
)


HISTORY_OLDEST_FIRST = "Oldest first"
HISTORY_NEWEST_FIRST = "Newest first"


def _ordered_history(history: list[list], order: str) -> list[list]:
    rows = list(history)
    if order == HISTORY_NEWEST_FIRST:
        rows.reverse()
    return rows


def _with_history_order(result: tuple, order: str) -> tuple:
    values = list(result)
    values[-1] = _ordered_history(values[-1], order)
    return tuple(values)


def _resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    return requested


def _resolve_artifacts(args) -> tuple[Path, Path, str]:
    if bool(args.checkpoint) != bool(args.thresholds):
        raise ValueError("--checkpoint and --thresholds must be supplied together")
    if args.checkpoint:
        return (
            Path(args.checkpoint),
            Path(args.thresholds),
            "explicit checkpoint/threshold arguments",
        )

    selection = select_frozen_deployment(
        args.experiment_root,
        model_name="crnn",
    )
    return (
        Path(selection.checkpoint),
        Path(selection.thresholds),
        (
            f"frozen CRNN seed {selection.seed}, selected by validation mAP "
            f"({selection.best_validation_mAP:.6f})"
        ),
    )


def build_demo(tagger: AudioTagger, *, deployment_text: str):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the UI. Install it with: "
            "python -m pip install gradio"
        ) from exc

    file_headers = file_results_headers(tagger)
    score_headers = ["class", "score", "threshold", "active"]
    history_headers = [
        "window",
        "start_s",
        "end_s",
        "active_labels",
        "status",
        "processing_ms",
    ]

    def analyze_uploaded_file(path):
        if not path:
            return [], "Choose an audio file first."
        predictions = analyze_file(tagger, path)
        rows = file_results_rows(tagger, predictions)
        return (
            rows,
            (
                f"Processed {len(rows)} sequential window(s): "
                f"{tagger.window_seconds:.1f} s window / "
                f"{tagger.hop_seconds:.1f} s hop."
            ),
        )

    def start_microphone(order):
        state = new_microphone_state(tagger, recording=True)
        return _with_history_order(
            microphone_display_tuple(state),
            order,
        )

    def stream_microphone(audio, state, order):
        result = process_microphone_chunk(tagger, audio, state)
        return _with_history_order(result, order)

    def stop_microphone(state):
        # IMPORTANT: stop only updates backend state + status. It does NOT
        # re-send Dataframe values. This preserves the exact table already
        # visible in the browser for both Newest-first and Oldest-first modes.
        stopped_state, status, *_ = mark_microphone_stopped(tagger, state)
        return stopped_state, status

    def clear_microphone(order):
        result = clear_microphone_state(tagger)
        return _with_history_order(result, order)

    def reorder_history(state, order):
        if state is None:
            return []
        return _ordered_history(state.history, order)

    with gr.Blocks(title="Environmental Audio Tagger") as demo:
        gr.Markdown(
            "# Environmental Audio Tagger\n"
            f"**Deployment:** {deployment_text}  \n"
            f"**Runtime device:** `{next(tagger.model.parameters()).device}`  \n"
            f"**Protocol:** {tagger.window_seconds:.1f} s window / "
            f"{tagger.hop_seconds:.1f} s hop / batch size 1\n\n"
            "Scores are sigmoid model outputs and are **not calibrated "
            "probabilities**. `No confident known class` means no trained "
            "class crossed its frozen validation-selected threshold; it is "
            "**not** a general open-set recognition claim."
        )

        with gr.Tab("File"):
            file_audio = gr.Audio(
                label="Audio file",
                sources=["upload"],
                type="filepath",
            )
            analyze_button = gr.Button(
                "Analyze sequential windows",
                variant="primary",
            )
            file_status = gr.Textbox(label="Status", interactive=False)
            file_table = gr.Dataframe(
                headers=file_headers,
                datatype=[
                    "number",
                    "number",
                    "number",
                    "number",
                    "bool",
                    "str",
                    "str",
                    "number",
                    *(["number"] * len(tagger.class_names)),
                ],
                interactive=False,
                wrap=True,
                label="Window-level results",
            )
            analyze_button.click(
                analyze_uploaded_file,
                inputs=file_audio,
                outputs=[file_table, file_status],
            )

        with gr.Tab("Microphone"):
            gr.Markdown(
                "Use the microphone component's **record / stop** controls. "
                "The first result appears after a complete "
                f"{tagger.window_seconds:.1f} s window; later results follow "
                f"the {tagger.hop_seconds:.1f} s hop.\n\n"
                "**Important:** live microphone audio is out-of-domain relative "
                "to the controlled UrbanSound8K evaluation. The simple frozen "
                "threshold rejection can false-accept background/fan noise. "
                "Stopping recording preserves the complete session history.\n\n"
                "Use **History order** for newest/oldest ordering. "
                "Use **Clear results** only when you intentionally want to "
                "erase the current session results."
            )
            mic_state = gr.State(value=None)
            microphone = gr.Audio(
                label="Live microphone",
                sources=["microphone"],
                type="numpy",
                streaming=True,
            )
            history_order = gr.Radio(
                choices=[HISTORY_NEWEST_FIRST, HISTORY_OLDEST_FIRST],
                value=HISTORY_NEWEST_FIRST,
                label="History order",
            )
            clear_results = gr.Button("Clear results")

            with gr.Row():
                mic_status = gr.Textbox(
                    label="Current status",
                    interactive=False,
                )
                mic_active = gr.Textbox(
                    label="Active labels",
                    interactive=False,
                )
                mic_processing = gr.Number(
                    label="Latest processing time (ms)",
                    interactive=False,
                )
            mic_scores = gr.Dataframe(
                headers=score_headers,
                datatype=["str", "number", "number", "bool"],
                interactive=False,
                label="Current class scores",
            )
            mic_history = gr.Dataframe(
                headers=history_headers,
                datatype=[
                    "number",
                    "number",
                    "number",
                    "str",
                    "str",
                    "number",
                ],
                interactive=False,
                label="Live window history",
            )

            outputs = [
                mic_state,
                mic_status,
                mic_active,
                mic_processing,
                mic_scores,
                mic_history,
            ]
            microphone.start_recording(
                start_microphone,
                inputs=history_order,
                outputs=outputs,
            )
            microphone.stream(
                stream_microphone,
                inputs=[microphone, mic_state, history_order],
                outputs=outputs,
                stream_every=0.5,
                time_limit=3600,
            )

            # Deliberately only update state + status on Stop.
            microphone.stop_recording(
                stop_microphone,
                inputs=mic_state,
                outputs=[mic_state, mic_status],
            )

            # Clearing is an explicit UI action, not tied to the Audio
            # component lifecycle.
            clear_results.click(
                clear_microphone,
                inputs=history_order,
                outputs=outputs,
            )

            history_order.change(
                reorder_history,
                inputs=[mic_state, history_order],
                outputs=mic_history,
            )

        with gr.Tab("Interpretation"):
            gr.Markdown(
                "This demo performs **window-level multi-label audio tagging / "
                "event presence detection**. It does not estimate exact onset/"
                "offset times, source counts, source separation, or sound-source "
                "location. The microphone history timestamps refer to analysis "
                "windows relative to the beginning of the current recording."
            )

    demo.queue()
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase-13 file + streaming microphone Gradio demo"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--thresholds")
    parser.add_argument(
        "--experiment-root",
        default="artifacts/experiments_phase11",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    checkpoint, thresholds, deployment_text = _resolve_artifacts(args)
    device = _resolve_device(args.device)
    tagger = AudioTagger.from_files(
        checkpoint,
        thresholds,
        device=device,
    )
    demo = build_demo(tagger, deployment_text=deployment_text)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
