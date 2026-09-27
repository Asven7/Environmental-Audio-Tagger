#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from esaudio.inference import AudioTagger


def build_demo(tagger: AudioTagger) -> gr.Blocks:
    def analyze(audio_path: str | None):
        if not audio_path:
            return "Please provide an audio file or record from the microphone.", pd.DataFrame(), None
        try:
            records = tagger.predict_file_windows(audio_path)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}", pd.DataFrame(), None

        rows = []
        for record in records:
            active = ", ".join(record["active_labels"]) if record["active_labels"] else "NO_CONFIDENT_KNOWN_CLASS"
            row = {
                "window": record["window_index"],
                "start_s": round(record["start_seconds"], 3),
                "end_s": round(record["end_seconds"], 3),
                "active_labels": active,
                "status": record["status"],
                "total_ms": round(record["total_ms"], 2),
            }
            for name in tagger.class_names:
                row[name] = round(record["scores"][name], 4)
            rows.append(row)
        frame = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(9, 4))
        if rows:
            x = [record["start_seconds"] for record in records]
            for name in tagger.class_names:
                ax.plot(x, [record["scores"][name] for record in records], label=name)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Window start time (s)")
            ax.set_ylabel("Sigmoid score")
            ax.set_title("Per-class scores across audio windows")
            if len(tagger.class_names) <= 10:
                ax.legend(loc="upper right", fontsize="small", ncol=2)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()

        mean_ms = sum(record["total_ms"] for record in records) / max(len(records), 1)
        summary = (
            f"Processed **{len(records)}** window(s). "
            f"Window={tagger.window_seconds:.2f}s, hop={tagger.hop_seconds:.2f}s, "
            f"mean compute={mean_ms:.1f} ms."
        )
        return summary, frame, fig

    with gr.Blocks(title="Environmental Audio Tagger") as demo:
        gr.Markdown(
            "# Environmental Audio Tagger\n"
            "Window-level multi-label environmental audio tagging. "
            "The `NO_CONFIDENT_KNOWN_CLASS` state means no trained class crossed its validation-selected threshold; "
            "it is **not** a general open-set recognition claim."
        )
        audio = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio input",
            format="wav",
        )
        run_button = gr.Button("Analyze audio", variant="primary")
        status = gr.Markdown()
        table = gr.Dataframe(label="Window-level predictions", interactive=False, wrap=True)
        plot = gr.Plot(label="Score timeline")
        run_button.click(analyze, inputs=audio, outputs=[status, table, plot])
        audio.clear(lambda: ("", pd.DataFrame(), None), outputs=[status, table, plot])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch local Gradio demo UI")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    tagger = AudioTagger.from_files(args.checkpoint, args.thresholds, device=args.device)
    demo = build_demo(tagger)
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
