# Running inference and the local interface

Run commands from the repository root in an activated Python environment after installing the package (see [Reproducibility](REPRODUCIBILITY.md)).

## 1. Process a bundled synthetic example

The included `demo_crnn` checkpoint is trained on generated tones, **not** on UrbanSound8K. It is suitable for testing the software interfaces only.

```powershell
python scripts\infer_file.py `
  --checkpoint artifacts\demo_crnn\best_model.pt `
  --thresholds artifacts\demo_crnn\thresholds.json `
  --audio sample_data\demo\audio\test\demo_tone_180\000.wav
```

The package provides the `esaudio-infer` console entry point as well; run `esaudio-infer --help` to inspect options.

## 2. Run the file and browser-microphone UI

Install the optional interface dependency (`python -m pip install ".[ui]"` or `.[all]`) and run:

```powershell
python scripts\run_ui.py `
  --checkpoint artifacts\demo_crnn\best_model.pt `
  --thresholds artifacts\demo_crnn\thresholds.json `
  --device cpu
```

The Gradio interface defaults to local host `127.0.0.1`, port `7860`, and no public sharing. It supports sequential file analysis and browser microphone chunks. It records per-window labels, scores, timing and timestamped history. Stopping microphone analysis preserves the current history; the user can clear it explicitly.

## 3. Run the selected research model (local artifacts required)

When the complete original frozen experiment directory exists locally, the UI resolves the chosen CRNN run from `artifacts/experiments_phase11`:

```powershell
python scripts\run_ui.py `
  --experiment-root artifacts\experiments_phase11 `
  --device cpu
```

This directory, including its research checkpoints and threshold artifacts, is **not distributed** with the public GitHub clone. The program will not automatically download it.

## 4. Operating-system microphone interface

For the optional `sounddevice` path:

```powershell
python scripts\live_microphone.py --list-devices
```

With research artifacts present:

```powershell
python scripts\live_microphone.py --experiment-root artifacts\experiments_phase11 --device cpu
```

Use a suitable local input device and obtain permission before capturing speech or other identifiable recordings. The live program is not a reliable unknown-sound detector, and its predictions should not be interpreted as measured field accuracy without separately annotated samples.

## 5. Generate synthetic fixtures (optional)

```powershell
python scripts\generate_demo_data.py
python scripts\run_demo_pipeline.py
```

These commands create or update synthetic software-verification data and outputs. They may overwrite local example artifacts and should **not** be used as a substitute for scientific training or evaluation. Run them only when you intentionally want to regenerate those examples.

## 6. Troubleshooting

- If `ModuleNotFoundError: esaudio` occurs, install the project first with `python -m pip install -e ".[all]"`.
- If CUDA is unavailable, use `--device cpu` and a CPU PyTorch installation.
- For a browser audio permission issue, check local microphone permissions and that the UI is served on localhost.
- If a research checkpoint cannot be found, supply both `--checkpoint` and `--thresholds`, or restore the original frozen research artifact directory.
- Use `python scripts\run_ui.py --help` and `python scripts\live_microphone.py --help` to inspect current options.
