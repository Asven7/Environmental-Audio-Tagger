# Phase 0 Audit — Existing Implementation Baseline

## Purpose

This document records the baseline audit performed before the interactive, phase-gated implementation walkthrough. The existing repository is treated as a starting point, not as automatically trusted final code.

## Verified in the Agent Environment

- Python package syntax compilation succeeded (`python -m compileall -q src scripts tests`).
- After editable installation, the automated test suite passed (`13 passed`).
- The synthetic end-to-end engineering smoke pipeline completed successfully for both CNN and CRNN.
- The smoke pipeline produced checkpoints, thresholds, evaluation JSON, and runtime JSON.
- The synthetic smoke run is engineering verification only and is not an UrbanSound8K research result.

## Important Baseline Observation

Running `pytest` immediately after extracting the archive fails with `ModuleNotFoundError: esaudio` because this is a `src/`-layout package and the project must first be installed (for example, `python -m pip install -e .`). This is expected for the current package structure, but the local setup guide must make the install-before-test order explicit.

## Reuse Without Major Redesign

The following existing design choices are appropriate and should be retained:

- single local Python research application instead of client/server architecture;
- no database, REST API, authentication, message broker, or microservices;
- YAML configuration;
- CSV/JSON experiment manifests and results;
- fixed-window waveform preprocessing;
- shared Log-Mel feature extraction for training and inference;
- CNN baseline and CNN+GRU CRNN;
- `BCEWithLogitsLoss` for multi-label learning;
- validation-only threshold tuning;
- held-out-class rejection described as limited rejection rather than general open-set recognition;
- deterministic synthetic mixture manifests;
- runtime benchmarking and streaming buffer separation;
- optional local Gradio UI and optional `sounddevice` microphone adapter;
- synthetic smoke data for software verification only.

## Corrections / Improvements Required During the Walkthrough

1. **Environment setup must be revalidated on the user's actual OS and NVIDIA setup.**
   The previous build environment does not prove local CUDA compatibility.
2. **Python version should be selected conservatively for the user's machine.**
   Prefer Python 3.11 or 3.12 unless the local environment already has a fully compatible newer version.
3. **PyTorch installation must be matched to the local NVIDIA driver/CUDA compatibility.**
   Do not assume the existing pinned Torch wheel is the correct local GPU build.
4. **The default target/held-out class identities must be reviewed against real UrbanSound8K metadata before the final research run.**
   The protocol requires classes to be frozen before experiments, but the existing choice was not validated against the user's local copy of the metadata in this audit.
5. **The `.env.example` is currently misleading.**
   It declares `ESAUDIO_*` variables that the source code does not consume. Either implement environment-variable loading later or remove the file. The preferred minimal choice is removal unless a real need appears.
6. **Git ignore rules need review before the first commit.**
   The current ordering of checkpoint ignore/negation patterns may not preserve the intended tiny demo checkpoints. Generated data/model artifacts also need an explicit repository policy.
7. **Setup, testing, troubleshooting, Git workflow, manual test, quickstart, demo, and defense documents are not all present yet.**
   They will be added incrementally when the corresponding workflow is actually verified.
8. **Current automated coverage is a useful baseline but not complete.**
   Additional tests should cover training orchestration, grouped evaluation/rejection, runtime behavior, and the main CLI smoke paths as those phases are revisited.
9. **Final UrbanSound8K training/evaluation is not yet verified.**
   No research metric should be claimed until the frozen real-data experiment is run locally.
10. **Physical microphone capture is not yet user-verified.**
    The stream buffer is tested, but OS permissions, PortAudio, and the actual input device must be verified on the demonstration machine.

## Recommendation Triage

### MUST IMPLEMENT

- window-level multi-label task terminology;
- fixed train/validation/test protocol before final experiments;
- split-before-mix leakage prevention;
- controlled relative-level and overlap experiments;
- shared training/inference preprocessing;
- CNN baseline and lightweight CRNN;
- validation-only threshold tuning;
- frozen test evaluation;
- measurable runtime/no-backlog criterion;
- reproducibility (seeds, configs, saved artifacts);
- automated tests for critical paths;
- truthful limitation of held-out rejection semantics;
- local CPU inference path.

### SHOULD IMPLEMENT

- multi-seed final experiments;
- parameter-count and runtime comparison;
- local Gradio defense UI;
- physical microphone demo if stable on the user's machine;
- small real/external qualitative or annotated check if time/data permit;
- simple GitHub CI for installation + tests after local workflow is stable;
- manual test checklist and defense-focused documentation.

### OPTIONAL / FUTURE WORK

- LSTM comparison in addition to GRU;
- MFCC/ZCR/RMS auxiliary baseline;
- temporal output smoothing;
- pretrained audio baseline;
- probability calibration;
- ONNX/quantization;
- advanced open-set recognition;
- richer live visualization.

### REMOVE / DO NOT IMPLEMENT

- database;
- REST backend/API;
- authentication/authorization;
- cloud-only services;
- mandatory Docker path;
- microservices;
- Kafka/message queues;
- Kubernetes/service mesh;
- source separation/counting/localization;
- exact onset/offset SED;
- AudioSet-scale or large foundation-model training.

## Phase 0 Status

- Existing code audited: **DONE**
- Baseline automated tests after installation: **PASS (13)**
- Synthetic end-to-end smoke pipeline: **PASS**
- User local environment verification: **PENDING (Phase 1)**
- Real UrbanSound8K experiment: **PENDING (later ML phases)**
- Physical microphone verification: **PENDING (later integration phase)**
