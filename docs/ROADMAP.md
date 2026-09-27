# Interactive Implementation Roadmap

This roadmap is intentionally phase-gated. A phase is not considered complete until the user verifies the required local checks.

## Phase 0 — Implementation Audit and Scope Lock
Audit proposal, prior recommendations, and existing code. Lock architecture/scope and identify local prerequisites.

## Phase 1 — Local Development Environment
Identify OS/shell, Python, Git, NVIDIA driver/GPU/VRAM, disk/RAM, and create a clean Python virtual environment. Install the minimum CPU-capable dependency set first; enable CUDA only after compatibility is verified.

## Phase 2 — Repository Baseline, Git, and Documentation Skeleton
Unpack/reconcile the existing repository, fix `.gitignore`, resolve misleading configuration files, initialize Git, establish the stable baseline commit, and add the first verified setup/testing/Git documents.

## Phase 3 — Core Audio and Configuration Foundation
Walk through and verify config validation, audio loading, mono conversion, resampling, fixed-window crop/pad, RMS normalization, and unit tests.

## Phase 4 — Dataset Protocol and Manifest Generation
Prepare UrbanSound8K metadata, review/freeze target and held-out classes, freeze folds, generate manifests, enforce source leakage checks, and document dataset handling/licensing.

## Phase 5 — Controlled Multi-Label Mixture Pipeline
Verify relative-level mixing, temporal-overlap synthesis, deterministic mixture metadata, and edge-case tests.

## Phase 6 — Log-Mel Feature Pipeline
Verify STFT/Mel parameters, feature normalization, tensor shapes, train/inference parity, and resource behavior.

## Phase 7 — CNN Baseline
Train/evaluate the CNN on a small smoke subset first, verify checkpointing and metrics, then prepare the real-data run.

## Phase 8 — CRNN (CNN + GRU)
Implement/verify the temporal model, compare parameter count and behavior with CNN, and keep LSTM outside the default experiment matrix.

## Phase 9 — Training Protocol and Reproducibility
Verify `BCEWithLogitsLoss`, class weighting, optimizer, early stopping, deterministic seeds, checkpoint selection, and experiment metadata.

## Phase 10 — Threshold Tuning and Frozen Evaluation
Tune thresholds on validation only; evaluate frozen checkpoints on test; produce micro/macro metrics, mAP, Hamming Loss, per-class and controlled-condition breakdowns.

## Phase 11 — Multi-Seed Research Experiments
Run the final CNN/CRNN experiment matrix, aggregate mean/std, record actual hardware and runtime, and generate defensible result tables without fabricated claims.

## Phase 12 — File Inference and Runtime Benchmark
Verify end-to-end file windowing, prediction output, batch=1 CPU/GPU timing, p95 no-backlog criterion, and failure handling.

## Phase 13 — Local UI and Microphone Integration
Verify the Gradio demo, then optionally verify continuous microphone capture on the actual machine. Keep the UI separate from scientific evaluation.

## Phase 14 — Testing, QA, and Troubleshooting Hardening
Expand meaningful automated coverage, build the manual test checklist, run resource/performance checks, and document encountered failures/fixes.

## Phase 15 — GitHub and CI
Create/connect the GitHub repository with user authorization, push the stable history, and add a simple CI workflow if local installation/tests are already reliable.

## Phase 16 — Documentation, Clean Install, and Reproducibility Audit
Perform a clean-install simulation and finalize README, setup, testing, troubleshooting, decisions, traceability, quickstart, project log, and changelog.

## Phase 17 — University Demo and Defense Preparation
Finalize the 5–15 minute demo script, screenshots, result tables, limitations, likely technical questions, defense notes, and academic release/tag.
