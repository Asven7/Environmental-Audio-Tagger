from __future__ import annotations

import argparse

import torch

from esaudio.config import load_config, window_samples
from esaudio.features import feature_extractor_from_config
from esaudio.models import CRNNTagger, build_model, count_parameters


def _make_waveform_batch(batch_size: int, sample_rate: int, samples: int) -> torch.Tensor:
    """Create deterministic waveforms for an engineering smoke test."""
    t = torch.arange(samples, dtype=torch.float32) / float(sample_rate)
    base_frequencies = [220.0, 440.0, 880.0, 1320.0]
    rows = []
    for index in range(batch_size):
        frequency = base_frequencies[index % len(base_frequencies)]
        waveform = 0.25 * torch.sin(2.0 * torch.pi * frequency * t)
        waveform += 0.05 * torch.sin(2.0 * torch.pi * (frequency * 1.5) * t)
        rows.append(waveform)
    return torch.stack(rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Phase-8 CRNN architecture.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    torch.manual_seed(1234)
    config = load_config(args.config)
    sample_rate = int(config["project"]["sample_rate"])
    samples = int(window_samples(config))
    num_classes = len(config["project"]["target_classes"])
    waveforms = _make_waveform_batch(args.batch_size, sample_rate, samples)

    extractor = feature_extractor_from_config(config)
    model, spec = build_model(config, "crnn", num_classes)
    if not isinstance(model, CRNNTagger):
        raise RuntimeError("build_model did not return CRNNTagger")

    extractor.eval()
    model.eval()

    with torch.no_grad():
        features = extractor(waveforms)
        sequence_input = model.sequence_features(features)
        recurrent_output, _ = model.rnn(sequence_input)
        logits_1 = model(features)
        logits_2 = model(features)
        scores = torch.sigmoid(logits_1)

    print("=== Phase 8 CRNN Architecture Smoke Test ===")
    print(f"sample_rate={sample_rate} samples={samples} batch_size={args.batch_size}")
    print(f"feature_shape={tuple(features.shape)}")
    print(f"sequence_input_shape={tuple(sequence_input.shape)}")
    print(f"recurrent_output_shape={tuple(recurrent_output.shape)}")
    print(f"logit_shape={tuple(logits_1.shape)}")
    print(
        "recurrent="
        f"{model.recurrent_type} hidden={model.recurrent_hidden} "
        f"layers={model.recurrent_layers} bidirectional={model.rnn.bidirectional}"
    )
    print(f"parameter_count={count_parameters(model)}")
    print(f"raw_logit_range=({float(logits_1.min()):.6f}, {float(logits_1.max()):.6f})")
    print(f"external_sigmoid_score_range=({float(scores.min()):.6f}, {float(scores.max()):.6f})")

    expected_features = (
        args.batch_size,
        1,
        int(config["features"]["n_mels"]),
        features.shape[-1],
    )
    if tuple(features.shape) != expected_features:
        raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
    if tuple(sequence_input.shape) != (args.batch_size, features.shape[-1], 64):
        raise RuntimeError(f"Unexpected recurrent input shape: {tuple(sequence_input.shape)}")
    if tuple(recurrent_output.shape) != (args.batch_size, features.shape[-1], model.recurrent_hidden):
        raise RuntimeError(f"Unexpected recurrent output shape: {tuple(recurrent_output.shape)}")
    if tuple(logits_1.shape) != (args.batch_size, num_classes):
        raise RuntimeError(f"Unexpected CRNN output shape: {tuple(logits_1.shape)}")
    if model.rnn.bidirectional:
        raise RuntimeError("Default CRNN must remain unidirectional")
    if not torch.isfinite(logits_1).all():
        raise RuntimeError("CRNN produced non-finite CPU logits")
    if not torch.equal(logits_1, logits_2):
        raise RuntimeError("CRNN eval forward is not deterministic on CPU")
    print("CPU forward/sequence/determinism: PASSED")

    if torch.cuda.is_available():
        cuda_extractor = feature_extractor_from_config(config).cuda()
        cuda_model, _ = build_model(config, "crnn", num_classes)
        cuda_model.load_state_dict(model.state_dict())
        cuda_model = cuda_model.cuda()
        cuda_extractor.eval()
        cuda_model.eval()

        with torch.no_grad():
            cuda_features = cuda_extractor(waveforms.cuda())
            cuda_logits = cuda_model(cuda_features).cpu()

        if tuple(cuda_logits.shape) != (args.batch_size, num_classes):
            raise RuntimeError("Unexpected CUDA CRNN output shape")
        if not torch.isfinite(cuda_logits).all():
            raise RuntimeError("CRNN produced non-finite CUDA logits")

        max_abs_diff = float((logits_1 - cuda_logits).abs().max())
        if max_abs_diff > 1e-3:
            raise RuntimeError(
                f"CPU/CUDA CRNN logits differ unexpectedly (max_abs_diff={max_abs_diff:.6f})"
            )
        print(
            "CUDA forward/finite check: PASSED "
            f"(device={torch.cuda.get_device_name(0)}, max_abs_diff={max_abs_diff:.6f})"
        )
    else:
        print("CUDA forward/finite check: SKIPPED (CUDA unavailable)")

    print("CRNN architecture smoke test: PASSED")
    print("No model training was performed and no checkpoint was written.")


if __name__ == "__main__":
    main()
