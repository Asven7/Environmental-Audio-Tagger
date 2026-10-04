from __future__ import annotations

import argparse

import torch

from esaudio.config import load_config, window_samples
from esaudio.features import feature_extractor_from_config
from esaudio.models import build_model, count_parameters


def _make_waveform_batch(batch_size: int, sample_rate: int, samples: int) -> torch.Tensor:
    """Create a deterministic engineering-smoke batch; this is not evaluation data."""
    t = torch.arange(samples, dtype=torch.float32) / float(sample_rate)
    base_frequencies = [220.0, 440.0, 880.0, 1320.0]
    rows = []
    for index in range(batch_size):
        frequency = base_frequencies[index % len(base_frequencies)]
        waveform = 0.25 * torch.sin(2.0 * torch.pi * frequency * t)
        waveform += 0.05 * torch.sin(2.0 * torch.pi * (frequency * 1.5) * t)
        rows.append(waveform)
    return torch.stack(rows, dim=0)


def _forward(
    config: dict,
    waveforms: torch.Tensor,
    device: torch.device,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    extractor = feature_extractor_from_config(config).to(device)
    model, _ = build_model(config, "cnn", len(config["project"]["target_classes"]))
    model = model.to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    extractor.eval()

    with torch.no_grad():
        features = extractor(waveforms.to(device))
        logits = model(features)
    return features.cpu(), logits.cpu(), count_parameters(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Phase-7 CNN baseline.")
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

    model, _ = build_model(config, "cnn", num_classes)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    # Reuse the same initialized weights on CPU and CUDA.
    cpu_extractor = feature_extractor_from_config(config)
    cpu_model, _ = build_model(config, "cnn", num_classes)
    cpu_model.load_state_dict(initial_state)
    cpu_extractor.eval()
    cpu_model.eval()

    with torch.no_grad():
        cpu_features = cpu_extractor(waveforms)
        cpu_logits_1 = cpu_model(cpu_features)
        cpu_logits_2 = cpu_model(cpu_features)
        cpu_scores = torch.sigmoid(cpu_logits_1)

    frontend = cpu_model.frontend(cpu_features)

    print("=== Phase 7 CNN Baseline Smoke Test ===")
    print(f"sample_rate={sample_rate} samples={samples} batch_size={args.batch_size}")
    print(f"feature_shape={tuple(cpu_features.shape)}")
    print(f"frontend_shape={tuple(frontend.shape)}")
    print(f"logit_shape={tuple(cpu_logits_1.shape)}")
    print(f"parameter_count={count_parameters(cpu_model)}")
    print(f"raw_logit_range=({float(cpu_logits_1.min()):.6f}, {float(cpu_logits_1.max()):.6f})")
    print(f"external_sigmoid_score_range=({float(cpu_scores.min()):.6f}, {float(cpu_scores.max()):.6f})")

    if tuple(cpu_logits_1.shape) != (args.batch_size, num_classes):
        raise RuntimeError("Unexpected CNN output shape")
    if not torch.isfinite(cpu_logits_1).all():
        raise RuntimeError("CNN produced non-finite CPU logits")
    if not torch.equal(cpu_logits_1, cpu_logits_2):
        raise RuntimeError("CNN eval forward is not deterministic on CPU")
    if frontend.shape[-1] != cpu_features.shape[-1]:
        raise RuntimeError("Convolutional frontend unexpectedly changed the time axis")
    print("CPU forward/determinism: PASSED")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_extractor = feature_extractor_from_config(config).to(device)
        cuda_model, _ = build_model(config, "cnn", num_classes)
        cuda_model.load_state_dict(initial_state)
        cuda_model = cuda_model.to(device)
        cuda_extractor.eval()
        cuda_model.eval()
        with torch.no_grad():
            cuda_features = cuda_extractor(waveforms.to(device))
            cuda_logits = cuda_model(cuda_features).cpu()

        if tuple(cuda_logits.shape) != (args.batch_size, num_classes):
            raise RuntimeError("Unexpected CUDA CNN output shape")
        if not torch.isfinite(cuda_logits).all():
            raise RuntimeError("CNN produced non-finite CUDA logits")
        max_abs_diff = float((cpu_logits_1 - cuda_logits).abs().max())
        if max_abs_diff > 1e-3:
            raise RuntimeError(
                f"CPU/CUDA CNN logits differ unexpectedly (max_abs_diff={max_abs_diff:.6f})"
            )
        print(
            "CUDA forward/finite check: PASSED "
            f"(device={torch.cuda.get_device_name(0)}, max_abs_diff={max_abs_diff:.6f})"
        )
    else:
        print("CUDA forward/finite check: SKIPPED (CUDA unavailable)")

    print("CNN baseline smoke test: PASSED")
    print("No model training was performed and no checkpoint was written.")


if __name__ == "__main__":
    main()
