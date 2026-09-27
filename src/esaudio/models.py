from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Reduce only the frequency axis; preserve temporal resolution.
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvFrontend(nn.Module):
    def __init__(self, channels: list[int], dropout: float) -> None:
        super().__init__()
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, int(out_channels), dropout))
            in_channels = int(out_channels)
        self.blocks = nn.Sequential(*blocks)
        self.output_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CNNTagger(nn.Module):
    """Baseline CNN using the same convolutional frontend as the CRNN."""

    def __init__(self, num_classes: int, channels: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        self.frontend = ConvFrontend(channels, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.frontend.output_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.frontend(features)  # [B,C,F,T]
        x = x.mean(dim=(-2, -1))  # global frequency/time average
        x = self.dropout(x)
        return self.classifier(x)


class CRNNTagger(nn.Module):
    """Lightweight CNN + one-directional GRU/LSTM + temporal mean pooling."""

    def __init__(
        self,
        num_classes: int,
        channels: list[int],
        recurrent_type: str = "gru",
        recurrent_hidden: int = 64,
        recurrent_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.frontend = ConvFrontend(channels, dropout)
        recurrent_type = recurrent_type.lower()
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError("recurrent_type must be 'gru' or 'lstm'")
        rnn_cls = nn.GRU if recurrent_type == "gru" else nn.LSTM
        self.recurrent_type = recurrent_type
        self.rnn = rnn_cls(
            input_size=self.frontend.output_channels,
            hidden_size=int(recurrent_hidden),
            num_layers=int(recurrent_layers),
            batch_first=True,
            bidirectional=False,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(int(recurrent_hidden), num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.frontend(features)  # [B,C,F,T]
        x = x.mean(dim=2)  # frequency pooling -> [B,C,T]
        x = x.transpose(1, 2)  # [B,T,C]
        sequence, _ = self.rnn(x)
        x = sequence.mean(dim=1)  # temporal mean pooling
        x = self.dropout(x)
        return self.classifier(x)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_classes: int
    channels: list[int]
    dropout: float
    recurrent_type: str = "gru"
    recurrent_hidden: int = 64
    recurrent_layers: int = 1

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "channels": list(self.channels),
            "dropout": self.dropout,
            "recurrent_type": self.recurrent_type,
            "recurrent_hidden": self.recurrent_hidden,
            "recurrent_layers": self.recurrent_layers,
        }


def build_model(config: dict, model_name: str, num_classes: int) -> tuple[nn.Module, ModelSpec]:
    model_cfg = config["model"]
    channels = [int(x) for x in model_cfg["cnn_channels"]]
    dropout = float(model_cfg.get("dropout", 0.2))
    model_name = model_name.lower()

    if model_name == "cnn":
        spec = ModelSpec(
            name="cnn",
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
        )
        return CNNTagger(num_classes, channels, dropout), spec
    if model_name == "crnn":
        recurrent_type = str(model_cfg.get("recurrent_type", "gru"))
        recurrent_hidden = int(model_cfg.get("recurrent_hidden", 64))
        recurrent_layers = int(model_cfg.get("recurrent_layers", 1))
        spec = ModelSpec(
            name="crnn",
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
            recurrent_type=recurrent_type,
            recurrent_hidden=recurrent_hidden,
            recurrent_layers=recurrent_layers,
        )
        return (
            CRNNTagger(
                num_classes=num_classes,
                channels=channels,
                recurrent_type=recurrent_type,
                recurrent_hidden=recurrent_hidden,
                recurrent_layers=recurrent_layers,
                dropout=dropout,
            ),
            spec,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def model_from_spec(spec: dict) -> nn.Module:
    name = spec["name"]
    if name == "cnn":
        return CNNTagger(
            num_classes=int(spec["num_classes"]),
            channels=[int(x) for x in spec["channels"]],
            dropout=float(spec["dropout"]),
        )
    if name == "crnn":
        return CRNNTagger(
            num_classes=int(spec["num_classes"]),
            channels=[int(x) for x in spec["channels"]],
            recurrent_type=str(spec.get("recurrent_type", "gru")),
            recurrent_hidden=int(spec.get("recurrent_hidden", 64)),
            recurrent_layers=int(spec.get("recurrent_layers", 1)),
            dropout=float(spec["dropout"]),
        )
    raise ValueError(f"Unsupported model spec name: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
