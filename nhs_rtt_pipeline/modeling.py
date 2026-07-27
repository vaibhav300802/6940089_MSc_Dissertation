from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (int(kernel_size) - 1) * int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TCNQuantileRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_trusts: int,
        n_specialties: int,
        prediction_length: int,
        quantiles: Sequence[float],
        hidden_channels: int,
        tcn_levels: int,
        kernel_size: int,
        dropout: float,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.prediction_length = int(prediction_length)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.trust_embedding = nn.Embedding(int(n_trusts), int(embedding_dim))
        self.specialty_embedding = nn.Embedding(int(n_specialties), int(embedding_dim))

        input_channels = int(n_features) + 2 * int(embedding_dim)
        blocks = []
        for level in range(int(tcn_levels)):
            in_channels = input_channels if level == 0 else int(hidden_channels)
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=int(hidden_channels),
                    kernel_size=int(kernel_size),
                    dilation=2 ** level,
                    dropout=float(dropout),
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(int(hidden_channels), int(hidden_channels)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_channels), self.prediction_length * len(self.quantiles)),
        )

    def forward(self, x: torch.Tensor, trust_idx: torch.Tensor, specialty_idx: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        trust_emb = self.trust_embedding(trust_idx).unsqueeze(1).expand(batch_size, sequence_length, -1)
        specialty_emb = self.specialty_embedding(specialty_idx).unsqueeze(1).expand(batch_size, sequence_length, -1)
        encoded_input = torch.cat([x, trust_emb, specialty_emb], dim=-1).transpose(1, 2)
        encoded = self.tcn(encoded_input)[:, :, -1]
        output = self.head(encoded)
        return output.view(batch_size, self.prediction_length, len(self.quantiles))


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32).view(1, 1, -1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target.unsqueeze(-1) - prediction
        return torch.maximum((self.quantiles - 1.0) * errors, self.quantiles * errors).mean()


def build_tcn_model_config(
    *,
    n_features: int,
    n_trusts: int,
    n_specialties: int,
    prediction_length: int,
    quantiles: Sequence[float],
    hidden_channels: int,
    tcn_levels: int,
    kernel_size: int,
    dropout: float,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "model_class": "TCNQuantileRegressor",
        "format": "manual_pytorch_tcn_state_dict",
        "n_features": int(n_features),
        "n_trusts": int(n_trusts),
        "n_specialties": int(n_specialties),
        "prediction_length": int(prediction_length),
        "quantiles": [float(q) for q in quantiles],
        "hidden_channels": int(hidden_channels),
        "tcn_levels": int(tcn_levels),
        "kernel_size": int(kernel_size),
        "dropout": float(dropout),
        "embedding_dim": int(embedding_dim),
    }


def model_from_config(model_config: Mapping[str, Any]) -> TCNQuantileRegressor:
    if model_config.get("model_class") != "TCNQuantileRegressor":
        raise ValueError(f"Unsupported model class: {model_config.get('model_class')}")
    return TCNQuantileRegressor(
        n_features=int(model_config["n_features"]),
        n_trusts=int(model_config["n_trusts"]),
        n_specialties=int(model_config["n_specialties"]),
        prediction_length=int(model_config["prediction_length"]),
        quantiles=model_config["quantiles"],
        hidden_channels=int(model_config["hidden_channels"]),
        tcn_levels=int(model_config["tcn_levels"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"]),
        embedding_dim=int(model_config["embedding_dim"]),
    )


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tcn_from_artifacts(
    state_dict_path: str | Path,
    model_config_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[TCNQuantileRegressor, dict[str, Any]]:
    state_path = Path(state_dict_path)
    config_path = Path(model_config_path)
    if not state_path.exists():
        raise FileNotFoundError(f"Missing TCN state dict: {state_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing TCN model config: {config_path}")
    model_config = load_json(config_path)
    model = model_from_config(model_config)
    try:
        state_dict = torch.load(state_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(state_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        raise ValueError(
            f"{state_path} is a legacy full artifact. Expected the canonical state dict file "
            "models/tcn_state_dict.pt. Re-run Layer 1 with the standardised pipeline."
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_config
