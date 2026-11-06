"""Modelos de Deep Learning en PyTorch para detección de fase P sísmica.

Diseñados para CC3092 Deep Learning (UVG) y optimizados para NVIDIA RTX (CUDA).
Incluye dos arquitecturas comparables:
1. ConvPhaseNet1D: Red Convolucional 1D profunda con bloques residuales
   y convoluciones dilatadas sobre los 3 canales de señal (Z, N, E).
2. DeepMLP: Red neuronal densa profunda en PyTorch con regularización
   (BatchNorm + Dropout) sobre características temporales extraídas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Bloque residual 1D con convoluciones dilatadas y normalización."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ConvPhaseNet1D(nn.Module):
    """Detector convolucional 1D profundo para señales sísmicas de 3 componentes.

    Recibe directamente las señales Z, N, E de dimensión (B, 3, T) y predice
    la probabilidad P(t) de llegada de la onda P en cada muestra temporal t.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.init_kwargs = {
            "in_channels": in_channels,
            "base_filters": base_filters,
            "num_blocks": num_blocks,
            "dropout": dropout,
        }
        self.in_channels = in_channels
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
        )

        dilations = [1, 2, 4, 8]
        blocks = []
        for i in range(num_blocks):
            d = dilations[i % len(dilations)]
            blocks.append(ResidualBlock1D(base_filters, dilation=d, dropout=dropout))
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv1d(base_filters, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forma de entrada: (B, 3, T). Salida: (B, T) logits o probabilidades."""
        h = self.entry(x)
        h = self.res_blocks(h)
        logits = self.head(h).squeeze(1)  # (B, T)
        return logits


class DeepMLP(nn.Module):
    """MLP profundo en PyTorch sobre vectores de características temporales.

    Equivalente y escalable en GPU al MLPClassifier de sklearn, pero con
    BatchNorm, Dropout y ejecución vectorizada en CUDA.
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden_sizes: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.init_kwargs = {
            "in_features": in_features,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
        }
        layers = []
        prev = in_features
        for size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev, size),
                    nn.BatchNorm1d(size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soporta tanto (B*T, in_features) como (B, T, in_features)."""
        if x.dim() == 3:
            b, t, c = x.shape
            flat = x.view(b * t, c)
            out = self.net(flat).view(b, t)
            return out
        return self.net(x).squeeze(-1)


def build_model(model_name: str, **kwargs) -> nn.Module:
    """Fábrica de modelos según nombre."""
    name = model_name.lower().strip()
    if name in ("conv1d", "phasenet", "convphasenet", "convphasenet1d"):
        model = ConvPhaseNet1D(**kwargs)
        model.model_name = "conv1d"
        return model
    elif name in ("mlp", "deepmlp", "deepphase"):
        model = DeepMLP(**kwargs)
        model.model_name = "mlp"
        return model
    else:
        raise ValueError(
            f"Modelo desconocido '{model_name}'. Opciones: 'conv1d', 'mlp'."
        )
