"""Pruebas unitarias del sistema de entrenamiento, modelos PyTorch y checkpoints."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models import ConvPhaseNet1D, DeepMLP, build_model
from src.seismic import PyTorchPhaseDetector, generate_example
from src.train import (
    SeismicWindowDataset,
    evaluate_model_on_dataset,
    save_checkpoint,
)


def test_models_forward_shapes():
    conv = ConvPhaseNet1D(in_channels=3, base_filters=16, num_blocks=2)
    x_conv = torch.randn(2, 3, 1200)
    out_conv = conv(x_conv)
    assert out_conv.shape == (2, 1200)

    mlp = DeepMLP(in_features=4, hidden_sizes=(16, 8))
    x_mlp = torch.randn(2, 1200, 4)
    out_mlp = mlp(x_mlp)
    assert out_mlp.shape == (2, 1200)


def test_build_model_factory():
    m1 = build_model("conv1d")
    assert isinstance(m1, ConvPhaseNet1D)

    m2 = build_model("mlp")
    assert isinstance(m2, DeepMLP)

    with pytest.raises(ValueError):
        build_model("unknown_model_arch")


def test_dataset_item_generation():
    rng = np.random.default_rng(42)
    example = generate_example(rng, event=True)
    ds = SeismicWindowDataset([example], model_type="conv1d")
    assert len(ds) == 1
    item = ds[0]
    assert item["x"].shape == (3, 1200)
    assert item["target"].shape == (1200,)
    assert item["label"].item() == 1.0


def test_checkpoint_atomic_saving_and_resuming(tmp_path):
    ckpt_file = tmp_path / "test_checkpoint.pt"
    model = build_model("conv1d", base_filters=8, num_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_checkpoint(
        ckpt_file,
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        best_metric=1234.5,
        best_epoch=3,
        history=[{"epoch": 1}, {"epoch": 2}],
        args_dict={"model": "conv1d", "lr": 0.001},
    )

    assert ckpt_file.exists()
    assert not (tmp_path / "test_checkpoint.tmp").exists()  # comprobación atómica

    # Cargar y verificar restauración de estado
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 5
    assert ckpt["best_metric"] == 1234.5
    assert ckpt["best_epoch"] == 3
    assert len(ckpt["history"]) == 2
    assert ckpt["config"]["model"] == "conv1d"


def test_pytorch_phase_detector_inference(tmp_path):
    ckpt_file = tmp_path / "mock_model.pt"
    model = build_model("conv1d", base_filters=8, num_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_checkpoint(
        ckpt_file,
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        best_metric=100.0,
        best_epoch=1,
        history=[],
        args_dict={"model": "conv1d"},
    )

    detector = PyTorchPhaseDetector(checkpoint_path=ckpt_file, device="cpu")
    rng = np.random.default_rng(99)
    example = generate_example(rng, event=True)

    detection = detector.detect(example.signal, threshold=0.5)
    assert isinstance(detection.detected, bool)
    assert 0.0 <= detection.probability <= 1.0
    assert detection.latency_ms >= 0.0
