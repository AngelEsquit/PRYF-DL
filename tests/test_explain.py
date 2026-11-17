"""Pruebas rápidas del módulo de explicabilidad (sin datos reales ni checkpoints)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("captum")

from explain import causal_check, classify, decision_sample, integrated_gradients, window_scores  # noqa: E402
from models import build_model  # noqa: E402
from seismic import SignalExample  # noqa: E402


def _model():
    torch.manual_seed(0)
    return build_model("conv1d", base_filters=8, num_blocks=2).eval()


def test_integrated_gradients_is_complete_and_reproducible():
    model, x = _model(), torch.randn(2, 3, 400)
    targets = np.array([150, 220])
    a1, delta = integrated_gradients(model, x, targets, n_steps=128)
    a2, _ = integrated_gradients(model, x, targets, n_steps=128)
    assert a1.shape == (2, 3, 400)
    assert np.array_equal(a1, a2)
    with torch.no_grad():  # completitud: sum(IG) = f(x) - f(baseline)
        gap = model(x).numpy()[[0, 1], targets] - model(torch.zeros_like(x)).numpy()[[0, 1], targets]
    assert np.allclose(a1.sum(axis=(1, 2)), gap, atol=0.1 * np.abs(gap).max() + 1e-3)
    assert delta.max() < 0.2


def test_decision_sample_and_classification():
    logits = np.full((2, 500), -5.0)
    logits[0, 250:] = 5.0  # alerta en 250
    t_star, alerted = decision_sample(logits, 0.5)
    assert t_star[0] == 250 and alerted[0] and not alerted[1]
    event = SignalExample(np.zeros((500, 3), np.float32), 240, 400, 1)
    noise = SignalExample(np.zeros((500, 3), np.float32), None, None, 0)
    assert classify(event, True, 250) == "TP"
    assert classify(event, True, 100) == "FN"  # prematura
    assert classify(event, False, 0) == "FN"
    assert classify(noise, True, 250) == "FP" and classify(noise, False, 0) == "TN"


def test_window_scores_and_causal_check_shapes():
    assert window_scores(np.ones((2, 3, 400))).shape == (2, 8)
    out = causal_check(_model(), torch.randn(2, 3, 400), np.array([150, 220]), 0.5)
    assert all(0.0 <= v <= 1.0 for v in out.values())
