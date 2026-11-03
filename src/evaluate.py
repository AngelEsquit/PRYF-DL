"""Ejecuta la evaluación técnica y de negocio del sistema."""

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic import (
    DEFAULT_VALUE_PARAMS,
    VALIDATION_SEED,
    DeepPhaseDetector,
    evaluate_baseline,
    evaluate_detector,
    make_eval_set,
    optimal_operating_point,
    sensitivity_analysis,
    value_curve,
)


def plot_value_curve(curve: dict, chosen: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for axis, (name, rows), label in zip(axes, curve.items(), ("Umbral de probabilidad", "Umbral del cociente STA/LTA")):
        axis.plot([r["threshold"] for r in rows], [r["business_value"] for r in rows], marker="o")
        axis.axvline(chosen[name], color="green", linestyle="--", label=f"Elegido en validación = {chosen[name]}")
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(title=name, xlabel=label, ylabel="Valor neto (USD)")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    detector = DeepPhaseDetector()
    validation = make_eval_set(seed=VALIDATION_SEED, n_events=60, n_noise=60)
    test = make_eval_set()

    # El umbral se elige por valor en validación y se reporta sobre prueba.
    val_curve = value_curve(detector, examples=validation)
    chosen = {name: optimal_operating_point(rows)["threshold"] for name, rows in val_curve.items()}
    test_curve = value_curve(detector, examples=test)
    optimal_on_test = {
        name: next(r for r in rows if r["threshold"] == chosen[name]) for name, rows in test_curve.items()
    }

    result = {
        "STA/LTA": evaluate_baseline(threshold=chosen["STA/LTA"]),
        "DeepPhase": evaluate_detector(detector, threshold=chosen["DeepPhase"]),
        "value_model": asdict(DEFAULT_VALUE_PARAMS),
        "operating_point_chosen_on_validation": chosen,
        "test_at_chosen_threshold": optimal_on_test,
        "value_curve_test": test_curve,
        "value_curve_validation": val_curve,
        "sensitivity": sensitivity_analysis(
            detector,
            validation=validation,
            test=test,
            deep_threshold=chosen["DeepPhase"],
            baseline_threshold=chosen["STA/LTA"],
        ),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_value_curve(test_curve, chosen, Path("results/value_curve.png"))
    print(json.dumps({k: result[k] for k in ("STA/LTA", "DeepPhase", "operating_point_chosen_on_validation")}, indent=2))
