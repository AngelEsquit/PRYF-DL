"""Ejecuta la evaluación técnica y de negocio del sistema.

Uso:
    python src/evaluate.py                      # datos sintéticos (por defecto)
    python src/evaluate.py --source stead --stead-dir RUTA [--fdsn]
"""

import argparse
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
    PyTorchPhaseDetector,
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


def run(detector, validation, test, *, suffix: str = "", fdsn_events=None) -> dict:
    """Elige el umbral por valor en validación y reporta en prueba."""
    val_curve = value_curve(detector, examples=validation)
    chosen = {name: optimal_operating_point(rows)["threshold"] for name, rows in val_curve.items()}
    test_curve = value_curve(detector, examples=test)
    optimal_on_test = {name: next(r for r in rows if r["threshold"] == chosen[name]) for name, rows in test_curve.items()}

    result = {
        "STA/LTA": evaluate_baseline(threshold=chosen["STA/LTA"], examples=test),
        "DeepPhase": evaluate_detector(detector, threshold=chosen["DeepPhase"], examples=test),
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
    if fdsn_events:
        # Eventos FDSN (llegadas teóricas) + ruido del conjunto de prueba de STEAD.
        noise = [e for e in test if not e.label][: len(fdsn_events)]
        local = fdsn_events + noise
        result["fdsn_validation"] = {
            "windows": len(local),
            "events": len(fdsn_events),
            "note": "Llegadas teóricas (TauP iasp91), no picks manuales.",
            "STA/LTA": evaluate_baseline(threshold=chosen["STA/LTA"], examples=local),
            "DeepPhase": evaluate_detector(detector, threshold=chosen["DeepPhase"], examples=local),
        }

    Path("results").mkdir(exist_ok=True)
    Path(f"results/metrics{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_value_curve(test_curve, chosen, Path(f"results/value_curve{suffix}.png"))
    keys = ("STA/LTA", "DeepPhase", "operating_point_chosen_on_validation", "fdsn_validation")
    print(json.dumps({k: result[k] for k in keys if k in result}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("synthetic", "stead"), default="synthetic")
    parser.add_argument("--stead-dir", help="Carpeta con merged.csv y merged.hdf5 (descarga manual de STEAD)")
    parser.add_argument("--stead-backend", choices=("hdf5", "seisbench"), default="hdf5")
    parser.add_argument("--n-events", type=int, default=3000, help="Eventos STEAD totales (train+dev+test)")
    parser.add_argument("--n-noise", type=int, default=3000, help="Ventanas de ruido STEAD totales")
    parser.add_argument("--cache", default="data/stead_subset.npz", help="Caché de ventanas procesadas")
    parser.add_argument("--fdsn", action="store_true", help="Agregar validación local con FDSN/IRIS")
    parser.add_argument("--fdsn-cache", default="data/fdsn_events.npz")
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta a checkpoint PyTorch entrenado (.pt)")
    args = parser.parse_args()

    # Selección del detector: PyTorch si hay checkpoint, o clásico DeepPhase por defecto
    if args.checkpoint:
        detector = PyTorchPhaseDetector(checkpoint_path=args.checkpoint)
        print(f"[*] Evaluando modelo PyTorch ({detector.model_name}) cargado desde: {args.checkpoint}")
    else:
        detector = None

    if args.source == "synthetic":
        if args.fdsn:
            parser.error("--fdsn requiere --source stead")
        run(
            detector or DeepPhaseDetector(),
            make_eval_set(seed=VALIDATION_SEED, n_events=60, n_noise=60),
            make_eval_set(),
        )
        return

    from fdsn import fetch_fdsn_examples
    from stead import SteadConfig, load_stead

    splits = load_stead(
        args.stead_dir,
        backend=args.stead_backend,
        config=SteadConfig(n_events=args.n_events, n_noise=args.n_noise),
        cache=args.cache,
    )
    for name, examples in splits.items():
        print(f"{name}: {sum(e.label for e in examples)} eventos, {sum(1 - e.label for e in examples)} ruido")
    if detector is None:
        detector = DeepPhaseDetector(train_examples=splits["train"])
    fdsn_events = fetch_fdsn_examples(cache=args.fdsn_cache) if args.fdsn else None
    run(detector, splits["dev"], splits["test"], suffix="_stead", fdsn_events=fdsn_events)


if __name__ == "__main__":
    main()
