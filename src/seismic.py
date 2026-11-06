"""Pipeline reproducible de detección sísmica y estimación de valor."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.neural_network import MLPClassifier

# Ambos detectores ignoran el primer segundo: el filtro de fase cero y la
# ventana de aprendizaje de STA/LTA producen transitorios que no son llegadas.
WARMUP_SAMPLES = 100


@dataclass(frozen=True)
class SignalExample:
    signal: np.ndarray
    p_arrival: int | None
    s_arrival: int | None
    label: int
    meta: dict = field(default_factory=dict, compare=False)  # origen, SNR, estación, etc.


@dataclass(frozen=True)
class Detection:
    detected: bool
    sample: int | None
    probability: float
    latency_ms: float


def _bandpass(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    low, high = 1.0 / (sample_rate / 2), 18.0 / (sample_rate / 2)
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def generate_example(
    rng: np.random.Generator,
    *,
    sample_rate: float = 100.0,
    duration: float = 12.0,
    event: bool = True,
    noise_scale: float | None = None,
) -> SignalExample:
    """Genera una ventana de tres componentes con una llegada P y S controlables."""
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    noise_scale = float(noise_scale or rng.uniform(0.18, 0.42))
    signal = rng.normal(0, noise_scale, (n, 3)).astype(np.float32)
    if not event:
        return SignalExample(_bandpass(signal, sample_rate), None, None, 0)
    p = int(rng.uniform(2.0, 5.0) * sample_rate)
    s = p + int(rng.uniform(1.0, 2.8) * sample_rate)
    p_wave = np.exp(-((t - p / sample_rate) / 0.22) ** 2) * rng.uniform(0.7, 1.4)
    s_wave = np.exp(-((t - s / sample_rate) / 0.65) ** 2) * rng.uniform(1.1, 2.2)
    frequencies = rng.uniform(3.0, 10.0, 3)
    for component in range(3):
        signal[:, component] += p_wave * np.sin(2 * np.pi * frequencies[component] * t)
        signal[:, component] += s_wave * np.sin(2 * np.pi * (frequencies[component] / 2) * t)
    return SignalExample(_bandpass(signal, sample_rate), p, s, 1)


def _features(signal: np.ndarray, window: int = 20) -> np.ndarray:
    """Calcula energía, envolvente y estadísticos por ventana para el detector."""
    energy = np.sqrt(np.mean(signal**2, axis=1))
    envelope = np.abs(signal).mean(axis=1)
    rolling = np.convolve(energy, np.ones(window) / window, mode="same")
    gradient = np.gradient(rolling)
    return np.column_stack((energy, envelope, rolling, gradient))


def training_arrays(examples: list[SignalExample]) -> tuple[np.ndarray, np.ndarray]:
    """Convierte ventanas etiquetadas en filas (características, etiqueta P/no P)."""
    rows, labels = [], []
    for example in examples:
        features = _features(example.signal)
        if example.label:
            start = max(0, example.p_arrival - 25)
            end = min(len(features), example.p_arrival + 25)
            rows.extend(features[start:end])
            labels.extend([1] * (end - start))
            outside = np.concatenate((features[: max(0, example.p_arrival - 80)], features[example.p_arrival + 80 :]))
            rows.extend(outside[::8])
            labels.extend([0] * len(outside[::8]))
        else:
            rows.extend(features[::8])
            labels.extend([0] * len(features[::8]))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def make_training_data(
    seed: int = 23361, n_events: int = 100, n_noise: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    examples = [generate_example(rng, event=event) for event in [True] * n_events + [False] * n_noise]
    return training_arrays(examples)


class DeepPhaseDetector:
    """Detector MLP de dos capas, inspirado en un picker de fases."""

    def __init__(self, seed: int = 23361, train_examples: list[SignalExample] | None = None) -> None:
        """Entrena con ``train_examples`` (p. ej. STEAD) o, si no hay, con señales sintéticas."""
        self.seed = seed
        self.train_examples = train_examples
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=160,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self._fit()

    def _fit(self) -> None:
        if self.train_examples is not None:
            x, y = training_arrays(self.train_examples)
        else:
            x, y = make_training_data(self.seed)
        self.model.fit(x, y)

    def detect(self, signal: np.ndarray, threshold: float = 0.65) -> Detection:
        started = time.perf_counter()
        probabilities = self.model.predict_proba(_features(signal))[:, 1]
        # The first second is excluded because zero-phase filtering creates edge
        # transients that are not physically observable arrivals.
        candidates = np.flatnonzero(probabilities >= threshold)
        candidates = candidates[candidates >= WARMUP_SAMPLES]
        sample = int(candidates[0]) if len(candidates) else None
        return Detection(
            sample is not None,
            sample,
            float(probabilities.max()),
            round((time.perf_counter() - started) * 1000, 3),
        )


class PyTorchPhaseDetector:
    """Detector de Deep Learning basado en PyTorch (soporta CUDA / GPU RTX 5050 y CPU).

    Permite cargar checkpoints entrenados con `src/train.py` (ej. `best_model.pt`)
    y realizar inferencia optimizada con baja latencia.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model_name: str = "conv1d",
        device: str = "auto",
    ) -> None:
        from pathlib import Path
        import torch
        try:
            from .models import build_model
        except ImportError:
            from models import build_model

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name.lower()
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model_name = ckpt.get("model_name") or ckpt.get("config", {}).get("model", self.model_name)
            kwargs = ckpt.get("model_kwargs", {})
            self.model = build_model(self.model_name, **kwargs).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model = build_model(self.model_name).to(self.device)

        self.model.eval()

    def detect(self, signal: np.ndarray, threshold: float = 0.50) -> Detection:
        import torch

        started = time.perf_counter()
        with torch.no_grad():
            if self.model_name in ("conv1d", "phasenet", "convphasenet", "convphasenet1d"):
                # Input: (1, 3, T)
                x = (
                    torch.from_numpy(np.ascontiguousarray(signal.T, dtype=np.float32))
                    .unsqueeze(0)
                    .to(self.device)
                )
            else:
                # Input: (1, T, 4)
                feat = _features(signal).astype(np.float32)
                x = torch.from_numpy(feat).unsqueeze(0).to(self.device)

            logits = self.model(x)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        candidates = np.flatnonzero(probabilities >= threshold)
        candidates = candidates[candidates >= WARMUP_SAMPLES]
        sample = int(candidates[0]) if len(candidates) else None
        max_prob = float(probabilities.max()) if len(probabilities) else 0.0
        return Detection(
            sample is not None,
            sample,
            max_prob,
            round((time.perf_counter() - started) * 1000, 3),
        )


def _trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Media móvil causal (solo muestras pasadas); al inicio usa la ventana disponible."""
    cumulative = np.concatenate(([0.0], np.cumsum(x)))
    end = np.arange(1, len(x) + 1)
    start = np.maximum(0, end - n)
    return (cumulative[end] - cumulative[start]) / (end - start)


def sta_lta_detect(signal: np.ndarray, sample_rate: float = 100.0, threshold: float = 3.0) -> Detection:
    """Baseline clásico STA/LTA causal con primera excedencia del cociente.

    Usa ventanas hacia atrás (STA 0.25 s, LTA 3 s) y la misma exclusión inicial
    que DeepPhase, para que la comparación sea justa.
    """
    started = time.perf_counter()
    energy = np.sqrt(np.mean(signal**2, axis=1)) + 1e-8
    short = max(2, int(0.25 * sample_rate))
    long = max(short + 1, int(3.0 * sample_rate))
    ratio = _trailing_mean(energy, short) / (_trailing_mean(energy, long) + 1e-8)
    ratio[:WARMUP_SAMPLES] = 0.0
    candidates = np.flatnonzero(ratio > threshold)
    sample = int(candidates[0]) if len(candidates) else None
    return Detection(
        sample is not None,
        sample,
        float(ratio.max() / threshold),
        round((time.perf_counter() - started) * 1000, 3),
    )


# ---------------------------------------------------------------------------
# Modelo de valor
# ---------------------------------------------------------------------------

SAMPLE_RATE = 100.0
PICK_TOLERANCE_S = 0.5  # una alerta antes de P - tolerancia es un disparo prematuro
VALIDATION_SEED = 777  # distinto de la semilla de entrenamiento (23361) y de prueba (2026)
MEAN_S_MINUS_P_S = 1.9  # media de U(1.0, 2.8) s del generador sintético


@dataclass(frozen=True)
class ValueParams:
    """Supuestos económicos (USD) del modelo de valor. Todos son estimaciones.

    value_per_second: pérdida evitada por cada segundo de aviso útil antes de la
        onda S (paro de maquinaria, cierre de gas, aviso al personal).
    false_alarm_cost: costo de un paro innecesario provocado por una falsa alarma.
    missed_event_cost: pérdida adicional de un evento no detectado (sin aviso).
    operating_cost_per_window: costo de cómputo/operación por ventana evaluada.
    """

    value_per_second: float = 1000.0
    false_alarm_cost: float = 2500.0
    missed_event_cost: float = 10000.0
    operating_cost_per_window: float = 1.0


DEFAULT_VALUE_PARAMS = ValueParams()


def business_value(
    tp: int,
    fp: int,
    fn: int = 0,
    *,
    n_windows: int = 0,
    seconds_advance: float = MEAN_S_MINUS_P_S,
    params: ValueParams = DEFAULT_VALUE_PARAMS,
) -> float:
    """Valor neto = TP*anticipación*$/s - FP*costo_FA - FN*costo_FN - ventanas*costo_op."""
    return (
        tp * seconds_advance * params.value_per_second
        - fp * params.false_alarm_cost
        - fn * params.missed_event_cost
        - n_windows * params.operating_cost_per_window
    )


def make_eval_set(*, seed: int = 2026, n_events: int = 30, n_noise: int = 30) -> list[SignalExample]:
    """Conjunto de prueba reproducible, independiente del de entrenamiento."""
    rng = np.random.default_rng(seed)
    return [generate_example(rng, event=event) for event in [True] * n_events + [False] * n_noise]


def _outcome(detect: Callable[[np.ndarray], Detection], examples: list[SignalExample]) -> dict[str, float | None]:
    """Cuenta TP/FP/FN y acumula error de picking, anticipación y latencia."""
    tp = fp = fn = 0
    errors: list[float] = []
    advances: list[float] = []
    latencies: list[float] = []
    for example in examples:
        detection = detect(example.signal)
        latencies.append(detection.latency_ms)
        if example.label:
            if detection.detected:
                errors.append(abs(detection.sample - example.p_arrival) / SAMPLE_RATE)
            # Solo es un acierto una alerta con aviso útil: no prematura (antes de
            # P - tolerancia) y antes de la llegada de S. Si no, el evento se
            # considera no detectado, aunque haya sonado una alerta.
            earliest = example.p_arrival - PICK_TOLERANCE_S * SAMPLE_RATE
            if detection.detected and earliest <= detection.sample < example.s_arrival:
                tp += 1
                # La anticipación útil no puede superar S-P.
                advance = (example.s_arrival - detection.sample) / SAMPLE_RATE
                max_advance = (example.s_arrival - example.p_arrival) / SAMPLE_RATE
                advances.append(float(min(advance, max_advance)))
            else:
                fn += 1
        elif detection.detected:
            fp += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_windows": len(examples),
        "mean_error": float(np.mean(errors)) if errors else None,
        "mean_advance": float(np.mean(advances)) if advances else 0.0,
        "mean_latency": float(np.mean(latencies)),
    }


def _value(outcome: dict[str, float], params: ValueParams) -> float:
    return business_value(
        int(outcome["tp"]),
        int(outcome["fp"]),
        int(outcome["fn"]),
        n_windows=int(outcome["n_windows"]),
        seconds_advance=outcome["mean_advance"],
        params=params,
    )


def _summary(outcome: dict[str, float], params: ValueParams) -> dict[str, float | int]:
    tp, fp, fn = int(outcome["tp"]), int(outcome["fp"]), int(outcome["fn"])
    return {
        "precision": round(tp / (tp + fp) if tp + fp else 0, 4),
        "recall": round(tp / (tp + fn) if tp + fn else 0, 4),
        "f1": round(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, 4),
        "mean_picking_error_s": None if outcome["mean_error"] is None else round(outcome["mean_error"], 4),
        "mean_advance_s": round(outcome["mean_advance"], 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "business_value": round(_value(outcome, params), 2),
    }


def evaluate_detector(
    detector: DeepPhaseDetector | None = None,
    *,
    seed: int = 2026,
    n_events: int = 30,
    n_noise: int = 30,
    threshold: float = 0.65,
    params: ValueParams = DEFAULT_VALUE_PARAMS,
    examples: list[SignalExample] | None = None,
) -> dict[str, float | int]:
    detector = detector or DeepPhaseDetector()
    examples = examples or make_eval_set(seed=seed, n_events=n_events, n_noise=n_noise)
    outcome = _outcome(lambda s: detector.detect(s, threshold=threshold), examples)
    result = _summary(outcome, params)
    events = sum(e.label for e in examples)
    result.update(events=events, noise_windows=len(examples) - events, mean_latency_ms=round(outcome["mean_latency"], 4))
    return result


def evaluate_baseline(
    *,
    seed: int = 2026,
    n_events: int = 30,
    n_noise: int = 30,
    threshold: float = 3.0,
    params: ValueParams = DEFAULT_VALUE_PARAMS,
    examples: list[SignalExample] | None = None,
) -> dict[str, float | int]:
    examples = examples or make_eval_set(seed=seed, n_events=n_events, n_noise=n_noise)
    outcome = _outcome(lambda s: sta_lta_detect(s, threshold=threshold), examples)
    return _summary(outcome, params)


DEEP_THRESHOLDS = tuple(round(0.40 + 0.05 * i, 2) for i in range(11))  # 0.40 ... 0.90
STALTA_THRESHOLDS = tuple(round(1.5 + 0.5 * i, 1) for i in range(10))  # 1.5 ... 6.0


def _sweep(
    make_detect: Callable[[float], Callable[[np.ndarray], Detection]],
    thresholds: tuple[float, ...],
    examples: list[SignalExample],
) -> list[dict[str, float]]:
    return [{"threshold": t, **_outcome(make_detect(t), examples)} for t in thresholds]


def _deep_sweep(detector: DeepPhaseDetector, examples: list[SignalExample]) -> list[dict[str, float]]:
    return _sweep(lambda t: (lambda s: detector.detect(s, threshold=t)), DEEP_THRESHOLDS, examples)


def value_curve(
    detector: DeepPhaseDetector | None = None,
    *,
    examples: list[SignalExample] | None = None,
    params: ValueParams = DEFAULT_VALUE_PARAMS,
) -> dict[str, list[dict[str, float | int]]]:
    """Valor neto y métricas técnicas al variar el umbral de decisión."""
    detector = detector or DeepPhaseDetector()
    examples = examples or make_eval_set()
    sweeps = {
        "DeepPhase": _deep_sweep(detector, examples),
        "STA/LTA": _sweep(lambda t: (lambda s: sta_lta_detect(s, threshold=t)), STALTA_THRESHOLDS, examples),
    }
    return {
        name: [{"threshold": row["threshold"], **_summary(row, params)} for row in rows]
        for name, rows in sweeps.items()
    }


def optimal_operating_point(curve: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Umbral con mayor valor neto; en empate se prefiere el umbral más alto."""
    best = curve[0]
    for row in curve[1:]:
        if row["business_value"] >= best["business_value"]:
            best = row
    return best


SENSITIVITY_FACTORS = (0.5, 0.8, 1.2, 1.5)  # -50 %, -20 %, +20 %, +50 %
SENSITIVITY_FIELDS = ("value_per_second", "false_alarm_cost", "missed_event_cost")


def sensitivity_analysis(
    detector: DeepPhaseDetector | None = None,
    *,
    validation: list[SignalExample] | None = None,
    test: list[SignalExample] | None = None,
    base: ValueParams = DEFAULT_VALUE_PARAMS,
    deep_threshold: float = 0.65,
    baseline_threshold: float = 3.0,
) -> list[dict[str, float | str]]:
    """Cambia cada supuesto economico +-20 % y +-50 % y mide el efecto en la decision.

    El umbral optimo de cada escenario se elige en ``validation`` y el valor se
    reporta en ``test``, para no optimizar sobre los datos de prueba.
    """
    detector = detector or DeepPhaseDetector()
    validation = validation or make_eval_set(seed=VALIDATION_SEED, n_events=60, n_noise=60)
    test = test or make_eval_set()
    val_rows = _deep_sweep(detector, validation)
    test_rows = _deep_sweep(detector, test)
    baseline_row = _sweep(lambda t: (lambda s: sta_lta_detect(s, threshold=t)), (baseline_threshold,), test)[0]
    test_at_default = next(r for r in test_rows if r["threshold"] == deep_threshold)

    rows: list[dict[str, float | str]] = []
    for field in SENSITIVITY_FIELDS:
        for factor in SENSITIVITY_FACTORS:
            params = replace(base, **{field: getattr(base, field) * factor})
            chosen = max(val_rows, key=lambda r: (round(_value(r, params), 2), r["threshold"]))
            test_at_chosen = next(r for r in test_rows if r["threshold"] == chosen["threshold"])
            deep_value = _value(test_at_default, params)
            baseline_value = _value(baseline_row, params)
            rows.append(
                {
                    "parameter": field,
                    "change_pct": round((factor - 1) * 100),
                    "deep_value_at_default_threshold": round(deep_value, 2),
                    "optimal_threshold": chosen["threshold"],
                    "deep_value_at_optimal_threshold": round(_value(test_at_chosen, params), 2),
                    "baseline_value": round(baseline_value, 2),
                    "advantage_vs_baseline": round(deep_value - baseline_value, 2),
                }
            )
    return rows
