"""Pipeline reproducible de detección sísmica y estimación de valor."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class SignalExample:
    signal: np.ndarray
    p_arrival: int | None
    s_arrival: int | None
    label: int


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


def make_training_data(
    seed: int = 23361, n_events: int = 100, n_noise: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for event in [True] * n_events + [False] * n_noise:
        example = generate_example(rng, event=event)
        features = _features(example.signal)
        if event:
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


class DeepPhaseDetector:
    """Detector MLP de dos capas, inspirado en un picker de fases."""

    def __init__(self, seed: int = 23361) -> None:
        self.seed = seed
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
        x, y = make_training_data(self.seed)
        self.model.fit(x, y)

    def detect(self, signal: np.ndarray, threshold: float = 0.65) -> Detection:
        started = time.perf_counter()
        probabilities = self.model.predict_proba(_features(signal))[:, 1]
        # The first second is excluded because zero-phase filtering creates edge
        # transients that are not physically observable arrivals.
        candidates = np.flatnonzero(probabilities >= threshold)
        candidates = candidates[candidates >= 100]
        sample = int(candidates[0]) if len(candidates) else None
        return Detection(
            sample is not None,
            sample,
            float(probabilities.max()),
            round((time.perf_counter() - started) * 1000, 3),
        )


def sta_lta_detect(signal: np.ndarray, sample_rate: float = 100.0, threshold: float = 3.0) -> Detection:
    """Baseline clásico STA/LTA con primera excedencia del cociente."""
    started = time.perf_counter()
    energy = np.sqrt(np.mean(signal**2, axis=1)) + 1e-8
    short = max(2, int(0.25 * sample_rate))
    long = max(short + 1, int(3.0 * sample_rate))
    sta = np.convolve(energy, np.ones(short) / short, mode="same")
    lta = np.convolve(energy, np.ones(long) / long, mode="same") + 1e-8
    ratio = sta / lta
    candidates = np.flatnonzero(ratio > threshold)
    sample = int(candidates[0]) if len(candidates) else None
    return Detection(
        sample is not None,
        sample,
        float(ratio.max() / threshold),
        round((time.perf_counter() - started) * 1000, 3),
    )


def business_value(
    tp: int,
    fp: int,
    *,
    seconds_advance: float = 4.0,
    value_per_second: float = 1000.0,
    false_alarm_cost: float = 2500.0,
) -> float:
    """Valor neto de operar el detector en una ventana de decisión."""
    return tp * seconds_advance * value_per_second - fp * false_alarm_cost


def evaluate_detector(
    detector: DeepPhaseDetector | None = None,
    *,
    seed: int = 2026,
    n_events: int = 30,
    n_noise: int = 30,
    threshold: float = 0.65,
) -> dict[str, float | int]:
    detector = detector or DeepPhaseDetector()
    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = {"tp": [], "fp": [], "error": [], "latency": []}
    for event in [True] * n_events + [False] * n_noise:
        example = generate_example(rng, event=event)
        detection = detector.detect(example.signal, threshold=threshold)
        if event:
            results["tp"].append(float(detection.detected))
            if detection.detected and example.p_arrival is not None:
                results["error"].append(abs(detection.sample - example.p_arrival) / 100)
        else:
            results["fp"].append(float(detection.detected))
        results["latency"].append(detection.latency_ms)
    tp, fp = sum(results["tp"]), sum(results["fp"])
    fn = n_events - tp
    return {
        "events": n_events,
        "noise_windows": n_noise,
        "precision": round(tp / (tp + fp) if tp + fp else 0, 4),
        "recall": round(tp / n_events, 4),
        "f1": round(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, 4),
        "mean_picking_error_s": round(float(np.mean(results["error"])) if results["error"] else 99, 4),
        "mean_latency_ms": round(float(np.mean(results["latency"])), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "business_value": round(business_value(int(tp), int(fp)), 2),
    }


def evaluate_baseline(*, seed: int = 2026, n_events: int = 30, n_noise: int = 30) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    tp = fp = 0
    errors: list[float] = []
    for event in [True] * n_events + [False] * n_noise:
        example = generate_example(rng, event=event)
        detection = sta_lta_detect(example.signal)
        if event:
            tp += int(detection.detected)
            if detection.detected and example.p_arrival is not None:
                errors.append(abs(detection.sample - example.p_arrival) / 100)
        else:
            fp += int(detection.detected)
    fn = n_events - tp
    return {
        "precision": round(tp / (tp + fp) if tp + fp else 0, 4),
        "recall": round(tp / n_events, 4),
        "f1": round(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, 4),
        "mean_picking_error_s": round(float(np.mean(errors)) if errors else 99, 4),
        "business_value": round(business_value(tp, fp), 2),
        "true_positives": tp,
        "false_positives": fp,
    }
