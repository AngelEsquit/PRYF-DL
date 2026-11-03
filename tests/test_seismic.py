import numpy as np

from src.seismic import (
    DEEP_THRESHOLDS,
    SENSITIVITY_FIELDS,
    SENSITIVITY_FACTORS,
    VALIDATION_SEED,
    DeepPhaseDetector,
    Detection,
    ValueParams,
    _outcome,
    business_value,
    generate_example,
    make_eval_set,
    optimal_operating_point,
    sensitivity_analysis,
    sta_lta_detect,
    value_curve,
)


def test_signal_has_three_components_and_labels():
    example = generate_example(np.random.default_rng(1), event=True)
    assert example.signal.shape == (1200, 3)
    assert example.p_arrival is not None
    assert example.s_arrival is not None


def test_noise_has_no_ground_truth_arrivals():
    example = generate_example(np.random.default_rng(2), event=False)
    assert example.label == 0
    assert example.p_arrival is None


def test_detectors_return_valid_outputs():
    example = generate_example(np.random.default_rng(3), event=True)
    deep = DeepPhaseDetector(seed=3).detect(example.signal)
    classic = sta_lta_detect(example.signal)
    assert 0 <= deep.probability <= 1
    assert deep.latency_ms >= 0
    assert isinstance(classic.detected, bool)


def test_business_value_penalizes_false_alarms():
    assert business_value(1, 0) > business_value(1, 1)


def test_business_value_penalizes_missed_events_and_operation():
    assert business_value(1, 0, 0) > business_value(1, 0, 1)
    assert business_value(1, 0, 0, n_windows=0) > business_value(1, 0, 0, n_windows=10)


def test_business_value_formula_matches_documented_terms():
    params = ValueParams(value_per_second=100, false_alarm_cost=50, missed_event_cost=200, operating_cost_per_window=1)
    value = business_value(3, 2, 1, n_windows=10, seconds_advance=2.0, params=params)
    assert value == 3 * 2.0 * 100 - 2 * 50 - 1 * 200 - 10 * 1


def test_value_curve_covers_thresholds_and_optimum_is_max():
    examples = make_eval_set(seed=5, n_events=8, n_noise=8)
    curve = value_curve(DeepPhaseDetector(seed=5), examples=examples)
    rows = curve["DeepPhase"]
    assert [r["threshold"] for r in rows] == list(DEEP_THRESHOLDS)
    best = optimal_operating_point(rows)
    assert best["business_value"] == max(r["business_value"] for r in rows)


def test_sensitivity_scenarios_are_complete():
    validation = make_eval_set(seed=6, n_events=8, n_noise=8)
    test = make_eval_set(seed=7, n_events=8, n_noise=8)
    rows = sensitivity_analysis(DeepPhaseDetector(seed=5), validation=validation, test=test)
    assert len(rows) == len(SENSITIVITY_FIELDS) * len(SENSITIVITY_FACTORS)
    assert {r["change_pct"] for r in rows} == {-50, -20, 20, 50}
    assert all(r["optimal_threshold"] in DEEP_THRESHOLDS for r in rows)


def test_validation_and_test_sets_differ():
    a = make_eval_set(seed=VALIDATION_SEED, n_events=2, n_noise=2)
    b = make_eval_set(n_events=2, n_noise=2)
    assert not np.allclose(a[0].signal, b[0].signal)


def test_sta_lta_is_causal():
    """Cambiar muestras futuras no debe alterar la alerta ya emitida."""
    example = generate_example(np.random.default_rng(11), event=True)
    first = sta_lta_detect(example.signal, threshold=2.0)
    assert first.detected
    cut = first.sample + 1
    truncated = sta_lta_detect(example.signal[:cut], threshold=2.0)
    assert truncated.sample == first.sample


def test_sta_lta_ignores_warmup_window():
    signal = np.zeros((1200, 3), dtype=np.float32)
    signal[:50] = 10.0  # transitorio en el primer segundo
    assert not sta_lta_detect(signal, threshold=2.0).detected


def _outcome_with_sample(sample):
    example = generate_example(np.random.default_rng(4), event=True)
    detection = Detection(sample is not None, sample, 1.0, 0.0)
    return _outcome(lambda _signal: detection, [example]), example


def test_late_or_premature_alert_counts_as_missed_event():
    _, example = _outcome_with_sample(None)
    late, _ = _outcome_with_sample(example.s_arrival + 10)
    premature, _ = _outcome_with_sample(max(0, example.p_arrival - 100))
    on_time, _ = _outcome_with_sample(example.p_arrival + 20)
    assert (late["tp"], late["fn"]) == (0, 1)
    assert (premature["tp"], premature["fn"]) == (0, 1)
    assert (on_time["tp"], on_time["fn"]) == (1, 0)
