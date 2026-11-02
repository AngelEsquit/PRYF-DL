import numpy as np

from src.seismic import DeepPhaseDetector, business_value, generate_example, sta_lta_detect


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
