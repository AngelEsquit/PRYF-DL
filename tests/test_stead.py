"""Pruebas del cargador STEAD y de FDSN con archivos falsos (sin descargas)."""

import h5py
import numpy as np
import pandas as pd
import pytest

from src.fdsn import FdsnConfig, components_to_window
from src.seismic import DeepPhaseDetector, SignalExample
from src.stead import (
    SteadConfig,
    assign_split,
    load_examples,
    load_stead,
    make_window,
    normalize_metadata,
    open_source,
    save_examples,
    select_subset,
)


def build_fake_stead(root, n_sources=60, per_source=2, n_noise=60):
    """Crea merged.csv y merged.hdf5 con el mismo formato que STEAD (trazas E,N,Z de 6000 muestras)."""
    rng = np.random.default_rng(0)
    rows = []
    with h5py.File(root / "merged.hdf5", "w") as f:
        group = f.create_group("data")
        for source in range(n_sources):
            for station in range(per_source):
                name = f"ev{source}_{station}"
                p = int(rng.integers(1000, 3000))
                s = p + int(rng.integers(200, 1500))
                data = rng.normal(0, 1, (6000, 3)).astype(np.float32)
                t = np.arange(6000)
                data += (np.exp(-((t - p) / 40.0) ** 2) * 8 * np.sin(2 * np.pi * 5 * t / 100))[:, None]
                group.create_dataset(name, data=data)
                rows.append(
                    dict(trace_name=name, trace_category="earthquake_local", p_arrival_sample=p, s_arrival_sample=s,
                         snr_db="[10.0 11.0 12.0]", source_id=f"src{source}")
                )
        for i in range(n_noise):
            name = f"noise{i}"
            group.create_dataset(name, data=rng.normal(0, 1, (6000, 3)).astype(np.float32))
            rows.append(
                dict(trace_name=name, trace_category="noise", p_arrival_sample=np.nan, s_arrival_sample=np.nan,
                     snr_db=np.nan, source_id=np.nan)
            )
    pd.DataFrame(rows).to_csv(root / "merged.csv", index=False)


def test_missing_files_give_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="github.com/smousavi05/STEAD"):
        open_source(tmp_path, "hdf5")
    with pytest.raises(ValueError, match="github.com/smousavi05/STEAD"):
        open_source(None, "hdf5")


def test_normalize_metadata_accepts_raw_and_seisbench_names():
    raw = pd.DataFrame(
        {"trace_name": ["a"], "trace_category": ["noise"], "p_arrival_sample": [1], "s_arrival_sample": [2],
         "snr_db": ["[1.0 3.0 5.0]"], "source_id": ["x"]}
    )
    seisbench = raw.rename(
        columns={"p_arrival_sample": "trace_p_arrival_sample", "s_arrival_sample": "trace_s_arrival_sample",
                 "snr_db": "trace_snr_db"}
    )
    for frame in (raw, seisbench):
        meta = normalize_metadata(frame)
        assert meta["snr_db"].iloc[0] == pytest.approx(3.0)
        assert meta["p"].iloc[0] == 1
    with pytest.raises(ValueError):
        normalize_metadata(raw.drop(columns="trace_name"))


def test_assign_split_is_deterministic_and_roughly_proportional():
    keys = [f"src{i}" for i in range(2000)]
    first = [assign_split(k, (0.7, 0.15, 0.15)) for k in keys]
    assert first == [assign_split(k, (0.7, 0.15, 0.15)) for k in keys]
    assert 0.65 < first.count("train") / len(keys) < 0.75


def test_split_is_by_event_no_source_in_two_splits(tmp_path):
    build_fake_stead(tmp_path)
    source = open_source(tmp_path, "hdf5")
    chosen = select_subset(normalize_metadata(source.metadata), SteadConfig(n_events=100, n_noise=40))
    seen = {}
    for split, groups in chosen.items():
        for source_id in groups["events"]["source_id"]:
            assert seen.setdefault(source_id, split) == split


def test_load_stead_builds_windows_and_caches(tmp_path):
    build_fake_stead(tmp_path)
    cache = tmp_path / "subset.npz"
    config = SteadConfig(n_events=80, n_noise=40, seed=1)
    splits = load_stead(tmp_path, config=config, cache=cache)
    train = splits["train"]
    assert train and any(e.label for e in train) and any(not e.label for e in train)
    for example in train:
        assert example.signal.shape == (1200, 3)
        assert np.isfinite(example.signal).all()
        if example.label:
            assert 150 <= example.p_arrival <= 1100
            assert example.s_arrival > example.p_arrival
        else:
            assert example.p_arrival is None

    (tmp_path / "merged.hdf5").unlink()  # la segunda carga debe salir solo de la caché
    again = load_stead(tmp_path, config=config, cache=cache)
    assert [len(again[s]) for s in again] == [len(splits[s]) for s in splits]


def test_make_window_rejects_p_inside_warmup():
    wave = np.random.default_rng(0).normal(size=(6000, 3))
    window = make_window(wave, 120, 400, np.random.default_rng(0), event=True, config=SteadConfig())
    assert window is None


def test_make_window_rejects_flat_signal():
    window = make_window(np.zeros((6000, 3)), None, None, np.random.default_rng(0), event=False, config=SteadConfig())
    assert window is None


def test_save_and_load_examples_roundtrip(tmp_path):
    signal = np.random.default_rng(0).normal(size=(1200, 3)).astype(np.float32)
    examples = {"train": [SignalExample(signal, 300, 500, 1, {"snr_db": 12.0}), SignalExample(signal, None, None, 0)]}
    save_examples(tmp_path / "x.npz", examples)
    loaded = load_examples(tmp_path / "x.npz")["train"]
    assert (loaded[0].p_arrival, loaded[0].s_arrival, loaded[0].label) == (300, 500, 1)
    assert loaded[1].p_arrival is None and loaded[1].label == 0


def test_detector_can_train_on_external_examples(tmp_path):
    build_fake_stead(tmp_path)
    splits = load_stead(tmp_path, config=SteadConfig(n_events=60, n_noise=30, seed=2))
    detector = DeepPhaseDetector(seed=1, train_examples=splits["train"])
    detection = detector.detect(splits["test"][0].signal)
    assert 0 <= detection.probability <= 1


def test_fdsn_components_to_window_orders_and_validates():
    rng = np.random.default_rng(0)
    config = FdsnConfig()
    comps = {"Z": rng.normal(size=1300), "1": rng.normal(size=1300), "2": rng.normal(size=1300)}
    example = components_to_window(comps, 3.5, 6.0, config, {"station": "XX.AAA"})
    assert example.signal.shape == (1200, 3)
    assert (example.p_arrival, example.s_arrival, example.label) == (350, 600, 1)
    assert components_to_window({"Z": comps["Z"]}, 3.5, 6.0, config, {}) is None
