"""Carga de STEAD y conversión a ventanas ``SignalExample``.

Nada se descarga ni se abre al importar este módulo. STEAD **no se descarga
automáticamente**: hay que bajar ``merged.csv`` y ``merged.hdf5`` desde
https://github.com/smousavi05/STEAD y pasar la carpeta con ``--stead-dir``.

Dos backends leen esos archivos:

* ``hdf5`` (por defecto): lee solo las trazas del subconjunto con ``h5py``.
* ``seisbench``: usa ``seisbench.data.STEAD``. Con ``basepath`` convierte TODO el
  dataset a su formato (horas y decenas de GB extra) la primera vez; solo conviene
  si ya tienes la caché de SeisBench.

La partición train/dev/test es **por evento** (``source_id``), para que varias
estaciones del mismo sismo no queden a ambos lados del corte.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # importado como paquete (pytest) o como script (python src/evaluate.py)
    from .seismic import SAMPLE_RATE, WARMUP_SAMPLES, SignalExample, _bandpass
except ImportError:  # pragma: no cover
    from seismic import SAMPLE_RATE, WARMUP_SAMPLES, SignalExample, _bandpass

SPLITS = ("train", "dev", "test")
EVENT_CATEGORY = "earthquake_local"
NOISE_CATEGORY = "noise"
DOWNLOAD_HELP = (
    "Descarga STEAD desde https://github.com/smousavi05/STEAD y pasa la carpeta que "
    "contiene merged.csv y merged.hdf5 con --stead-dir."
)


@dataclass(frozen=True)
class SteadConfig:
    n_events: int = 3000  # total entre train/dev/test
    n_noise: int = 3000
    window_s: float = 12.0
    p_offset_s: tuple[float, float] = (2.0, 5.0)  # igual que el generador sintético
    min_snr_db: float | None = None
    min_s_minus_p_s: float = 1.0
    max_s_minus_p_s: float = 30.0
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 0


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------


def finalize_window(segment: np.ndarray) -> np.ndarray | None:
    """Quita la media, filtra 1-18 Hz y normaliza por desviación estándar.

    La normalización es necesaria porque STEAD está en cuentas sin escala común.
    Devuelve None si la ventana es inutilizable (NaN o señal plana).
    """
    segment = np.asarray(segment, dtype=np.float64)
    if not np.all(np.isfinite(segment)):
        return None
    segment = segment - segment.mean(axis=0, keepdims=True)
    filtered = _bandpass(segment, SAMPLE_RATE).astype(np.float64)
    std = filtered.std()
    if not np.isfinite(std) or std < 1e-9:
        return None
    return (filtered / std).astype(np.float32)


def make_window(
    waveform: np.ndarray,
    p_sample: float | None,
    s_sample: float | None,
    rng: np.random.Generator,
    *,
    event: bool,
    config: SteadConfig,
    meta: dict | None = None,
) -> SignalExample | None:
    """Recorta una ventana de ``window_s`` s de una traza (T, 3) en orden ZNE."""
    n = int(config.window_s * SAMPLE_RATE)
    total = len(waveform)
    if total < n:
        return None
    if event:
        offset = int(rng.uniform(*config.p_offset_s) * SAMPLE_RATE)
        start = int(np.clip(int(p_sample) - offset, 0, total - n))
        p_index = int(p_sample) - start
        # La P debe caer fuera del calentamiento y con margen antes del final.
        if not (WARMUP_SAMPLES + 50 <= p_index <= n - 100):
            return None
        s_index = int(s_sample) - start  # puede quedar más allá de la ventana
    else:
        start = int(rng.integers(0, total - n + 1))
        p_index = s_index = None
    signal = finalize_window(waveform[start : start + n])
    if signal is None:
        return None
    return SignalExample(signal, p_index, s_index, int(event), dict(meta or {}))


# ---------------------------------------------------------------------------
# Metadatos y partición por evento
# ---------------------------------------------------------------------------


def assign_split(key: str, fractions: tuple[float, float, float]) -> str:
    """Partición determinista por hash estable (no depende de PYTHONHASHSEED)."""
    value = int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000
    if value < fractions[0]:
        return "train"
    if value < fractions[0] + fractions[1]:
        return "dev"
    return "test"


def _parse_snr(value) -> float:
    if isinstance(value, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        return float(np.mean([float(x) for x in numbers])) if numbers else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_metadata(raw):
    """Unifica los nombres de columnas de STEAD crudo y de SeisBench."""
    import pandas as pd

    def pick(*names):
        for name in names:
            if name in raw.columns:
                return raw[name]
        return None

    required = {
        "name": pick("trace_name"),
        "category": pick("trace_category"),
        "p": pick("trace_p_arrival_sample", "p_arrival_sample"),
        "s": pick("trace_s_arrival_sample", "s_arrival_sample"),
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Faltan columnas de metadatos de STEAD para: {missing}")
    snr = pick("trace_snr_db", "snr_db")
    source = pick("source_id")
    frame = pd.DataFrame(
        {
            "pos": np.arange(len(raw)),
            "name": required["name"].astype(str).values,
            "category": required["category"].astype(str).values,
            "p": pd.to_numeric(required["p"], errors="coerce").values,
            "s": pd.to_numeric(required["s"], errors="coerce").values,
            "snr_db": snr.map(_parse_snr).values if snr is not None else np.nan,
        }
    )
    frame["source_id"] = source.astype(str).values if source is not None else frame["name"]
    return frame


def select_subset(meta, config: SteadConfig) -> dict[str, dict[str, object]]:
    """Filtra y muestrea eventos y ruido por partición, sin leer formas de onda."""
    sp = (meta["s"] - meta["p"]) / SAMPLE_RATE
    events = meta[
        (meta["category"] == EVENT_CATEGORY)
        & meta["p"].notna()
        & meta["s"].notna()
        & (meta["p"] > 0)
        & (sp >= config.min_s_minus_p_s)
        & (sp <= config.max_s_minus_p_s)
    ]
    if config.min_snr_db is not None:
        events = events[events["snr_db"] >= config.min_snr_db]
    noise = meta[meta["category"] == NOISE_CATEGORY]

    events = events.assign(split=[assign_split(k, config.split_fractions) for k in events["source_id"]])
    noise = noise.assign(split=[assign_split(k, config.split_fractions) for k in noise["name"]])

    chosen: dict[str, dict[str, object]] = {}
    for index, split in enumerate(SPLITS):
        fraction = config.split_fractions[index]
        n_ev, n_no = round(config.n_events * fraction), round(config.n_noise * fraction)
        ev, no = events[events["split"] == split], noise[noise["split"] == split]
        chosen[split] = {
            "events": ev.sample(n=min(n_ev, len(ev)), random_state=config.seed),
            "noise": no.sample(n=min(n_no, len(no)), random_state=config.seed),
        }
    return chosen


# ---------------------------------------------------------------------------
# Backends de lectura
# ---------------------------------------------------------------------------


class Hdf5Source:
    """Lee ``merged.csv`` y ``merged.hdf5`` de STEAD directamente con h5py."""

    def __init__(self, root: Path | str) -> None:
        import pandas as pd

        root = Path(root)
        csv_path, h5_path = root / "merged.csv", root / "merged.hdf5"
        for path in (csv_path, h5_path):
            if not path.is_file():
                raise FileNotFoundError(f"No existe {path}. {DOWNLOAD_HELP}")
        self.metadata = pd.read_csv(csv_path, low_memory=False)
        self._h5_path = h5_path
        self._file = None

    def read(self, position: int) -> np.ndarray:
        import h5py

        if self._file is None:
            self._file = h5py.File(self._h5_path, "r")
        name = self.metadata["trace_name"].iloc[position]
        data = self._file["data"][name][()]  # (T, 3) en orden E, N, Z
        return np.asarray(data)[:, ::-1]  # -> Z, N, E

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class SeisBenchSource:
    """Usa ``seisbench.data.STEAD`` (ver advertencia en el docstring del módulo)."""

    def __init__(self, root: Path | str | None) -> None:
        try:
            import seisbench.data as sbd
        except ImportError as error:  # pragma: no cover
            raise ImportError("Instala seisbench: pip install -r requirements-data.txt") from error
        kwargs = {"download_kwargs": {"basepath": str(root)}} if root else {}
        self._data = sbd.STEAD(component_order="ZNE", dimension_order="NCW", sampling_rate=100, **kwargs)
        self.metadata = self._data.metadata

    def read(self, position: int) -> np.ndarray:
        waveform, _ = self._data.get_sample(position)  # (C, W)
        return np.asarray(waveform).T

    def close(self) -> None:
        return None


def open_source(root: Path | str | None, backend: str = "hdf5"):
    if backend == "hdf5":
        if root is None:
            raise ValueError(f"Falta la carpeta de STEAD. {DOWNLOAD_HELP}")
        return Hdf5Source(root)
    if backend == "seisbench":
        return SeisBenchSource(root)
    raise ValueError(f"Backend desconocido: {backend!r} (usa 'hdf5' o 'seisbench')")


# ---------------------------------------------------------------------------
# Caché de ventanas y carga completa
# ---------------------------------------------------------------------------


def save_examples(path: Path | str, splits: dict[str, list[SignalExample]]) -> None:
    """Guarda las ventanas ya procesadas para no volver a leer el HDF5."""
    arrays: dict[str, np.ndarray] = {}
    for split, examples in splits.items():
        if not examples:
            continue
        arrays[f"{split}_signal"] = np.stack([e.signal for e in examples])
        arrays[f"{split}_p"] = np.array([-1 if e.p_arrival is None else e.p_arrival for e in examples])
        arrays[f"{split}_s"] = np.array([-1 if e.s_arrival is None else e.s_arrival for e in examples])
        arrays[f"{split}_label"] = np.array([e.label for e in examples])
        arrays[f"{split}_snr"] = np.array([e.meta.get("snr_db", np.nan) for e in examples], dtype=float)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_examples(path: Path | str) -> dict[str, list[SignalExample]]:
    out: dict[str, list[SignalExample]] = {split: [] for split in SPLITS}
    with np.load(path) as data:
        for split in SPLITS:
            if f"{split}_signal" not in data:
                continue
            for signal, p, s, label, snr in zip(
                data[f"{split}_signal"], data[f"{split}_p"], data[f"{split}_s"], data[f"{split}_label"], data[f"{split}_snr"]
            ):
                out[split].append(
                    SignalExample(
                        signal, None if p < 0 else int(p), None if s < 0 else int(s), int(label), {"snr_db": float(snr)}
                    )
                )
    return out


def load_stead(
    root: Path | str | None,
    *,
    backend: str = "hdf5",
    config: SteadConfig = SteadConfig(),
    cache: Path | str | None = None,
) -> dict[str, list[SignalExample]]:
    """Devuelve ``{'train': [...], 'dev': [...], 'test': [...]}`` con ventanas STEAD."""
    if cache is not None and Path(cache).is_file():
        return load_examples(cache)

    source = open_source(root, backend)
    try:
        chosen = select_subset(normalize_metadata(source.metadata), config)
        rng = np.random.default_rng(config.seed)
        splits: dict[str, list[SignalExample]] = {split: [] for split in SPLITS}
        for split, groups in chosen.items():
            for kind, rows in groups.items():
                for row in rows.itertuples():
                    example = make_window(
                        source.read(int(row.pos)),
                        row.p if kind == "events" else None,
                        row.s if kind == "events" else None,
                        rng,
                        event=kind == "events",
                        config=config,
                        meta={"snr_db": float(row.snr_db), "source_id": row.source_id, "trace_name": row.name},
                    )
                    if example is not None:
                        splits[split].append(example)
    finally:
        source.close()

    if cache is not None:
        save_examples(cache, splits)
    return splits
