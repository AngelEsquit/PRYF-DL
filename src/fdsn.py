"""Validación local con registros FDSN/IRIS (Guatemala y región).

Nada se descarga al importar. ``fetch_fdsn_examples`` consulta el servicio solo
cuando se llama explícitamente y guarda el resultado en una caché ``.npz``.

IMPORTANTE: las llegadas P y S de estas ventanas son TEÓRICAS (modelo iasp91 de
TauP, con hipocentro y estación), no picks manuales. Su error típico es de uno o
dos segundos, así que sirven como validación gruesa de transferencia, no como
medida precisa del error de picking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .seismic import SAMPLE_RATE, SignalExample
    from .stead import finalize_window, load_examples, save_examples
except ImportError:  # pragma: no cover
    from seismic import SAMPLE_RATE, SignalExample
    from stead import finalize_window, load_examples, save_examples

P_PHASES = ("P", "p", "Pn", "Pg")
S_PHASES = ("S", "s", "Sn", "Sg")


@dataclass(frozen=True)
class FdsnConfig:
    provider: str = "IRIS"
    start: str = "2015-01-01"
    end: str = "2025-12-31"
    min_magnitude: float = 3.5
    event_lat: tuple[float, float] = (10.0, 20.0)  # Guatemala y alrededores
    event_lon: tuple[float, float] = (-95.0, -85.0)
    station_lat: tuple[float, float] = (13.0, 18.5)
    station_lon: tuple[float, float] = (-92.5, -88.0)
    channels: str = "HH?,BH?,EH?"
    max_events: int = 30
    max_stations_per_event: int = 3
    max_distance_km: float = 400.0
    window_s: float = 12.0
    p_offset_s: float = 3.5  # P cae dentro del rango 2-5 s usado en el entrenamiento


def theoretical_arrivals(
    depth_km: float, event_lat: float, event_lon: float, station_lat: float, station_lon: float
) -> tuple[float, float] | None:
    """Segundos desde el origen hasta P y S (iasp91). None si no hay fase."""
    from obspy.taup import TauPyModel

    model = TauPyModel(model="iasp91")

    def first(phases):
        arrivals = model.get_travel_times_geo(
            source_depth_in_km=max(depth_km, 0.0),
            source_latitude_in_deg=event_lat,
            source_longitude_in_deg=event_lon,
            receiver_latitude_in_deg=station_lat,
            receiver_longitude_in_deg=station_lon,
            phase_list=list(phases),
        )
        return min(a.time for a in arrivals) if arrivals else None

    tp, ts = first(P_PHASES), first(S_PHASES)
    return None if tp is None or ts is None else (tp, ts)


def components_to_window(
    components: dict[str, np.ndarray], p_seconds: float, s_seconds: float, config: FdsnConfig, meta: dict
) -> SignalExample | None:
    """Arma una ventana ZNE ya remuestreada a 100 Hz. ``p_seconds`` es relativo al inicio."""
    n = int(config.window_s * SAMPLE_RATE)
    needed = {"Z": ("Z",), "N": ("N", "1"), "E": ("E", "2")}
    columns = []
    for letters in needed.values():
        key = next((k for k in letters if k in components), None)
        if key is None or len(components[key]) < n:
            return None
        columns.append(np.asarray(components[key][:n], dtype=np.float64))
    signal = finalize_window(np.column_stack(columns))
    if signal is None:
        return None
    return SignalExample(signal, round(p_seconds * SAMPLE_RATE), round(s_seconds * SAMPLE_RATE), 1, meta)


def fetch_fdsn_examples(
    config: FdsnConfig = FdsnConfig(), cache: Path | str | None = None
) -> list[SignalExample]:
    """Descarga eventos y formas de onda; devuelve solo ventanas con evento.

    Las ventanas de ruido para esta validación se reutilizan del conjunto de
    prueba de STEAD (ver ``evaluate.py``). No probado contra el servicio real.
    """
    if cache is not None and Path(cache).is_file():
        return load_examples(cache)["test"]

    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics import gps2dist_azimuth

    client = Client(config.provider)
    events = client.get_events(
        starttime=UTCDateTime(config.start),
        endtime=UTCDateTime(config.end),
        minmagnitude=config.min_magnitude,
        minlatitude=config.event_lat[0],
        maxlatitude=config.event_lat[1],
        minlongitude=config.event_lon[0],
        maxlongitude=config.event_lon[1],
        orderby="magnitude",
        limit=config.max_events,
    )
    inventory = client.get_stations(
        network="*",
        station="*",
        channel=config.channels,
        minlatitude=config.station_lat[0],
        maxlatitude=config.station_lat[1],
        minlongitude=config.station_lon[0],
        maxlongitude=config.station_lon[1],
        level="channel",
        starttime=UTCDateTime(config.start),
    )
    stations = [
        (net.code, sta.code, sta.latitude, sta.longitude) for net in inventory for sta in net
    ]

    examples: list[SignalExample] = []
    for event in events:
        origin = event.preferred_origin() or event.origins[0]
        depth_km = (origin.depth or 0.0) / 1000.0
        ranked = sorted(
            (gps2dist_azimuth(origin.latitude, origin.longitude, lat, lon)[0] / 1000.0, net, sta, lat, lon)
            for net, sta, lat, lon in stations
        )
        for distance_km, net, sta, lat, lon in ranked[: config.max_stations_per_event]:
            if distance_km > config.max_distance_km:
                continue
            arrivals = theoretical_arrivals(depth_km, origin.latitude, origin.longitude, lat, lon)
            if arrivals is None:
                continue
            tp, ts = arrivals
            start = origin.time + tp - config.p_offset_s
            try:
                stream = client.get_waveforms(
                    net, sta, "*", config.channels, start - 5, start + config.window_s + 5
                )
                stream.merge(fill_value=0)
                stream.detrend("demean")
                stream.resample(SAMPLE_RATE)
                stream.trim(start, start + config.window_s, pad=False)
            except Exception:  # sin datos, canal faltante, etc.: se omite esa estación
                continue
            components = {trace.stats.channel[-1]: trace.data for trace in stream}
            example = components_to_window(
                components,
                config.p_offset_s,
                ts - tp + config.p_offset_s,
                config,
                {"station": f"{net}.{sta}", "distance_km": distance_km, "event_time": str(origin.time)},
            )
            if example is not None:
                examples.append(example)

    if cache is not None:
        save_examples(cache, {"test": examples})
    return examples
