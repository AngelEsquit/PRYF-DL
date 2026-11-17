"""Explicabilidad (Laboratorio 10): Integrated Gradients y oclusión sobre la CNN 1-D.

Aplica XAI al modelo ya entrenado (sin reentrenar) y regenera, con semilla fija,
todas las figuras y tablas del informe de explicabilidad en ``results/xai/``.

Uso:
    python src/explain.py                              # STEAD si existe el caché, si no sintético
    python src/explain.py --source stead --cache data/stead_subset_10000ev.npz
    python src/explain.py --source synthetic           # desarrollo sin datos reales

Definiciones clave
------------------
* El modelo devuelve un logit por muestra, ``f(x)[t]``. Para explicar *una decisión*
  se elige una muestra objetivo ``t*``: la primera que supera el umbral de alerta
  (tras el calentamiento) o, si no hay alerta, el máximo del logit. El escalar
  explicado es ``f(x)[t*]``.
* Método principal: Integrated Gradients (captum) con línea base cero. Alternativa
  para contraste y verificación: oclusión por ventanas de 0.5 s.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import spearmanr

try:  # permite ejecutar como script (python src/explain.py) o como módulo
    from .models import build_model
    from .seismic import PICK_TOLERANCE_S, SAMPLE_RATE, WARMUP_SAMPLES, SignalExample, make_eval_set
    from .stead import load_examples
except ImportError:
    from models import build_model
    from seismic import PICK_TOLERANCE_S, SAMPLE_RATE, WARMUP_SAMPLES, SignalExample, make_eval_set
    from stead import load_examples

CHANNELS = ("Z", "N", "E")
OCC_WINDOW = 50  # 0.5 s a 100 Hz: ventana de oclusión y de borrado
OCC_STRIDE = 25
DEFAULT_CHECKPOINTS = {"stead": "checkpoints_stead/best_model.pt", "synthetic": "checkpoints/best_model.pt"}


# ---------------------------------------------------------------------------
# Modelo e inferencia
# ---------------------------------------------------------------------------


def load_model(path: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    name = ckpt.get("model_name") or ckpt.get("config", {}).get("model", "conv1d")
    if name != "conv1d":
        raise SystemExit(f"explain.py solo soporta la CNN 1-D (conv1d); el checkpoint es '{name}'.")
    model = build_model(name, **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def to_tensor(signals: np.ndarray) -> torch.Tensor:
    """(N, T, 3) -> (N, 3, T) float32, igual que PyTorchPhaseDetector.detect."""
    return torch.from_numpy(np.ascontiguousarray(np.transpose(signals, (0, 2, 1)), dtype=np.float32))


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, batch: int = 64) -> np.ndarray:
    return np.concatenate([model(x[i : i + batch]).numpy() for i in range(0, len(x), batch)])


def decision_sample(logits: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Muestra objetivo t* por ventana y bandera de alerta (primera excedencia tras el calentamiento)."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    t_star = np.zeros(len(logits), dtype=int)
    alerted = np.zeros(len(logits), dtype=bool)
    for i, p in enumerate(probs):
        candidates = np.flatnonzero(p >= threshold)
        candidates = candidates[candidates >= WARMUP_SAMPLES]
        if len(candidates):
            t_star[i], alerted[i] = candidates[0], True
        else:
            t_star[i] = WARMUP_SAMPLES + int(np.argmax(logits[i, WARMUP_SAMPLES:]))
    return t_star, alerted


def classify(example: SignalExample, alerted: bool, t_star: int) -> str:
    """TP/FN/FP/TN con el mismo criterio de 'aviso útil' que el resto del proyecto."""
    if not example.label:
        return "FP" if alerted else "TN"
    earliest = example.p_arrival - PICK_TOLERANCE_S * SAMPLE_RATE
    latest = example.s_arrival if example.s_arrival is not None else np.inf
    return "TP" if alerted and earliest <= t_star < latest else "FN"


def snr_value(example: SignalExample) -> float:
    """SNR de STEAD si existe; si no, un proxy RMS(post-P) / RMS(pre-P) en dB."""
    snr = example.meta.get("snr_db", np.nan)
    if snr is not None and np.isfinite(snr):
        return float(snr)
    if not example.label:
        return float("nan")
    p = example.p_arrival
    pre = example.signal[WARMUP_SAMPLES : max(p - 50, WARMUP_SAMPLES + 20)]
    post = example.signal[p : p + 100]
    return float(20 * np.log10((post.std() + 1e-9) / (pre.std() + 1e-9)))


# ---------------------------------------------------------------------------
# Métodos de atribución
# ---------------------------------------------------------------------------


def integrated_gradients(model, x, targets, *, n_steps, baseline=None, batch=16):
    """IG con línea base ``baseline`` (por defecto ceros). Devuelve (attr, delta relativo)."""
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    attrs, rel_deltas = [], []
    for i in range(0, len(x), batch):
        xb, tb = x[i : i + batch], torch.as_tensor(targets[i : i + batch])
        base = torch.zeros_like(xb) if baseline is None else baseline[i : i + batch]
        attr, delta = ig.attribute(
            xb, baselines=base, target=tb, n_steps=n_steps, internal_batch_size=n_steps, return_convergence_delta=True
        )
        with torch.no_grad():
            gap = model(xb).gather(1, tb[:, None]).squeeze(1) - model(base).gather(1, tb[:, None]).squeeze(1)
        attrs.append(attr.detach().numpy())
        rel_deltas.append((delta.abs() / (gap.abs() + 1e-9)).detach().numpy())
    return np.concatenate(attrs), np.concatenate(rel_deltas)


def occlusion(model, x, targets, batch=16):
    """Oclusión: caída del logit objetivo al poner a cero una ventana de 0.5 s (todos los canales)."""
    from captum.attr import Occlusion

    occ = Occlusion(model)
    out = []
    for i in range(0, len(x), batch):
        xb, tb = x[i : i + batch], torch.as_tensor(targets[i : i + batch])
        out.append(
            occ.attribute(
                xb,
                sliding_window_shapes=(3, OCC_WINDOW),
                strides=(3, OCC_STRIDE),
                baselines=0.0,
                target=tb,
            ).numpy()
        )
    return np.concatenate(out)


def window_scores(attr: np.ndarray) -> np.ndarray:
    """(N, 3, T) -> (N, T//OCC_WINDOW): atribución firmada sumada por ventana no solapada."""
    n, _, t = attr.shape
    k = t // OCC_WINDOW
    return attr.sum(axis=1)[:, : k * OCC_WINDOW].reshape(n, k, OCC_WINDOW).sum(axis=2)


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------


def deletion_curves(model, x, targets, scores: dict[str, np.ndarray], rng, max_k=8, n_random=5):
    """Fidelidad: caída media del logit objetivo al borrar las k ventanas más atribuidas."""
    base = predict_target(model, x, targets)
    n_windows = next(iter(scores.values())).shape[1]
    curves = {name: [0.0] for name in (*scores, "aleatorio")}
    for k in range(1, max_k + 1):
        for name, sc in scores.items():
            order = np.argsort(-sc, axis=1)[:, :k]
            curves[name].append(float(np.mean(base - predict_target(model, delete(x, order), targets))))
        drops = []
        for _ in range(n_random):
            order = np.stack([rng.permutation(n_windows)[:k] for _ in range(len(x))])
            drops.append(np.mean(base - predict_target(model, delete(x, order), targets)))
        curves["aleatorio"].append(float(np.mean(drops)))
    return curves


def delete(x: torch.Tensor, windows: np.ndarray) -> torch.Tensor:
    out = x.clone()
    for i, idx in enumerate(windows):
        for w in idx:
            out[i, :, w * OCC_WINDOW : (w + 1) * OCC_WINDOW] = 0.0
    return out


@torch.no_grad()
def predict_target(model, x, targets) -> np.ndarray:
    logits = predict_logits(model, x)
    return logits[np.arange(len(x)), targets]


def temporal_profile(attr: np.ndarray) -> np.ndarray:
    """|atribución| sumada sobre canales: (N, T)."""
    return np.abs(attr).sum(axis=1)


RECEPTIVE_HALF = 100  # el campo receptivo de la CNN es ~+-93 muestras; fuera de eso toda atribución es 0


def mean_spearman(a: np.ndarray, b: np.ndarray, centers: np.ndarray) -> float:
    """Correlación de Spearman media entre dos perfiles, solo en +-1 s alrededor de t*.

    Fuera del campo receptivo ambos perfiles valen exactamente 0 por construcción de la red;
    incluirlo inflaría la correlación de forma trivial (~1.0) en cualquier comparación.
    """
    vals = []
    for i, c in enumerate(centers):
        lo, hi = max(0, c - RECEPTIVE_HALF), min(a.shape[1], c + RECEPTIVE_HALF + 1)
        vals.append(spearmanr(a[i, lo:hi], b[i, lo:hi])[0])
    return float(np.nanmean(vals))


def causal_check(model, x, targets, threshold, delays=(0, 10, 20, 30, 50, 100)):
    """Fracción de alertas TP que se mantienen si solo se conocen las muestras hasta t* + retraso.

    La CNN usa convoluciones simétricas (no causales): el logit en t* depende de muestras
    posteriores. Se anula la señal futura y se revisa si la probabilidad en t* sigue sobre el umbral.
    """
    out = {}
    for d in delays:
        trunc = x.clone()
        for i, t in enumerate(targets):
            trunc[i, :, t + d + 1 :] = 0.0
        prob = 1 / (1 + np.exp(-predict_target(model, trunc, targets)))
        out[f"{d / SAMPLE_RATE:.2f}s"] = float(np.mean(prob >= threshold))
    return out


def aligned_profile(profiles: np.ndarray, p_arrivals: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Alinea cada perfil respecto a la P verdadera: columnas = lag en muestras [lo, hi)."""
    out = np.full((len(profiles), hi - lo), np.nan)
    for i, (prof, p) in enumerate(zip(profiles, p_arrivals)):
        a, b = max(0, p + lo), min(len(prof), p + hi)
        out[i, a - (p + lo) : b - (p + lo)] = prof[a:b]
    return out


# ---------------------------------------------------------------------------
# Selección de casos locales
# ---------------------------------------------------------------------------


def select_cases(examples, outcomes, max_prob, snr, threshold) -> list[tuple[str, int]]:
    """Casos pedidos por el laboratorio: acierto confiado, error, cercano al umbral, subgrupo."""
    outcomes = np.array(outcomes)
    chosen: list[tuple[str, int]] = []
    used: set[int] = set()

    def pick(label: str, mask: np.ndarray, key: np.ndarray, largest: bool) -> None:
        idx = np.flatnonzero(mask & ~np.isin(np.arange(len(mask)), list(used)))
        if len(idx) == 0:
            return
        best = idx[np.argmax(key[idx])] if largest else idx[np.argmin(key[idx])]
        chosen.append((label, int(best)))
        used.add(int(best))

    pick("Acierto de alta confianza", outcomes == "TP", max_prob, largest=True)
    # Error: el falso negativo con mayor probabilidad máxima (casi detectado) es el más instructivo.
    fn_mask = outcomes == "FN"
    if fn_mask.any():
        pick("Error: sin aviso útil", fn_mask, max_prob, largest=True)
    pick("Error: falsa alarma", outcomes == "FP", max_prob, largest=True)
    pick("Cercano al umbral", np.ones(len(outcomes), dtype=bool), -np.abs(max_prob - threshold), largest=True)
    ev = np.array([bool(e.label) for e in examples])
    pick("Subgrupo: evento de menor SNR", ev & np.isfinite(snr), snr, largest=False)
    pick("Rechazo correcto de ruido", outcomes == "TN", -max_prob, largest=True)
    return chosen


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def plot_local(cases, examples, logits, ig_attr, occ_attr, t_star, outcomes, snr, threshold, path: Path) -> None:
    cols = 2
    rows = int(np.ceil(len(cases) / cols))
    fig = plt.figure(figsize=(13, 3.3 * rows))
    outer = GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.18)
    time = np.arange(examples[0].signal.shape[0]) / SAMPLE_RATE
    for n, (label, i) in enumerate(cases):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[n], height_ratios=[2.4, 1], hspace=0.08)
        ax, ax_attr = fig.add_subplot(inner[0]), fig.add_subplot(inner[1], sharex=None)
        ex = examples[i]
        scale = np.abs(ex.signal).max() + 1e-9
        for c, color in zip(range(3), ("#222222", "#666666", "#999999")):
            ax.plot(time, ex.signal[:, c] / scale + (2 - c) * 1.1, color=color, lw=0.6)
        prob = 1 / (1 + np.exp(-logits[i]))
        axp = ax.twinx()
        axp.plot(time, prob, color="tab:blue", lw=1.2)
        axp.axhline(threshold, color="tab:blue", ls=":", lw=0.8)
        axp.set_ylim(0, 1.05)
        axp.set_ylabel("P(fase P)", color="tab:blue", fontsize=7)
        if ex.label:
            ax.axvline(ex.p_arrival / SAMPLE_RATE, color="green", lw=1.2)
            if ex.s_arrival is not None and ex.s_arrival < len(time):
                ax.axvline(ex.s_arrival / SAMPLE_RATE, color="darkorange", lw=1.2)
        ax.axvline(t_star[i] / SAMPLE_RATE, color="red", ls="--", lw=1.2)
        ax.set_yticks([2.2, 1.1, 0])
        ax.set_yticklabels(CHANNELS, fontsize=7)
        ax.set_xlim(time[0], time[-1])
        ax.tick_params(labelbottom=False)
        s = snr[i]
        ax.set_title(
            f"{label} [{outcomes[i]}]  |  p_max={prob[WARMUP_SAMPLES:].max():.2f}"
            + (f"  |  SNR={s:.1f} dB" if np.isfinite(s) else ""),
            fontsize=8,
        )
        ig_prof = np.abs(ig_attr[i]).sum(axis=0)
        ax_attr.fill_between(time, ig_prof / (ig_prof.max() + 1e-12), color="tab:red", alpha=0.55, label="|IG|")
        occ_prof = np.clip(occ_attr[i].sum(axis=0), 0, None)
        ax_attr.plot(time, occ_prof / (occ_prof.max() + 1e-12), color="black", lw=0.9, label="Oclusión (+)")
        ax_attr.set_xlim(time[0], time[-1])
        ax_attr.set_ylim(0, 1.05)
        ax_attr.set_yticks([])
        ax_attr.set_xlabel("Tiempo (s)", fontsize=7)
        ax_attr.tick_params(labelsize=7)
        if n == 0:
            ax_attr.legend(fontsize=6, loc="upper left")
    fig.suptitle(
        "Explicaciones locales: verde = P real, naranja = S real, rojo discontinuo = muestra decisión t*", fontsize=9, y=0.995
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global(aligned_mean, lags, channel_share, curves, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(lags, aligned_mean, color="tab:red")
    axes[0].axvline(0, color="green", lw=1.2, label="P real")
    axes[0].set(xlabel="Tiempo respecto a la P real (s)", ylabel="|IG| medio", title="Atribución media (eventos TP)")
    axes[0].legend(fontsize=8)
    axes[1].bar(CHANNELS, channel_share, color=("#333333", "#777777", "#aaaaaa"))
    axes[1].set(ylabel="Fracción de |IG|", title="Atribución por componente")
    ks = np.arange(len(curves["aleatorio"]))
    for name, style in (("Integrated Gradients", "-o"), ("Oclusión", "-s"), ("aleatorio", "--^")):
        axes[2].plot(ks, curves[name], style, label=name if name != "aleatorio" else "Ventanas al azar", ms=4)
    axes[2].set(xlabel="Ventanas de 0.5 s borradas (k)", ylabel="Caída media del logit", title="Fidelidad (prueba de borrado)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI (IG + oclusión) sobre la CNN 1-D de AlertaP")
    parser.add_argument("--source", choices=("auto", "stead", "synthetic"), default="auto")
    parser.add_argument("--cache", default="data/stead_subset_10000ev.npz", help="Caché de STEAD (npz)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=0.50, help="Umbral de alerta (0.50 = elegido en validación)")
    parser.add_argument("--n-global", type=int, default=200, help="Eventos para la explicación global")
    parser.add_argument("--n-steps", type=int, default=64, help="Pasos de integración de IG")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="results/xai")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    source = args.source
    if source == "auto":
        source = "stead" if Path(args.cache).is_file() else "synthetic"
        if source == "synthetic":
            print(f"[!] No existe {args.cache}: se usan datos SINTÉTICOS (solo para desarrollo).")
    if source == "stead":
        if not Path(args.cache).is_file():
            raise SystemExit(f"No existe el caché {args.cache}. Pídelo o genéralo con evaluate.py --source stead.")
        examples = load_examples(args.cache)["test"]
    else:
        examples = make_eval_set(seed=2026, n_events=300, n_noise=300)
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[source]
    suffix = "" if source == "stead" else "_synthetic"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[*] Fuente: {source} | ventanas de prueba: {len(examples)} | checkpoint: {checkpoint}")

    model = load_model(checkpoint)
    signals = np.stack([e.signal for e in examples])
    x = to_tensor(signals)
    logits = predict_logits(model, x)
    t_star, alerted = decision_sample(logits, args.threshold)
    outcomes = [classify(e, bool(a), int(t)) for e, a, t in zip(examples, alerted, t_star)]
    max_prob = (1 / (1 + np.exp(-logits[:, WARMUP_SAMPLES:]))).max(axis=1)
    snr = np.array([snr_value(e) for e in examples])
    counts = {k: outcomes.count(k) for k in ("TP", "FN", "FP", "TN")}
    print(f"[*] Resultados en prueba al umbral {args.threshold}: {counts}")

    # ---- Explicaciones locales (IG + oclusión) ----
    cases = select_cases(examples, outcomes, max_prob, snr, args.threshold)
    idx = np.array([i for _, i in cases])
    ig_local, _ = integrated_gradients(model, x[idx], t_star[idx], n_steps=args.n_steps)
    occ_local = occlusion(model, x[idx], t_star[idx])
    # Para el gráfico se necesitan arreglos indexados por ejemplo: se rellenan solo los casos.
    ig_map = {int(i): ig_local[k] for k, i in enumerate(idx)}
    occ_map = {int(i): occ_local[k] for k, i in enumerate(idx)}
    plot_local(
        cases,
        examples,
        logits,
        _Indexed(ig_map),
        _Indexed(occ_map),
        t_star,
        outcomes,
        snr,
        args.threshold,
        out / f"xai_local{suffix}.png",
    )
    local_rows = [
        {
            "caso": label,
            "indice": i,
            "resultado": outcomes[i],
            "p_max": round(float(max_prob[i]), 4),
            "t_decision_s": round(int(t_star[i]) / SAMPLE_RATE, 2),
            "p_real_s": None if examples[i].p_arrival is None else round(examples[i].p_arrival / SAMPLE_RATE, 2),
            "snr_db": None if not np.isfinite(snr[i]) else round(float(snr[i]), 2),
        }
        for label, i in cases
    ]

    # ---- Explicación global (eventos TP) ----
    tp = [i for i, o in enumerate(outcomes) if o == "TP"]
    tp = list(rng.permutation(tp)[: args.n_global])
    xg, tg = x[tp], t_star[tp]
    p_arr = np.array([examples[i].p_arrival for i in tp])
    ig_g, rel_delta = integrated_gradients(model, xg, tg, n_steps=args.n_steps)
    occ_g = occlusion(model, xg, tg)
    prof = temporal_profile(ig_g)

    lo, hi = -300, 400  # de -3 s a +4 s respecto a la P real
    aligned = aligned_profile(prof, p_arr, lo, hi)
    channel_share = np.abs(ig_g).sum(axis=(0, 2))
    channel_share = (channel_share / channel_share.sum()).tolist()

    # Uso de información futura: fracción de |IG| en muestras posteriores a t*.
    after = np.array([prof[k, tg[k] + 1 :].sum() / prof[k].sum() for k in range(len(tp))])
    reach = np.array([np.flatnonzero(prof[k] > 0.01 * prof[k].max()) for k in range(len(tp))], dtype=object)
    future_reach_s = np.array([(r.max() - tg[k]) / SAMPLE_RATE for k, r in enumerate(reach)])
    near_p = np.array(
        [prof[k, max(0, p - 100) : p + 100].sum() / prof[k].sum() for k, p in enumerate(p_arr)]
    )  # masa en P +- 1 s
    pre_noise = np.array([prof[k, WARMUP_SAMPLES : max(WARMUP_SAMPLES, p - 100)].sum() / prof[k].sum() for k, p in enumerate(p_arr)])

    # ---- Comparación IG vs oclusión, fidelidad y comprobaciones de cordura ----
    occ_prof = np.clip(occ_g.sum(axis=1), 0, None)
    agree = mean_spearman(prof, occ_prof, tg)
    scores = {"Integrated Gradients": window_scores(ig_g), "Oclusión": window_scores(occ_g)}
    curves = deletion_curves(model, xg, tg, scores, rng)

    # Sensibilidad a la línea base: ceros frente a una ventana de ruido de la propia partición de prueba.
    noise_idx = [i for i, e in enumerate(examples) if not e.label]
    base_pick = rng.choice(noise_idx, size=len(tp))
    ig_noise, _ = integrated_gradients(model, xg, tg, n_steps=args.n_steps, baseline=x[base_pick])
    baseline_corr = mean_spearman(prof, temporal_profile(ig_noise), tg)

    # Prueba de aleatorización del modelo (Adebayo et al., 2018): los pesos aleatorios deben cambiar la explicación.
    torch.manual_seed(args.seed)
    random_model = build_model("conv1d", **model.init_kwargs).eval()
    n_rand = min(60, len(tp))
    ig_rand, _ = integrated_gradients(random_model, xg[:n_rand], tg[:n_rand], n_steps=args.n_steps)
    rand_corr = mean_spearman(prof[:n_rand], temporal_profile(ig_rand), tg[:n_rand])
    # La oclusión mide el efecto sobre el logit del modelo, así que debería ser más sensible a los pesos.
    occ_rand = np.clip(occlusion(random_model, xg[:n_rand], tg[:n_rand]).sum(axis=1), 0, None)
    occ_rand_corr = mean_spearman(occ_prof[:n_rand], occ_rand, tg[:n_rand])
    # IG con línea base cero es entrada x gradiente medio: ¿cuánto se parece simplemente a |señal|?
    amp = np.abs(xg.numpy()).sum(axis=1)
    amp_corr = mean_spearman(prof, amp, tg)
    amp_corr_occ = mean_spearman(occ_prof, amp, tg)

    # Sensibilidad al número de pasos de IG.
    ig_half, _ = integrated_gradients(model, xg[:n_rand], tg[:n_rand], n_steps=max(8, args.n_steps // 4))
    steps_corr = mean_spearman(prof[:n_rand], temporal_profile(ig_half), tg[:n_rand])

    causal = causal_check(model, xg, tg, args.threshold)
    # Costo por explicación (CPU, una ventana a la vez; el primer cálculo se descarta como calentamiento).
    cost_n = min(10, len(tp))
    integrated_gradients(model, xg[:1], tg[:1], n_steps=args.n_steps)
    started = time.perf_counter()
    for k in range(cost_n):
        integrated_gradients(model, xg[k : k + 1], tg[k : k + 1], n_steps=args.n_steps)
    ig_ms = (time.perf_counter() - started) / cost_n * 1000
    started = time.perf_counter()
    for k in range(cost_n):
        occlusion(model, xg[k : k + 1], tg[k : k + 1])
    occ_ms = (time.perf_counter() - started) / cost_n * 1000
    summary = {
        "source": source,
        "checkpoint": checkpoint,
        "seed": args.seed,
        "threshold": args.threshold,
        "n_test_windows": len(examples),
        "test_outcomes": counts,
        "ig": {"baseline": "ceros", "n_steps": args.n_steps, "internal_batch": args.n_steps},
        "occlusion": {"window_samples": OCC_WINDOW, "stride_samples": OCC_STRIDE, "baseline": 0.0},
        "n_global_events": len(tp),
        "ig_completeness_rel_error_mean": float(np.mean(rel_delta)),
        "ig_completeness_rel_error_p95": float(np.percentile(rel_delta, 95)),
        "mass_within_1s_of_P": float(np.mean(near_p)),
        "mass_before_P_minus_1s": float(np.mean(pre_noise)),
        "mass_after_decision_sample": float(np.mean(after)),
        "future_reach_after_decision_s_mean": float(np.mean(future_reach_s)),
        "channel_share_ZNE": dict(zip(CHANNELS, channel_share)),
        "ig_vs_occlusion_spearman": agree,
        "ig_zero_vs_noise_baseline_spearman": baseline_corr,
        "ig_trained_vs_random_model_spearman": rand_corr,
        "ig_steps_sensitivity_spearman": steps_corr,
        "occlusion_trained_vs_random_model_spearman": occ_rand_corr,
        "ig_vs_signal_amplitude_spearman": amp_corr,
        "occlusion_vs_signal_amplitude_spearman": amp_corr_occ,
        "cost_ms_per_explanation_cpu": {"integrated_gradients": ig_ms, "occlusion": occ_ms},
        "mean_decision_minus_true_P_s": float(np.mean((tg - p_arr) / SAMPLE_RATE)),
        "alerts_kept_if_future_hidden_until": causal,
        "deletion_logit_drop": curves,
        "local_cases": local_rows,
    }
    (out / f"xai_summary{suffix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lags = np.arange(lo, hi) / SAMPLE_RATE
    plot_global(np.nanmean(aligned, axis=0), lags, channel_share, curves, out / f"xai_global{suffix}.png")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("local_cases", "deletion_logit_drop")}, indent=2))
    print(f"[+] Figuras y tablas en {out}/")


class _Indexed:
    """Adaptador: permite ``attr[i]`` por índice de ejemplo cuando solo se calcularon algunos casos."""

    def __init__(self, data: dict[int, np.ndarray]) -> None:
        self.data = data

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[int(i)]


if __name__ == "__main__":
    main()
