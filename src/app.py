"""Sistema de Alerta Sísmica Temprana (AlertaP) - Dashboard Profesional.

CC3092 Deep Learning | Universidad del Valle de Guatemala
Detección en tiempo real de onda P antes de la llegada de onda S.
Diseño científico, responsivo y sin elementos informales.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Asegurar importación de módulos locales
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seismic import (
    DeepPhaseDetector,
    PyTorchPhaseDetector,
    SignalExample,
    generate_example,
    sta_lta_detect,
    _features,
    _trailing_mean,
)

# ---------------------------------------------------------------------------
# Configuración Principal de la Página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AlertaP | Sistema de Alerta Sísmica Temprana",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Rutas de recursos
SYNTH_CKPT = Path("checkpoints/best_model.pt")
STEAD_CKPT = Path("checkpoints_stead/best_model.pt")
DEMO_SAMPLES = Path("results/demo_samples.npz")
METRICS_SYNTH = Path("results/metrics.json")
METRICS_STEAD = Path("results/metrics_stead.json")
HISTORY_SYNTH = Path("checkpoints/history.csv")
HISTORY_STEAD = Path("checkpoints_stead/history.csv")


# ---------------------------------------------------------------------------
# Carga de Modelos y Datos en Caché
# ---------------------------------------------------------------------------
@st.cache_resource
def load_deep_phase() -> DeepPhaseDetector:
    return DeepPhaseDetector()


@st.cache_resource
def load_pytorch_model(path: str) -> PyTorchPhaseDetector:
    return PyTorchPhaseDetector(checkpoint_path=path, device="auto")


@st.cache_data
def load_demo_samples():
    if not DEMO_SAMPLES.exists():
        return None
    return dict(np.load(str(DEMO_SAMPLES)))


@st.cache_data
def load_json(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Barra Lateral: Configuración y Parámetros Operativos
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 4px 0 16px 0; border-bottom: 1px solid rgba(128, 128, 128, 0.2); margin-bottom: 16px;">
        <span style="font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b;">
            CENTRO DE CONTROL
        </span>
        <h2 style="margin: 4px 0 0 0; font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">
            AlertaP EEWS
        </h2>
        <span style="font-size: 12px; color: #94a3b8;">Sistema de Alerta Temprana</span>
    </div>
    """,
    unsafe_allow_html=True,
)

theme_mode = st.sidebar.selectbox(
    "Tema visual de interfaz:",
    ["Oscuro (Dark Industrial)", "Claro (Clean Scientific)"],
    index=0,
)
is_dark = "Oscuro" in theme_mode

st.sidebar.markdown(
    "<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b;'>Modelo Detector</span>",
    unsafe_allow_html=True,
)
options = []
if STEAD_CKPT.exists():
    options.append("ConvPhaseNet1D (STEAD Real - Recomendado)")
if SYNTH_CKPT.exists():
    options.append("ConvPhaseNet1D (Sintético)")
options.append("DeepPhase (MLP Clásico)")

model_choice = st.sidebar.selectbox("Arquitectura activa:", options, label_visibility="collapsed")

if "STEAD Real" in model_choice:
    detector = load_pytorch_model(str(STEAD_CKPT))
    default_thresh = 0.50
    model_tag = "ConvPhaseNet1D-STEAD"
elif "Sintético" in model_choice:
    detector = load_pytorch_model(str(SYNTH_CKPT))
    default_thresh = 0.85
    model_tag = "ConvPhaseNet1D-Synth"
else:
    detector = load_deep_phase()
    default_thresh = 0.65
    model_tag = "DeepPhase-MLP"

device_name = str(getattr(detector, "device", "CPU")).upper()
st.sidebar.markdown(
    f"""
    <div style="background: {'rgba(59, 130, 246, 0.1)' if is_dark else 'rgba(37, 99, 235, 0.08)'}; 
                border-left: 3px solid #3b82f6; padding: 8px 12px; border-radius: 4px; margin: 8px 0 16px 0;">
        <div style="font-size: 11px; color: #64748b; font-weight: 600;">HARDWARE DE INFERENCIA</div>
        <div style="font-size: 13px; font-weight: 600; color: {'#93c5fd' if is_dark else '#1d4ed8'}; font-family: monospace;">
            {device_name}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b;'>Fuente de Señales</span>",
    unsafe_allow_html=True,
)
has_samples = DEMO_SAMPLES.exists()
source_choice = st.sidebar.radio(
    "Fuente de datos:",
    ["Sismogramas reales STEAD (Test Set)", "Simulador sísmico sintético"],
    index=0 if has_samples else 1,
    label_visibility="collapsed",
)

st.sidebar.markdown(
    "<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b;'>Umbral de Decisión</span>",
    unsafe_allow_html=True,
)
threshold = st.sidebar.slider(
    "Umbral de probabilidad P:",
    min_value=0.20,
    max_value=0.95,
    value=default_thresh,
    step=0.05,
    help="Umbral de corte de la función de activación para emitir la alerta temprana.",
)
st.sidebar.caption(f"Umbral óptimo validado para {model_tag}: **{default_thresh:.2f}**")

# Baseline STA/LTA threshold
st.sidebar.markdown(
    "<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b;'>Umbral STA/LTA (Baseline)</span>",
    unsafe_allow_html=True,
)
stalta_threshold = st.sidebar.slider(
    "Cociente STA/LTA:",
    min_value=1.5,
    max_value=6.0,
    value=2.5 if "STEAD" in model_choice else 2.0,
    step=0.5,
    help="Cociente entre promedio de corto plazo (0.25 s) y largo plazo (3.0 s).",
)


# ---------------------------------------------------------------------------
# Inyección de Estilos CSS Adaptativos (Modo Claro / Modo Oscuro)
# ---------------------------------------------------------------------------
if is_dark:
    theme_css = """
    :root {
        --bg-app: #090d16;
        --bg-card: #111827;
        --border-card: #1f2937;
        --text-main: #f8fafc;
        --text-sub: #94a3b8;
        --accent-blue: #38bdf8;
        --badge-bg: #1e293b;
        --badge-text: #e2e8f0;
    }
    """
else:
    theme_css = """
    :root {
        --bg-app: #f8fafc;
        --bg-card: #ffffff;
        --border-card: #e2e8f0;
        --text-main: #0f172a;
        --text-sub: #64748b;
        --accent-blue: #0284c7;
        --badge-bg: #f1f5f9;
        --badge-text: #334155;
    }
    """

st.markdown(
    f"""
    <style>
    {theme_css}
    
    /* Tipografía y márgenes generales */
    .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    
    /* Contenedores de tarjetas de métricas */
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin: 16px 0;
    }}
    
    .kpi-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    
    .kpi-card:hover {{
        border-color: var(--accent-blue);
    }}
    
    .kpi-title {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-sub);
        margin-bottom: 6px;
    }}
    
    .kpi-value {{
        font-size: 24px;
        font-weight: 700;
        color: var(--text-main);
        font-family: "JetBrains Mono", "SF Mono", Consolas, monospace;
        line-height: 1.2;
    }}
    
    .kpi-subtext {{
        font-size: 12px;
        color: var(--text-sub);
        margin-top: 6px;
    }}
    
    /* Badges de Estado */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    .status-alert {{
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    
    .status-nominal {{
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}
    
    .status-warning {{
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }}
    
    .status-info {{
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }}
    
    /* Botones primarios */
    button[kind="primary"] {{
        background-color: #2563eb !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        border-radius: 6px !important;
        padding: 0.55rem 1rem !important;
    }}
    
    button[kind="primary"]:hover {{
        background-color: #1d4ed8 !important;
    }}
    
    /* Tablas y pestañas */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        font-size: 13px;
        padding: 10px 18px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Encabezado Institucional
# ---------------------------------------------------------------------------
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 11px; font-weight: 700; letter-spacing: 1.5px; color: #3b82f6; text-transform: uppercase;">
                SISTEMA DE MITIGACIÓN Y ALERTA TEMPRANA
            </div>
            <h1 style="margin: 2px 0 6px 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; color: {'#f8fafc' if is_dark else '#0f172a'};">
                AlertaP: Monitor Sísmico de Onda P
            </h1>
            <p style="margin: 0; font-size: 13px; color: {'#94a3b8' if is_dark else '#475569'}; max-width: 820px; line-height: 1.4;">
                Detección automática de fase primaria (P) previa a la llegada de onda secundaria (S) en ventanas triaxiales de 100 Hz.
                Cuantificación de anticipación operativa para parada industrial y protección de infraestructura crítica.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_head2:
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 10px;">
            <div style="display: inline-block; text-align: left; background: {'#111827' if is_dark else '#f1f5f9'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 6px; padding: 8px 12px;">
                <div style="font-size: 10px; color: #64748b; font-weight: 700;">ESTADO OPERATIVO</div>
                <div style="font-size: 12px; font-weight: 700; color: #10b981; letter-spacing: 0.5px;">EN LÍNEA | NORMAL</div>
                <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Tasa: 100 Hz | 3 Canales</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin: 12px 0 20px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.2);'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Organización por Pestañas Profesionales
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Monitor en Tiempo Real",
    "Rendimiento del Modelo",
    "Curvas de Valor y Sensibilidad",
    "Historial de Convergencia",
    "Especificaciones Técnicas",
])


# ===========================================================================
# PESTAÑA 1: MONITOR EN TIEMPO REAL
# ===========================================================================
with tabs[0]:
    demo_data = load_demo_samples() if source_choice.startswith("Sismogramas") else None

    # Panel de Selección de Señal
    if demo_data is not None:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #64748b; margin-bottom: 8px;'>PARÁMETROS DE LA SEÑAL DE ENTRADA</div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns([1.5, 2.5, 1.5])
        
        with col_c1:
            tipo_filtro = st.radio(
                "Naturaleza del registro:",
                ["Sismo Real (Evento)", "Ruido Sísmico Ambiental"],
                horizontal=False,
            )
        
        is_event_type = 1 if "Sismo Real" in tipo_filtro else 0
        indices_disponibles = np.where(demo_data["labels"] == is_event_type)[0]

        with col_c2:
            ejemplo_idx = st.selectbox(
                f"Sismograma registrado ({len(indices_disponibles)} disponibles):",
                indices_disponibles,
                format_func=lambda i: (
                    f"Registro #{i+1} | SNR: {demo_data['snr'][i]:.1f} dB"
                    if is_event_type
                    else f"Registro de Ruido #{i+1} (Sin arribo P/S)"
                ),
            )

        sig = demo_data["signals"][ejemplo_idx]
        p_arr = int(demo_data["p"][ejemplo_idx]) if is_event_type and demo_data["p"][ejemplo_idx] >= 0 else None
        s_arr = int(demo_data["s"][ejemplo_idx]) if is_event_type and demo_data["s"][ejemplo_idx] >= 0 else None
        snr_val = float(demo_data["snr"][ejemplo_idx]) if not np.isnan(demo_data["snr"][ejemplo_idx]) else 0.0
        example = SignalExample(sig, p_arr, s_arr, is_event_type, {"snr_db": snr_val})

        with col_c3:
            st.markdown(
                f"""
                <div style="background: {'#111827' if is_dark else '#f8fafc'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 6px; padding: 10px 14px; margin-top: 14px;">
                    <div style="font-size: 11px; color: #64748b; font-weight: 600;">METADATOS DEL REGISTRO</div>
                    <div style="font-size: 12px; margin-top: 4px; color: {'#cbd5e1' if is_dark else '#334155'};">
                        Tipo: <strong>{'Terremoto' if is_event_type else 'Ruido'}</strong><br>
                        Relación Señal/Ruido: <strong>{snr_val:.1f} dB</strong><br>
                        Llegada P Catálogo: <strong>{f'{p_arr/100:.2f} s' if p_arr else 'N/A'}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #64748b; margin-bottom: 8px;'>GENERADOR SINTÉTICO CONTROLADO</div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns([1.5, 2, 2.5])
        with col_s1:
            event = st.radio("Condición:", ["Presencia de Onda P/S", "Ruido Ambiental Sintético"]) == "Presencia de Onda P/S"
        with col_s2:
            seed = st.number_input("Semilla pseudoaleatoria (RNG):", min_value=1, value=23361, step=1)
        with col_s3:
            noise_scale = st.slider("Amplitud del ruido Gaussiano base:", min_value=0.10, max_value=0.60, value=0.25, step=0.05)
        
        rng = np.random.default_rng(seed)
        example = generate_example(rng, event=event, noise_scale=noise_scale)

    # Botón de Ejecución
    analyze_col1, analyze_col2 = st.columns([1, 4])
    with analyze_col1:
        run_analysis = st.button("Ejecutar Inferencia", type="primary", use_container_width=True)
    with analyze_col2:
        st.markdown(
            f"<div style='padding-top: 8px; font-size: 12px; color: #64748b;'>Procesa la señal con {model_tag} y compara simultáneamente con STA/LTA (umbral {stalta_threshold}).</div>",
            unsafe_allow_html=True,
        )

    # Inferencia
    # Si no se ha presionado el botón pero es la primera carga, se procesa por defecto
    deep = detector.detect(example.signal, threshold=threshold)
    classic = sta_lta_detect(example.signal, threshold=stalta_threshold)

    # Extracción de la curva continua de activación
    p_curve = None
    if isinstance(detector, PyTorchPhaseDetector):
        import torch
        with torch.no_grad():
            if detector.model_name in ("conv1d", "phasenet", "convphasenet", "convphasenet1d"):
                x = torch.from_numpy(np.ascontiguousarray(example.signal.T, dtype=np.float32)).unsqueeze(0).to(detector.device)
            else:
                feat = _features(example.signal).astype(np.float32)
                x = torch.from_numpy(feat).unsqueeze(0).to(detector.device)
            logits = detector.model(x)
            p_curve = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    elif isinstance(detector, DeepPhaseDetector):
        p_curve = detector.model.predict_proba(_features(example.signal))[:, 1]

    # STA/LTA continuous ratio curve
    energy = np.sqrt(np.mean(example.signal**2, axis=1)) + 1e-8
    sta_trace = _trailing_mean(energy, int(0.25 * 100.0))
    lta_trace = _trailing_mean(energy, int(3.0 * 100.0))
    stalta_ratio = sta_trace / lta_trace

    # Tarjetas de Diagnóstico y Telemetría
    time_hz = 100.0
    n_samples = len(example.signal)
    time_axis = np.arange(n_samples) / time_hz

    # Evaluación operativa de la alerta
    is_true_positive = False
    is_false_alarm = False
    is_missed = False
    is_late_alert = False
    advance_sec = 0.0

    if example.label:  # Es evento
        if deep.detected:
            if example.s_arrival is not None and deep.sample is not None:
                if deep.sample < example.s_arrival:
                    is_true_positive = True
                    advance_sec = (example.s_arrival - deep.sample) / time_hz
                else:
                    is_late_alert = True
            else:
                is_true_positive = True
        else:
            is_missed = True
    else:  # Es ruido
        if deep.detected:
            is_false_alarm = True

    # Badges y textos de estado
    if is_true_positive:
        status_badge_html = '<span class="status-badge status-alert">ALERTA CONFIRMADA</span>'
        status_desc = f"Onda P detectada {advance_sec:.2f} s antes de la onda S destructiva."
    elif is_late_alert:
        status_badge_html = '<span class="status-badge status-warning">ALERTA TARDÍA</span>'
        status_desc = "Disparo emitido posterior a la llegada de la onda destructiva S."
    elif is_false_alarm:
        status_badge_html = '<span class="status-badge status-warning">FALSA ALARMA</span>'
        status_desc = "Disparo erróneo generado sobre señal de ruido sísmico ambiental."
    elif is_missed:
        status_badge_html = '<span class="status-badge status-warning">NO DETECTADO</span>'
        status_desc = "El evento no superó el umbral configurado de probabilidad."
    else:
        status_badge_html = '<span class="status-badge status-nominal">ESTADO NOMINAL</span>'
        status_desc = "Ventana de ruido ambiental identificada y filtrada correctamente."

    # Render de tarjetas KPI
    picking_err_str = "N/A"
    if deep.detected and example.p_arrival is not None and deep.sample is not None:
        err_ms = abs(deep.sample - example.p_arrival) / time_hz * 1000.0
        picking_err_str = f"{err_ms:.1f} ms"

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">DIAGNÓSTICO OPERATIVO</div>
                <div style="margin: 6px 0;">{status_badge_html}</div>
                <div class="kpi-subtext">{status_desc}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">ANTICIPACIÓN OPERATIVA (Δt)</div>
                <div class="kpi-value">{f'{advance_sec:.2f} s' if is_true_positive else '0.00 s'}</div>
                <div class="kpi-subtext">Margen de maniobra antes de onda S</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">PROBABILIDAD MÁXIMA P</div>
                <div class="kpi-value">{deep.probability:.1%}</div>
                <div class="kpi-subtext">Corte activo: {threshold:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">LATENCIA DE INFERENCIA</div>
                <div class="kpi-value">{deep.latency_ms:.2f} ms</div>
                <div class="kpi-subtext">Ejecución en {device_name}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">PRECISIÓN DE PICKING</div>
                <div class="kpi-value">{picking_err_str}</div>
                <div class="kpi-subtext">Desviación vs catálogo manual</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Gráfico Multi-Canal Profesional (Matplotlib Estilizado)
    # -----------------------------------------------------------------------
    # Configuración de colores según tema
    if is_dark:
        fig_face = "#0b0f19"
        ax_face = "#111827"
        grid_col = "#1f293d"
        text_col = "#e2e8f0"
        subtext_col = "#94a3b8"
        spine_col = "#334155"
        col_z = "#38bdf8"       # Cyan
        col_n = "#818cf8"       # Indigo
        col_e = "#fb923c"       # Orange
        col_p = "#34d399"       # Emerald
        col_s = "#f87171"       # Red
        col_alert_deep = "#c084fc" # Purple
        col_alert_sta = "#facc15"  # Gold
    else:
        fig_face = "#ffffff"
        ax_face = "#f8fafc"
        grid_col = "#e2e8f0"
        text_col = "#0f172a"
        subtext_col = "#475569"
        spine_col = "#cbd5e1"
        col_z = "#0284c7"
        col_n = "#4f46e5"
        col_e = "#ea580c"
        col_p = "#059669"
        col_s = "#dc2626"
        col_alert_deep = "#7c3aed"
        col_alert_sta = "#d97706"

    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})
    fig.patch.set_facecolor(fig_face)

    channel_defs = [
        ("Canal Vertical (Z - Movimiento Compresional)", 0, col_z),
        ("Canal Norte-Sur (N - Componente Horizontal)", 1, col_n),
        ("Canal Este-Oeste (E - Componente Horizontal)", 2, col_e),
    ]

    for i, (ch_title, ch_idx, trace_color) in enumerate(channel_defs):
        ax = axes[i]
        ax.set_facecolor(ax_face)
        ax.plot(time_axis, example.signal[:, ch_idx], color=trace_color, linewidth=0.8, label=ch_title)
        
        # Líneas de referencia sísmica
        if example.p_arrival is not None:
            ax.axvline(example.p_arrival / time_hz, color=col_p, linestyle="--", linewidth=1.4, alpha=0.9, label="Onda P real")
        if example.s_arrival is not None:
            ax.axvline(example.s_arrival / time_hz, color=col_s, linestyle="--", linewidth=1.4, alpha=0.9, label="Onda S real")
        if deep.sample is not None:
            ax.axvline(deep.sample / time_hz, color=col_alert_deep, linestyle="-", linewidth=1.8, label=f"Pick {model_tag}")
        if classic.sample is not None:
            ax.axvline(classic.sample / time_hz, color=col_alert_sta, linestyle=":", linewidth=1.5, label="Pick STA/LTA")

        ax.set_ylabel("Amplitud", fontsize=9, color=text_col)
        ax.tick_params(colors=subtext_col, labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.5, color=grid_col)
        for spine in ax.spines.values():
            spine.set_color(spine_col)
        
        # Título interno discreto
        ax.text(
            0.01, 0.85, ch_title,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="600",
            color=text_col,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=ax_face, edgecolor=spine_col, alpha=0.85),
        )

    # 4to Subplot: Función de Decisión Temporal (Probabilidad P / Ratio STA/LTA)
    ax_dec = axes[3]
    ax_dec.set_facecolor(ax_face)

    if p_curve is not None:
        ax_dec.plot(time_axis, p_curve, color=col_alert_deep, linewidth=1.6, label=f"P(t) {model_tag}")
        ax_dec.axhline(threshold, color=col_alert_deep, linestyle=":", linewidth=1.2, label=f"Umbral Modelo ({threshold:.2f})")
        if deep.sample is not None:
            ax_dec.scatter([deep.sample / time_hz], [p_curve[deep.sample]], color=col_alert_deep, s=50, zorder=5)

    # Gráfica secundaria o superpuesta para STA/LTA normalizado
    # Mostramos ratio STA/LTA normalizado por su umbral para escala comparable
    ax_dec_sta = ax_dec.twinx()
    ax_dec_sta.plot(time_axis, stalta_ratio, color=col_alert_sta, linestyle="-.", linewidth=1.0, alpha=0.7, label="Ratio STA/LTA")
    ax_dec_sta.axhline(stalta_threshold, color=col_alert_sta, linestyle=":", linewidth=1.0, alpha=0.8, label=f"Umbral STA/LTA ({stalta_threshold:.1f})")
    ax_dec_sta.set_ylabel("Ratio STA/LTA", fontsize=8, color=col_alert_sta)
    ax_dec_sta.tick_params(colors=col_alert_sta, labelsize=8)
    ax_dec_sta.spines["right"].set_color(col_alert_sta)

    # Marcadores verticales también en la decisión
    if example.p_arrival is not None:
        ax_dec.axvline(example.p_arrival / time_hz, color=col_p, linestyle="--", linewidth=1.4, alpha=0.8)
    if example.s_arrival is not None:
        ax_dec.axvline(example.s_arrival / time_hz, color=col_s, linestyle="--", linewidth=1.4, alpha=0.8)

    ax_dec.set_ylim(-0.05, 1.05)
    ax_dec.set_ylabel("Probabilidad P(t)", fontsize=9, color=text_col)
    ax_dec.set_xlabel("Tiempo transcurrido (segundos)", fontsize=10, color=text_col)
    ax_dec.tick_params(colors=subtext_col, labelsize=8)
    ax_dec.grid(True, linestyle="--", alpha=0.5, color=grid_col)
    for spine in ax_dec.spines.values():
        spine.set_color(spine_col)

    # Leyenda unificada
    lines_1, labels_1 = axes[0].get_legend_handles_labels()
    lines_2, labels_2 = ax_dec.get_legend_handles_labels()
    lines_3, labels_3 = ax_dec_sta.get_legend_handles_labels()
    all_lines = lines_1[1:] + lines_2 + lines_3
    all_labels = labels_1[1:] + labels_2 + labels_3
    
    fig.legend(
        all_lines, all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        fontsize=8,
        frameon=True,
        facecolor=ax_face,
        edgecolor=spine_col,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)

    # Resumen Operativo de Mitigación
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    col_act1, col_act2 = st.columns([2, 1])
    with col_act1:
        st.markdown(
            f"""
            <div style="background: {'#111827' if is_dark else '#ffffff'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 8px; padding: 14px 18px;">
                <div style="font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.8px; text-transform: uppercase;">
                    PROTOCOLO DE MITIGACIÓN SEGÚN ANTICIPACIÓN DISPONIBLE
                </div>
                <div style="margin-top: 8px; font-size: 13px; color: {'#cbd5e1' if is_dark else '#334155'}; line-height: 1.5;">
                    {'<strong>Nivel de Alerta Crítico:</strong> Margen suficiente (' + f'{advance_sec:.2f} s' + ') para ejecución automática de: cierre de válvulas de gas natural, paro de emergencia en trenes suburbanos, detención de ascensores en piso inmediato y puesta a salvo de personal en líneas de ensamblaje.' if advance_sec >= 2.0 else ('<strong>Margen Marginal:</strong> Menos de 2 segundos de anticipación útil. Se activan señales de evacuación y corte rápido.' if advance_sec > 0 else '<strong>Sin Alerta:</strong> Señal dentro de los parámetros de ruido basal o evento no detectado.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_act2:
        st.markdown(
            f"""
            <div style="background: {'#111827' if is_dark else '#ffffff'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 8px; padding: 14px 18px;">
                <div style="font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.8px; text-transform: uppercase;">
                    COMPARATIVA VS BASELINE
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: {'#cbd5e1' if is_dark else '#334155'};">
                    STA/LTA Alerta: <strong>{'DISPARADA' if classic.detected else 'NO ACTIVADA'}</strong><br>
                    STA/LTA Pick: <strong>{f'{classic.sample/100:.2f} s' if classic.sample else 'N/A'}</strong><br>
                    STA/LTA Latencia: <strong>{classic.latency_ms:.2f} ms</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# PESTAÑA 2: RENDIMIENTO Y MÉTRICAS DEL MODELO
# ===========================================================================
with tabs[1]:
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 18px; font-weight: 700;">Evaluación Comparativa de Desempeño</h3>
            <p style="margin: 4px 0 0 0; font-size: 13px; color: #64748b;">
                Resultados cuantitativos obtenidos sobre el conjunto de prueba independiente (partición por sismo, sin fuga de datos).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_file = METRICS_STEAD if METRICS_STEAD.exists() else METRICS_SYNTH
    metrics_data = load_json(str(metrics_file))

    if metrics_data:
        dl_res = metrics_data.get("DeepPhase", {})
        sta_res = metrics_data.get("STA/LTA", {})

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                <div style="background: {'#111827' if is_dark else '#ffffff'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 8px; padding: 16px;">
                    <div style="font-size: 11px; font-weight: 700; color: #3b82f6; text-transform: uppercase;">MODELO DEEP LEARNING (CONVPHASENET1D)</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
                        <div>
                            <div style="font-size: 11px; color: #64748b;">F1-SCORE</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{dl_res.get('f1', 0):.4f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">PRECISIÓN</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{dl_res.get('precision', 0):.4f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">RECALL (SENSIBILIDAD)</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{dl_res.get('recall', 0):.4f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">ERROR DE PICKING</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{dl_res.get('mean_picking_error_s', 0)*1000:.1f} ms</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">LATENCIA MEDIA</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{dl_res.get('mean_latency_ms', 0):.2f} ms</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">VALOR ECONÓMICO NETO</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace; color: #10b981;">${dl_res.get('business_value', 0):,.0f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            st.markdown(
                f"""
                <div style="background: {'#111827' if is_dark else '#ffffff'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 8px; padding: 16px;">
                    <div style="font-size: 11px; font-weight: 700; color: #eab308; text-transform: uppercase;">BASELINE CLÁSICO (STA/LTA)</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
                        <div>
                            <div style="font-size: 11px; color: #64748b;">F1-SCORE</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{sta_res.get('f1', 0):.4f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">PRECISIÓN</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{sta_res.get('precision', 0):.4f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">RECALL (SENSIBILIDAD)</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{sta_res.get('recall', 0):.4f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">ERROR DE PICKING</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{sta_res.get('mean_picking_error_s', 0)*1000:.1f} ms</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">ANTICIPACIÓN MEDIA</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace;">{sta_res.get('mean_advance_s', 0):.2f} s</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #64748b;">VALOR ECONÓMICO NETO</div>
                            <div style="font-size: 22px; font-weight: 700; font-family: monospace; color: #10b981;">${sta_res.get('business_value', 0):,.0f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Matriz de Contingencia (TP, FP, FN)
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;'>Distribución de Aciertos y Errores (Test Set)</div>", unsafe_allow_html=True)
        
        df_conf = pd.DataFrame({
            "Métrica de Validación": [
                "Verdaderos Positivos (Alertas Útiles)",
                "Falsas Alarmas (Disparos en Ruido)",
                "Eventos No Detectados (Omitidos)",
                "Valor Económico Generado",
            ],
            "Deep Learning": [
                f"{dl_res.get('true_positives', 0):,}",
                f"{dl_res.get('false_positives', 0):,}",
                f"{dl_res.get('false_negatives', 0):,}",
                f"${dl_res.get('business_value', 0):,.0f}",
            ],
            "STA/LTA": [
                f"{sta_res.get('true_positives', 0):,}",
                f"{sta_res.get('false_positives', 0):,}",
                f"{sta_res.get('false_negatives', 0):,}",
                f"${sta_res.get('business_value', 0):,.0f}",
            ],
        })
        st.dataframe(df_conf, use_container_width=True, hide_index=True)


# ===========================================================================
# PESTAÑA 3: CURVAS DE VALOR Y SENSIBILIDAD
# ===========================================================================
with tabs[2]:
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 18px; font-weight: 700;">Optimización de Umbral y Modelo de Valor Económico</h3>
            <p style="margin: 4px 0 0 0; font-size: 13px; color: #64748b;">
                Relación explícita entre parámetros sísmicos y costo-beneficio operativo ($). Selección del punto de operación óptimo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_file = METRICS_STEAD if METRICS_STEAD.exists() else METRICS_SYNTH
    metrics_data = load_json(str(metrics_file))

    if metrics_data and "value_curve_test" in metrics_data:
        dl_curve = metrics_data["value_curve_test"].get("DeepPhase", [])
        sta_curve = metrics_data["value_curve_test"].get("STA/LTA", [])

        if dl_curve:
            df_dl = pd.DataFrame(dl_curve)
            
            fig_val, (ax_v1, ax_v2) = plt.subplots(1, 2, figsize=(13, 4.5))
            fig_val.patch.set_facecolor(fig_face)
            ax_v1.set_facecolor(ax_face)
            ax_v2.set_facecolor(ax_face)

            # Curva 1: Valor Económico vs Umbral
            ax_v1.plot(df_dl["threshold"], df_dl["business_value"] / 1e6, marker="o", color=col_z, linewidth=2, label="ConvPhaseNet1D")
            opt_idx = df_dl["business_value"].idxmax()
            opt_thresh = df_dl.loc[opt_idx, "threshold"]
            opt_val = df_dl.loc[opt_idx, "business_value"] / 1e6
            ax_v1.scatter([opt_thresh], [opt_val], color=col_p, s=100, zorder=5, label=f"Óptimo ({opt_thresh:.2f})")

            ax_v1.set_title("Valor Neto vs Umbral de Decisión", fontsize=11, color=text_col, fontweight="600")
            ax_v1.set_xlabel("Umbral de Probabilidad", fontsize=9, color=text_col)
            ax_v1.set_ylabel("Valor Económico Neto (Millones USD)", fontsize=9, color=text_col)
            ax_v1.tick_params(colors=subtext_col, labelsize=8)
            ax_v1.grid(True, linestyle="--", alpha=0.5, color=grid_col)
            for spine in ax_v1.spines.values():
                spine.set_color(spine_col)
            ax_v1.legend(facecolor=ax_face, edgecolor=spine_col, fontsize=8)

            # Curva 2: Trade-off Precisión vs Recall
            ax_v2.plot(df_dl["threshold"], df_dl["precision"], marker="s", color=col_z, linewidth=1.8, label="Precisión")
            ax_v2.plot(df_dl["threshold"], df_dl["recall"], marker="^", color=col_s, linewidth=1.8, label="Recall")
            ax_v2.plot(df_dl["threshold"], df_dl["f1"], marker="d", color=col_p, linewidth=1.8, label="F1-Score")
            ax_v2.axvline(opt_thresh, color=col_p, linestyle=":", linewidth=1.5, label=f"Umbral Óptimo ({opt_thresh:.2f})")

            ax_v2.set_title("Compromiso Precisión vs Sensibilidad", fontsize=11, color=text_col, fontweight="600")
            ax_v2.set_xlabel("Umbral de Probabilidad", fontsize=9, color=text_col)
            ax_v2.set_ylabel("Métrica (0.0 - 1.0)", fontsize=9, color=text_col)
            ax_v2.tick_params(colors=subtext_col, labelsize=8)
            ax_v2.grid(True, linestyle="--", alpha=0.5, color=grid_col)
            for spine in ax_v2.spines.values():
                spine.set_color(spine_col)
            ax_v2.legend(facecolor=ax_face, edgecolor=spine_col, fontsize=8)

            fig_val.tight_layout()
            st.pyplot(fig_val)

        # Análisis de Sensibilidad
        if "sensitivity" in metrics_data:
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;'>Análisis de Sensibilidad de Costos (±20% y ±50%)</div>", unsafe_allow_html=True)
            df_sens = pd.DataFrame(metrics_data["sensitivity"])
            df_sens = df_sens.rename(columns={
                "parameter": "Parámetro Económico",
                "change_pct": "Variación (%)",
                "deep_value_at_default_threshold": "Valor Modelo ($)",
                "optimal_threshold": "Umbral Óptimo",
                "baseline_value": "Valor STA/LTA ($)",
                "advantage_vs_baseline": "Ventaja Neta ($)",
            })
            st.dataframe(df_sens, use_container_width=True, hide_index=True)


# ===========================================================================
# PESTAÑA 4: HISTORIAL DE ENTRENAMIENTO
# ===========================================================================
with tabs[3]:
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 18px; font-weight: 700;">Convergencia y Registro de Entrenamiento</h3>
            <p style="margin: 4px 0 0 0; font-size: 13px; color: #64748b;">
                Evolución por época de la función de pérdida (BCEWithLogitsLoss), métrica F1 y función de valor económico.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_file = HISTORY_STEAD if HISTORY_STEAD.exists() else HISTORY_SYNTH
    df_hist = load_csv(str(history_file))

    if df_hist is not None and not df_hist.empty:
        fig_hist, (ax_h1, ax_h2, ax_h3) = plt.subplots(1, 3, figsize=(14, 4.2))
        fig_hist.patch.set_facecolor(fig_face)
        for ax in (ax_h1, ax_h2, ax_h3):
            ax.set_facecolor(ax_face)
            ax.tick_params(colors=subtext_col, labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.5, color=grid_col)
            for spine in ax.spines.values():
                spine.set_color(spine_col)

        # Gráfico 1: Pérdidas Train vs Val
        ax_h1.plot(df_hist["epoch"], df_hist["train_loss"], color=col_z, label="Train Loss", linewidth=1.8)
        ax_h1.plot(df_hist["epoch"], df_hist["val_loss"], color=col_s, label="Val Loss", linewidth=1.8)
        ax_h1.set_title("Pérdida de Entrenamiento y Validación", fontsize=10, color=text_col, fontweight="600")
        ax_h1.set_xlabel("Época", fontsize=9, color=text_col)
        ax_h1.set_ylabel("Loss", fontsize=9, color=text_col)
        ax_h1.legend(facecolor=ax_face, edgecolor=spine_col, fontsize=8)

        # Gráfico 2: F1-Score en Validación
        ax_h2.plot(df_hist["epoch"], df_hist["val_f1"], color=col_p, label="Val F1", linewidth=1.8)
        ax_h2.set_title("Convergencia de Métrica F1", fontsize=10, color=text_col, fontweight="600")
        ax_h2.set_xlabel("Época", fontsize=9, color=text_col)
        ax_h2.set_ylabel("F1-Score", fontsize=9, color=text_col)
        ax_h2.legend(facecolor=ax_face, edgecolor=spine_col, fontsize=8)

        # Gráfico 3: Valor Económico en Validación
        if "val_business_value" in df_hist.columns:
            ax_h3.plot(df_hist["epoch"], df_hist["val_business_value"] / 1e6, color=col_alert_deep, label="Valor Neto ($M)", linewidth=1.8)
            ax_h3.set_title("Evolución de Valor Económico", fontsize=10, color=text_col, fontweight="600")
            ax_h3.set_xlabel("Época", fontsize=9, color=text_col)
            ax_h3.set_ylabel("Millones USD", fontsize=9, color=text_col)
            ax_h3.legend(facecolor=ax_face, edgecolor=spine_col, fontsize=8)

        fig_hist.tight_layout()
        st.pyplot(fig_hist)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;'>Registro Detallado por Época</div>", unsafe_allow_html=True)
        st.dataframe(df_hist.tail(15), use_container_width=True, hide_index=True)


# ===========================================================================
# PESTAÑA 5: ESPECIFICACIONES TÉCNICAS
# ===========================================================================
with tabs[4]:
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 18px; font-weight: 700;">Arquitectura del Sistema y Parámetros</h3>
            <p style="margin: 4px 0 0 0; font-size: 13px; color: #64748b;">
                Definición formal de la red neuronal convolucional, preprocesamiento y supuestos económicos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            f"""
            <div style="background: {'#111827' if is_dark else '#ffffff'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 8px; padding: 16px;">
                <div style="font-size: 11px; font-weight: 700; color: #3b82f6; text-transform: uppercase;">ARQUITECTURA CONVPHASENET1D</div>
                <ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: {'#cbd5e1' if is_dark else '#334155'}; line-height: 1.6;">
                    <li><strong>Entrada:</strong> Tensor triaxial [Batch, 3, 1200] (Canales Z, N, E normalizados a 100 Hz).</li>
                    <li><strong>Capas Convolucionales:</strong> Convolución 1D inicial (kernel 7, 32 filtros) seguida de bloques residuales dilatados (dilataciones 1, 2 y 4).</li>
                    <li><strong>Normalización y Regularización:</strong> BatchNorm1D + Dropout (0.10) en cada bloque residual.</li>
                    <li><strong>Salida:</strong> Proyección 1D a logits temporales con función Sigmoide P(t) ∈ [0, 1].</li>
                    <li><strong>Optimizador:</strong> AdamW con tasa de aprendizaje adaptativa (CosineAnnealingLR).</li>
                    <li><strong>Función de Pérdida:</strong> BCEWithLogitsLoss con peso positivo ponderado por desbalance temporal.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t2:
        st.markdown(
            f"""
            <div style="background: {'#111827' if is_dark else '#ffffff'}; border: 1px solid {'#1f2937' if is_dark else '#e2e8f0'}; border-radius: 8px; padding: 16px;">
                <div style="font-size: 11px; font-weight: 700; color: #10b981; text-transform: uppercase;">MODELO DE VALOR OPERATIVO (USD)</div>
                <ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: {'#cbd5e1' if is_dark else '#334155'}; line-height: 1.6;">
                    <li><strong>Valor por Segundo de Anticipación:</strong> $1,000 / segundo útil de aviso antes de la onda destructiva S.</li>
                    <li><strong>Costo de Falsa Alarma:</strong> -$2,500 por paro innecesario de infraestructura y líneas operativas.</li>
                    <li><strong>Costo de Evento Omitido (Falso Negativo):</strong> -$10,000 por impacto sísmico sin advertencia temprana.</li>
                    <li><strong>Costo Operativo de Cómputo:</strong> -$1.00 por cada ventana analizada.</li>
                    <li><strong>Exclusión Inicial (Warmup):</strong> 100 muestras (1.0 s) ignoradas para eliminar transitorios de filtrado.</li>
                    <li><strong>Tolerancia de Picking:</strong> 0.5 s con respecto a la fase manual del catálogo.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
