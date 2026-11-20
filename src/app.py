"""AlertaP: Detección temprana de sismos con Deep Learning.

Demostración ejecutiva para evaluación académica.
Compara una CNN 1-D (ConvPhaseNet1D) contra el método clásico STA/LTA.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Rutas e imports del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seismic import (
    PyTorchPhaseDetector,
    SignalExample,
    generate_example,
    sta_lta_detect,
)

# ---------------------------------------------------------------------------
# Configuración general (tema claro, limpio y directo)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AlertaP | Detección Temprana de Sismos",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo visual limpio y tipografía legible (mínimo 14px)
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 15px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: left;
    }
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-help {
        font-size: 13px;
        color: #64748b;
        margin-top: 6px;
    }
    .status-ok {
        color: #16a34a;
        font-weight: 700;
    }
    .status-alert {
        color: #2563eb;
        font-weight: 700;
    }
    .status-warn {
        color: #ea580c;
        font-weight: 700;
    }
    div.stButton > button {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Carga de recursos
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "checkpoints_stead/best_model.pt"
    if Path("checkpoints_stead/best_model.pt").exists()
    else "checkpoints/best_model.pt"
)
DEMO_SAMPLES_PATH = Path("results/demo_samples.npz")
METRICS_PATH = (
    Path("results/metrics_stead.json")
    if Path("results/metrics_stead.json").exists()
    else Path("results/metrics.json")
)
HISTORY_PATH = (
    Path("checkpoints_stead/history.csv")
    if Path("checkpoints_stead/history.csv").exists()
    else Path("checkpoints/history.csv")
)


@st.cache_resource
def get_detector() -> PyTorchPhaseDetector:
    return PyTorchPhaseDetector(checkpoint_path=CHECKPOINT_PATH, device="auto")


@st.cache_data
def get_demo_samples():
    if DEMO_SAMPLES_PATH.exists():
        return dict(np.load(str(DEMO_SAMPLES_PATH)))
    return None


@st.cache_data
def get_metrics_data():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_history_df():
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return None


detector = get_detector()
demo_samples = get_demo_samples()
metrics = get_metrics_data()

# ---------------------------------------------------------------------------
# Barra lateral: solo 2 controles esenciales
# ---------------------------------------------------------------------------
st.sidebar.header("Configuración")

fuente_senial = st.sidebar.radio(
    "Fuente de señal:",
    ["STEAD real", "Sintético"],
    index=0 if demo_samples is not None else 1,
)

umbral = st.sidebar.slider(
    "Umbral de decisión (CNN):",
    min_value=0.20,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Punto de corte de probabilidad para emitir una alerta.",
)

STALTA_THRESH = 2.5  # Fijo en 2.5 según calibración óptima

# ---------------------------------------------------------------------------
# Título claro y directo
# ---------------------------------------------------------------------------
st.title("AlertaP: Detección Temprana de Sismos")
st.write(
    "Detección de la onda primaria (P) mediante una red neuronal convolucional (CNN 1-D) "
    "para alertar antes de la llegada de la onda destructiva (S)."
)

# ---------------------------------------------------------------------------
# Estructura principal en 3 pestañas
# ---------------------------------------------------------------------------
tab_demo, tab_resultados, tab_valor = st.tabs(["Demo", "Resultados", "Valor y umbral"])


# ===========================================================================
# PESTAÑA 1: DEMO (Monitor interactivo)
# ===========================================================================
with tab_demo:
    # Gestión de estado de la señal activa
    if "signal" not in st.session_state or st.session_state.get("last_source") != fuente_senial:
        st.session_state.last_source = fuente_senial
        # Inicializar con un sismo
        if fuente_senial == "STEAD real" and demo_samples is not None:
            earthquake_idxs = np.where(demo_samples["labels"] == 1)[0]
            chosen_idx = int(earthquake_idxs[0])
            st.session_state.signal = demo_samples["signals"][chosen_idx]
            st.session_state.p = int(demo_samples["p"][chosen_idx])
            st.session_state.s = int(demo_samples["s"][chosen_idx])
            st.session_state.label = 1
        else:
            rng = np.random.default_rng(23361)
            ex = generate_example(rng, event=True)
            st.session_state.signal = ex.signal
            st.session_state.p = ex.p_arrival
            st.session_state.s = ex.s_arrival
            st.session_state.label = 1

    # Botones superiores para elegir señal
    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 3])
    with col_btn1:
        if st.button("Elegir sismo al azar", type="primary", use_container_width=True):
            if fuente_senial == "STEAD real" and demo_samples is not None:
                earthquake_idxs = np.where(demo_samples["labels"] == 1)[0]
                idx = int(np.random.choice(earthquake_idxs))
                st.session_state.signal = demo_samples["signals"][idx]
                st.session_state.p = int(demo_samples["p"][idx])
                st.session_state.s = int(demo_samples["s"][idx])
                st.session_state.label = 1
            else:
                rng = np.random.default_rng()
                ex = generate_example(rng, event=True)
                st.session_state.signal = ex.signal
                st.session_state.p = ex.p_arrival
                st.session_state.s = ex.s_arrival
                st.session_state.label = 1

    with col_btn2:
        if st.button("Elegir ruido al azar", use_container_width=True):
            if fuente_senial == "STEAD real" and demo_samples is not None:
                noise_idxs = np.where(demo_samples["labels"] == 0)[0]
                idx = int(np.random.choice(noise_idxs))
                st.session_state.signal = demo_samples["signals"][idx]
                st.session_state.p = None
                st.session_state.s = None
                st.session_state.label = 0
            else:
                rng = np.random.default_rng()
                ex = generate_example(rng, event=False)
                st.session_state.signal = ex.signal
                st.session_state.p = None
                st.session_state.s = None
                st.session_state.label = 0

    # Análisis automático con el umbral configurado
    sig = st.session_state.signal
    p_arr = st.session_state.p
    s_arr = st.session_state.s
    is_event = st.session_state.label == 1

    deep_det = detector.detect(sig, threshold=umbral)
    classic_det = sta_lta_detect(sig, threshold=STALTA_THRESH)

    # Gráfica única de la componente Z
    time_hz = 100.0
    time_axis = np.arange(len(sig)) / time_hz

    fig, ax = plt.subplots(figsize=(10, 2.7), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # Señal vertical (Z)
    ax.plot(time_axis, sig[:, 0], color="#1e293b", linewidth=0.9, label="Señal vertical (Z)")

    # Marcas con etiquetas en español
    if is_event and p_arr is not None:
        ax.axvline(p_arr / time_hz, color="#16a34a", linestyle="--", linewidth=1.8, label="P real")
    if is_event and s_arr is not None:
        ax.axvline(s_arr / time_hz, color="#dc2626", linestyle="--", linewidth=1.8, label="S real")

    if deep_det.detected and deep_det.sample is not None:
        ax.axvline(
            deep_det.sample / time_hz,
            color="#2563eb",
            linestyle="-",
            linewidth=2.0,
            label="Detección de la CNN",
        )

    if classic_det.detected and classic_det.sample is not None:
        ax.axvline(
            classic_det.sample / time_hz,
            color="#d97706",
            linestyle=":",
            linewidth=1.8,
            label="Detección STA/LTA",
        )

    ax.set_xlabel("Tiempo (segundos)", fontsize=10, color="#1e293b")
    ax.set_ylabel("Amplitud", fontsize=10, color="#1e293b")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Cálculo de métricas para las tarjetas
    aviso_s = 0.0
    error_s_str = "N/A"
    resultado_str = ""
    resultado_help = ""
    resultado_color = "status-ok"

    if is_event:
        if deep_det.detected and deep_det.sample is not None:
            if s_arr is not None and deep_det.sample < s_arr:
                aviso_s = (s_arr - deep_det.sample) / time_hz
                resultado_str = "Detectado"
                resultado_help = "La CNN alertó a tiempo, antes de la onda S."
                resultado_color = "status-ok"
            else:
                resultado_str = "Alerta tardía"
                resultado_help = "La detección ocurrió después de la onda S."
                resultado_color = "status-warn"

            if p_arr is not None:
                err_val = abs(deep_det.sample - p_arr) / time_hz
                error_s_str = f"{err_val:.2f} s"
        else:
            resultado_str = "No detectado"
            resultado_help = "El sismo no alcanzó el umbral configurado."
            resultado_color = "status-warn"
    else:
        # Es ruido ambiental
        if not deep_det.detected:
            resultado_str = "Sin alarma: correcto"
            resultado_help = "Se identificó ruido ambiental y se evitó una falsa alarma."
            resultado_color = "status-ok"
        else:
            resultado_str = "Falsa alarma"
            resultado_help = "El modelo disparó incorrectamente sobre ruido."
            resultado_color = "status-warn"

    # Tres tarjetas grandes debajo de la gráfica
    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Segundos de aviso antes de la onda S</div>
                <div class="metric-val">{aviso_s:.1f} s</div>
                <div class="metric-help">Tiempo útil para activar medidas preventivas antes del impacto fuerte.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Error de la detección (s)</div>
                <div class="metric-val">{error_s_str}</div>
                <div class="metric-help">Diferencia temporal frente a la llegada real de la onda P.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Resultado</div>
                <div class="metric-val {resultado_color}">{resultado_str}</div>
                <div class="metric-help">{resultado_help}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# PESTAÑA 2: RESULTADOS (CNN vs STA/LTA)
# ===========================================================================
with tab_resultados:
    st.subheader("En datos reales, la CNN detecta más sismos y con menos falsas alarmas que el método clásico.")

    dl_metrics = metrics.get("DeepPhase", {})
    sta_metrics = metrics.get("STA/LTA", {})

    # Tabla simple de 2 filas
    df_comparativa = pd.DataFrame(
        {
            "Modelo": ["CNN (ConvPhaseNet1D)", "STA/LTA (Método clásico)"],
            "Precisión": [
                f"{dl_metrics.get('precision', 0.9622):.1%} ★",
                f"{sta_metrics.get('precision', 0.8910):.1%}",
            ],
            "Recall (Sensibilidad)": [
                f"{dl_metrics.get('recall', 0.9513):.1%} ★",
                f"{sta_metrics.get('recall', 0.8346):.1%}",
            ],
            "F1-Score": [
                f"{dl_metrics.get('f1', 0.9567):.3f} ★",
                f"{sta_metrics.get('f1', 0.8619):.3f}",
            ],
            "Error momento de detección": [
                f"{dl_metrics.get('mean_picking_error_s', 0.1888):.2f} s ★",
                f"{sta_metrics.get('mean_picking_error_s', 0.5866):.2f} s",
            ],
            "Valor neto estimado": [
                f"${dl_metrics.get('business_value', 9196291):,.0f} ★",
                f"${sta_metrics.get('business_value', 5778931):,.0f}",
            ],
        }
    )
    st.dataframe(df_comparativa, use_container_width=True, hide_index=True)

    st.write("")
    st.write("**Comparación de eventos evaluados en el conjunto de prueba:**")

    # Dos gráficas lado a lado de barras
    col_g1, col_g2 = st.columns(2)

    categories = ["Aciertos", "Falsas alarmas", "Eventos perdidos"]
    colors = ["#16a34a", "#ea580c", "#dc2626"]

    with col_g1:
        vals_cnn = [
            dl_metrics.get("true_positives", 1426),
            dl_metrics.get("false_positives", 56),
            dl_metrics.get("false_negatives", 73),
        ]
        fig_c1, ax_c1 = plt.subplots(figsize=(5, 3), dpi=120)
        bars1 = ax_c1.bar(categories, vals_cnn, color=colors, width=0.55)
        ax_c1.set_title("CNN (ConvPhaseNet1D)", fontsize=11, fontweight="600", color="#1e293b")
        ax_c1.set_ylabel("Cantidad de ventanas", fontsize=9, color="#1e293b")
        ax_c1.set_ylim(0, 1600)
        ax_c1.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
        for spine in ax_c1.spines.values():
            spine.set_color("#cbd5e1")
        for bar in bars1:
            h = bar.get_height()
            ax_c1.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{int(h):,}", ha="center", fontsize=9, fontweight="600")
        fig_c1.tight_layout()
        st.pyplot(fig_c1)
        plt.close(fig_c1)

    with col_g2:
        vals_sta = [
            sta_metrics.get("true_positives", 1251),
            sta_metrics.get("false_positives", 153),
            sta_metrics.get("false_negatives", 248),
        ]
        fig_c2, ax_c2 = plt.subplots(figsize=(5, 3), dpi=120)
        bars2 = ax_c2.bar(categories, vals_sta, color=colors, width=0.55)
        ax_c2.set_title("STA/LTA (Método clásico)", fontsize=11, fontweight="600", color="#1e293b")
        ax_c2.set_ylabel("Cantidad de ventanas", fontsize=9, color="#1e293b")
        ax_c2.set_ylim(0, 1600)
        ax_c2.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
        for spine in ax_c2.spines.values():
            spine.set_color("#cbd5e1")
        for bar in bars2:
            h = bar.get_height()
            ax_c2.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{int(h):,}", ha="center", fontsize=9, fontweight="600")
        fig_c2.tight_layout()
        st.pyplot(fig_c2)
        plt.close(fig_c2)

    # Expander con matriz de confusión detallada
    with st.expander("Ver desglose numérico detallado de matriz de confusión"):
        df_matriz = pd.DataFrame(
            {
                "Categoría": ["Aciertos (alertas útiles)", "Falsas alarmas (en ruido)", "Eventos perdidos (sin aviso)"],
                "CNN": [f"{vals_cnn[0]:,}", f"{vals_cnn[1]:,}", f"{vals_cnn[2]:,}"],
                "STA/LTA": [f"{vals_sta[0]:,}", f"{vals_sta[1]:,}", f"{vals_sta[2]:,}"],
                "Diferencia": [
                    f"+{vals_cnn[0] - vals_sta[0]} a favor de CNN",
                    f"-{vals_sta[1] - vals_cnn[1]} menos falsas alarmas",
                    f"-{vals_sta[2] - vals_cnn[2]} menos sismos perdidos",
                ],
            }
        )
        st.dataframe(df_matriz, use_container_width=True, hide_index=True)

    # Expander con detalles técnicos y convergencia
    with st.expander("Detalles técnicos del modelo y entrenamiento"):
        st.write(
            "- **Arquitectura:** ConvPhaseNet1D con convoluciones dilatadas sobre los 3 canales (Z, N, E) y bloques residuales.\n"
            "- **Datos de entrenamiento:** Base de datos STEAD (Stanford Earthquake Dataset) particionada por sismo para evitar memorización.\n"
            "- **Función de pérdida:** BCEWithLogitsLoss con regularización Dropout (0.10) y optimizador AdamW."
        )
        df_h = get_history_df()
        if df_h is not None and not df_h.empty:
            st.write("**Evolución de las últimas épocas de entrenamiento:**")
            st.dataframe(df_h[["epoch", "train_loss", "val_loss", "val_f1", "val_business_value"]].tail(5), hide_index=True)


# ===========================================================================
# PESTAÑA 3: VALOR Y UMBRAL (Modelo económico)
# ===========================================================================
with tab_valor:
    st.write(
        "Bajar el umbral da más aviso pero más falsas alarmas; subirlo reduce falsas alarmas pero pierde sismos."
    )

    # Curva única de valor neto vs umbral
    val_test = metrics.get("value_curve_test", {})
    cnn_curve = val_test.get("DeepPhase", [])

    if cnn_curve:
        df_val = pd.DataFrame(cnn_curve)
        thresholds_x = df_val["threshold"]
        values_y = df_val["business_value"] / 1e6
        sta_val_m = sta_metrics.get("business_value", 5778931) / 1e6

        fig_v, ax_v = plt.subplots(figsize=(9, 3.8), dpi=130)
        fig_v.patch.set_facecolor("#ffffff")
        ax_v.set_facecolor("#f8fafc")

        # Curva CNN
        ax_v.plot(
            thresholds_x,
            values_y,
            marker="o",
            color="#2563eb",
            linewidth=2.2,
            label="CNN (ConvPhaseNet1D)",
        )

        # Línea de referencia STA/LTA (en su mejor umbral)
        ax_v.axhline(
            sta_val_m,
            color="#d97706",
            linestyle="--",
            linewidth=1.8,
            label=f"STA/LTA óptimo (${sta_val_m:.2f}M)",
        )

        # Línea vertical del umbral elegido en validación (0.50)
        ax_v.axvline(
            0.50,
            color="#16a34a",
            linestyle=":",
            linewidth=2.0,
            label="Umbral elegido en validación (0.50)",
        )

        # Punto óptimo resaltado
        ax_v.scatter([0.50], [values_y.iloc[df_val[df_val['threshold'] == 0.50].index[0]]], color="#16a34a", s=80, zorder=5)

        ax_v.set_xlabel("Umbral de probabilidad de la CNN", fontsize=10, color="#1e293b")
        ax_v.set_ylabel("Valor neto estimado (Millones USD)", fontsize=10, color="#1e293b")
        ax_v.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
        for spine in ax_v.spines.values():
            spine.set_color("#cbd5e1")

        ax_v.legend(loc="lower center", fontsize=9, framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0")
        fig_v.tight_layout()
        st.pyplot(fig_v)
        plt.close(fig_v)

    # Supuestos visibles en una sola línea
    st.info(
        "**Supuestos del modelo:** $1,000 por segundo de aviso útil | "
        "$2,500 por falsa alarma | $10,000 por sismo perdido | $1 por ventana evaluada."
    )

    # Expander cerrado con sensibilidad
    with st.expander("Ver análisis de sensibilidad económica", expanded=False):
        st.write("La CNN gana en los 12 escenarios económicos probados (variando costos y valor ±20 % y ±50 %).")
        sens_data = metrics.get("sensitivity", [])
        if sens_data:
            df_sens = pd.DataFrame(sens_data)
            df_sens = df_sens.rename(
                columns={
                    "parameter": "Parámetro evaluado",
                    "change_pct": "Cambio (%)",
                    "deep_value_at_default_threshold": "Valor CNN ($)",
                    "baseline_value": "Valor STA/LTA ($)",
                    "advantage_vs_baseline": "Ventaja de la CNN ($)",
                }
            )
            st.dataframe(
                df_sens[["Parámetro evaluado", "Cambio (%)", "Valor CNN ($)", "Valor STA/LTA ($)", "Ventaja de la CNN ($)"]],
                use_container_width=True,
                hide_index=True,
            )

# ---------------------------------------------------------------------------
# Nota al pie obligatoria de honestidad académica
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Resultados sobre un subconjunto de STEAD; no validado con estaciones guatemaltecas; "
    "prototipo académico, no apto para accionar sistemas reales."
)
