"""Dashboard Streamlit interactivo para AlertaP.

Permite probar y comparar en tiempo real:
1. ConvPhaseNet1D entrenado sobre datos reales STEAD (PyTorch GPU / CPU).
2. ConvPhaseNet1D entrenado sobre datos sintéticos (PyTorch GPU / CPU).
3. Detector MLP clásico DeepPhase.
Funciona fluidamente tanto en PCs con GPU dedicada como en laptops con gráficos integrados (CPU).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from seismic import (
    DeepPhaseDetector,
    PyTorchPhaseDetector,
    SignalExample,
    generate_example,
    sta_lta_detect,
)

st.set_page_config(page_title="AlertaP - Detección Temprana de Sismos", page_icon="🌋", layout="wide")
st.title("🌋 AlertaP: Detección Temprana de Fase P")
st.subheader("Detección de la onda P antes de la onda S para anticipación operativa")
st.caption("CC3092 Deep Learning - CC3092 UVG | Compatible con CPU y GPU")

has_synth_ckpt = Path("checkpoints/best_model.pt").exists()
has_stead_ckpt = Path("checkpoints_stead/best_model.pt").exists()
has_demo_samples = Path("results/demo_samples.npz").exists()


@st.cache_resource
def load_deep_phase() -> DeepPhaseDetector:
    return DeepPhaseDetector()


@st.cache_resource
def load_pytorch_model(path: str) -> PyTorchPhaseDetector:
    # Carga automática en GPU si está disponible, o en CPU en laptops con gráficos integrados
    return PyTorchPhaseDetector(checkpoint_path=path, device="auto")


@st.cache_data
def load_demo_samples():
    if not has_demo_samples:
        return None
    return dict(np.load("results/demo_samples.npz"))


# --- SIDEBAR: Configuración del modelo y entorno ---
st.sidebar.header("Configuración")

options = []
if has_stead_ckpt:
    options.append("ConvPhaseNet1D (STEAD Real - Recomendado)")
if has_synth_ckpt:
    options.append("ConvPhaseNet1D (Sintético)")
options.append("DeepPhase (MLP clásico)")

model_choice = st.sidebar.selectbox("Modelo a evaluar:", options)

if "STEAD Real" in model_choice:
    detector = load_pytorch_model("checkpoints_stead/best_model.pt")
elif "Sintético" in model_choice:
    detector = load_pytorch_model("checkpoints/best_model.pt")
else:
    detector = load_deep_phase()

device_info = getattr(detector, "device", "CPU")
st.sidebar.info(f"Dispositivo de cómputo en uso: **{device_info}**")

# Selección de fuente de datos para probar
data_source = st.sidebar.radio(
    "Fuente de señal a probar:",
    ["Sismogramas reales STEAD (Prueba)", "Simulación sintética"],
    index=0 if has_demo_samples else 1,
)

threshold = st.sidebar.slider(
    "Umbral de decisión (óptimo STEAD: 0.50 | Sintético: 0.85):",
    min_value=0.20,
    max_value=0.95,
    value=0.50 if "STEAD" in model_choice else 0.85,
    step=0.05,
)

# --- PANEL PRINCIPAL ---
demo_data = load_demo_samples() if data_source.startswith("Sismogramas") else None

if demo_data is not None:
    st.markdown("### Seleccionar Sismograma Real de Prueba (STEAD)")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        tipo_filtro = st.radio("Tipo de registro:", ["Sismo real", "Ruido sísmico ambiental"], horizontal=True)
    
    is_event_type = 1 if tipo_filtro == "Sismo real" else 0
    indices_disponibles = np.where(demo_data["labels"] == is_event_type)[0]
    
    with col_sel2:
        ejemplo_idx = st.selectbox(
            f"Registro {tipo_filtro}:",
            indices_disponibles,
            format_func=lambda i: f"Registro #{i+1} (SNR: {demo_data['snr'][i]:.1f} dB)" if is_event_type else f"Ruido #{i+1}"
        )
    
    sig = demo_data["signals"][ejemplo_idx]
    p_arr = int(demo_data["p"][ejemplo_idx]) if is_event_type and demo_data["p"][ejemplo_idx] >= 0 else None
    s_arr = int(demo_data["s"][ejemplo_idx]) if is_event_type and demo_data["s"][ejemplo_idx] >= 0 else None
    example = SignalExample(sig, p_arr, s_arr, is_event_type, {"snr_db": float(demo_data["snr"][ejemplo_idx])})
else:
    st.markdown("### Simulación de Ventana Sísmica")
    col1, col2 = st.columns(2)
    with col1:
        event = st.toggle("Simular llegada de sismo", value=True)
    with col2:
        seed = st.number_input("Semilla reproducible:", min_value=1, value=23361, step=1)
    example = generate_example(np.random.default_rng(seed), event=event)

# --- INFERENCIA ---
if st.button("🚀 Analizar Señal Sísmica", type="primary", use_container_width=True):
    deep = detector.detect(example.signal, threshold=threshold)
    classic = sta_lta_detect(example.signal)
    
    # Gráfica de 3 componentes
    time_axis = np.arange(len(example.signal)) / 100.0  # 100 Hz
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    components = [("Vertical (Z)", 0, "black"), ("Norte-Sur (N)", 1, "blue"), ("Este-Oeste (E)", 2, "red")]
    
    for ax, (cname, cidx, color) in zip(axes, components):
        ax.plot(time_axis, example.signal[:, cidx], label=cname, color=color, linewidth=0.7)
        if example.p_arrival is not None:
            ax.axvline(example.p_arrival / 100.0, color="green", linestyle="--", linewidth=1.5, label="Onda P real")
        if example.s_arrival is not None:
            ax.axvline(example.s_arrival / 100.0, color="crimson", linestyle="--", linewidth=1.5, label="Onda S real")
        if deep.sample is not None:
            ax.axvline(deep.sample / 100.0, color="purple", linewidth=2.0, label="Alerta ConvPhaseNet")
        if classic.sample is not None:
            ax.axvline(classic.sample / 100.0, color="orange", linestyle=":", linewidth=1.5, label="Alerta STA/LTA")
        ax.set_ylabel("Amplitud normalizada", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    
    axes[-1].set_xlabel("Tiempo (segundos)", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    
    # Métricas del evento
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alerta AlertaP (Deep Learning)", "🚨 SÍSMICA" if deep.detected else "🟢 Normal")
    m2.metric("Probabilidad Máx. Onda P", f"{deep.probability:.1%}")
    m3.metric("Baseline STA/LTA", "🚨 Alerta" if classic.detected else "🟢 Normal")
    m4.metric("Latencia de Inferencia", f"{deep.latency_ms:.2f} ms")
    
    if deep.detected and example.s_arrival is not None and deep.sample is not None:
        anticipacion = (example.s_arrival - deep.sample) / 100.0
        if anticipacion > 0:
            st.success(f"⏱️ **Anticipación Operativa Útil:** **{anticipacion:.2f} segundos** antes de la llegada de la onda destructiva S. Tiempo suficiente para cerrar válvulas y detener maquinaria.")
        else:
            st.warning("⚠️ Alerta tardía: se detectó después de la llegada de la onda S.")
    elif not example.label:
        if not deep.detected:
            st.info("✅ **Ventana de ruido identificada correctamente.** El modelo evitó con éxito una costosa falsa alarma.")
        else:
            st.error("⚠️ Falsa alarma generada en ruido ambiental.")
    else:
        st.warning("⚠️ El sismo no superó el umbral de decisión.")
