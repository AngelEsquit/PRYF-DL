"""Dashboard Streamlit para demostrar AlertaP."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from seismic import DeepPhaseDetector, generate_example, sta_lta_detect

st.set_page_config(page_title="AlertaP", page_icon="🌋", layout="wide")
st.title("🌋 AlertaP")
st.subheader("Detección temprana de la onda P antes de la onda S")
st.caption("Prototipo educativo: no sustituye un sistema sismológico certificado.")

@st.cache_resource
def load_detector() -> DeepPhaseDetector:
    return DeepPhaseDetector()

detector = load_detector()
col1, col2 = st.columns(2)
with col1:
    event = st.toggle("Simular evento sísmico", value=True)
    threshold = st.slider("Umbral de alerta", 0.40, 0.90, 0.65, 0.05)
with col2:
    seed = st.number_input("Semilla reproducible", min_value=1, value=23361, step=1)

if st.button("Analizar ventana", type="primary"):
    example = generate_example(np.random.default_rng(seed), event=event)
    deep = detector.detect(example.signal, threshold=threshold)
    classic = sta_lta_detect(example.signal)
    figure, axis = plt.subplots(figsize=(12, 3.5))
    axis.plot(np.arange(len(example.signal)) / 100, example.signal[:, 0], label="Componente Z", linewidth=0.8)
    if example.p_arrival:
        axis.axvline(example.p_arrival / 100, color="green", linestyle="--", label="P real")
    if example.s_arrival:
        axis.axvline(example.s_arrival / 100, color="red", linestyle="--", label="S real")
    if deep.sample:
        axis.axvline(deep.sample / 100, color="purple", label="Alerta DeepPhase")
    axis.set(xlabel="Tiempo (s)", ylabel="Amplitud", title="Ventana de tres componentes (se muestra Z)")
    axis.legend(loc="upper right")
    st.pyplot(figure)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DeepPhase", "ALERTA" if deep.detected else "Sin alerta")
    m2.metric("Probabilidad", f"{deep.probability:.2f}")
    m3.metric("STA/LTA", "ALERTA" if classic.detected else "Sin alerta")
    m4.metric("Latencia", f"{deep.latency_ms:.2f} ms")
    if deep.detected and example.s_arrival:
        st.success(f"Anticipación estimada: {(example.s_arrival - deep.sample) / 100:.2f} s antes de la onda S.")
    elif not event:
        st.info("Ventana de ruido: una ausencia de alerta es el resultado esperado.")
    else:
        st.warning("No se alcanzó el umbral. En producción se requieren más estaciones y confirmación.")
