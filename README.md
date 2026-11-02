# AlertaP: detección temprana de sismos

Proyecto final de **CC3092 Deep Learning y Sistemas Inteligentes** basado en la
propuesta aprobada: detectar la llegada de la onda P en registros de tres
componentes antes de la onda S y convertir la anticipación en valor operativo.

## Qué incluye

- Generador reproducible de ventanas sísmicas sintéticas de tres componentes.
- Detector profundo `DeepPhase` (MLP de dos capas) inspirado en un picker de fases.
- Baseline clásico STA/LTA.
- Métricas técnicas: error de picking, precisión, recall, F1 y latencia.
- Métrica de negocio: valor neto = segundos de anticipación valorizados menos
  costo de falsas alarmas.
- Dashboard Streamlit para demostrar una ventana, la alerta y la comparación.
- Informe completo en [`docs/informe.tex`](docs/informe.tex) y PDF en
  [`docs/informe.pdf`](docs/informe.pdf).

## Instalación y ejecución

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run src/app.py
```

La demo funciona sin API externa. Para reproducir las métricas:

```bash
.venv/bin/pytest -q
.venv/bin/python src/evaluate.py
```

El comando genera `results/metrics.json`. Las señales sintéticas son una
validación controlada del pipeline; la integración con STEAD o FDSN/IRIS queda
preparada como siguiente etapa y se documenta explícitamente en el informe.

## Reproducibilidad y uso responsable

La semilla, la tasa de muestreo, la arquitectura y el umbral están definidos en
el código. Una alerta de este prototipo no debe accionar maquinaria ni sistemas
de seguridad reales: para producción se requieren datos STEAD/locales validados,
calibración por estación, redundancia de sensores, control de deriva y
certificación sismológica.
