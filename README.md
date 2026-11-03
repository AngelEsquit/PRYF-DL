# AlertaP: detección temprana de sismos

Proyecto final de **CC3092 Deep Learning y Sistemas Inteligentes** basado en la
propuesta aprobada: detectar la llegada de la onda P en registros de tres
componentes antes de la onda S y convertir la anticipación en valor operativo.

## Qué incluye

- Generador reproducible de ventanas sísmicas sintéticas de tres componentes.
- Detector profundo `DeepPhase` (MLP de dos capas) inspirado en un picker de fases.
- Baseline clásico STA/LTA.
- Métricas técnicas: error de picking, precisión, recall, F1 y latencia.
- Modelo de valor explícito: valor neto = anticipación valorizada - falsas
  alarmas - eventos no detectados - costo de operación, con supuestos
  documentados en `ValueParams` (`src/seismic.py`).
- Curva de valor por umbral, selección del umbral por valor en un conjunto de
  validación y análisis de sensibilidad (±20 % y ±50 %).
- Dashboard Streamlit para demostrar una ventana, la alerta y la comparación.
- Informe completo en [`docs/informe.tex`](docs/informe.tex) y PDF en
  [`docs/informe.pdf`](docs/informe.pdf).

## Instalación y ejecución

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run src/app.py
```

Para reproducir las métricas:

```bash
.venv/bin/pytest -q
.venv/bin/python src/evaluate.py
```

El comando genera `results/metrics.json` (métricas, curva de valor y sensibilidad) y
`results/value_curve.png`. Las señales sintéticas son una
validación controlada del pipeline; la integración con STEAD o FDSN/IRIS queda
preparada como siguiente etapa y se documenta explícitamente en el informe.

## Datos reales (STEAD) y validación local (FDSN/IRIS)

Nada se descarga automáticamente. Los datos pesan decenas de GB y se obtienen por
separado:

1. **STEAD**: descarga `merged.csv` y `merged.hdf5` desde
   <https://github.com/smousavi05/STEAD> (licencia CC BY 4.0; citar a Mousavi et al.,
   2019, *IEEE Access*). SeisBench no los descarga por ti: su cargador `STEAD`
   exige esos mismos archivos y convertirlos completos tarda horas.
2. Instala las dependencias opcionales: `pip install -r requirements-data.txt`.
3. Evalúa con un subconjunto (por defecto 3000 eventos y 3000 ventanas de ruido,
   particionados **por evento**):

```bash
python src/evaluate.py --source stead --stead-dir RUTA/A/STEAD
python src/evaluate.py --source stead --stead-dir RUTA/A/STEAD --fdsn   # + validación FDSN
```

Las ventanas procesadas se guardan en `data/` (ignorado por git), así que las
siguientes ejecuciones no vuelven a leer el HDF5. Los resultados reales se
escriben en `results/metrics_stead.json` y `results/value_curve_stead.png`, sin
tocar los sintéticos. El backend `--stead-backend seisbench` existe pero solo
conviene si ya tienes la caché de SeisBench.

La validación FDSN usa eventos de Guatemala y región (IRIS), y toma las llegadas P
y S **teóricas** (TauP, iasp91), no picks manuales; es una validación gruesa de
transferencia.

## Reproducibilidad y uso responsable

La semilla, la tasa de muestreo, la arquitectura y el umbral están definidos en
el código. Una alerta de este prototipo no debe accionar maquinaria ni sistemas
de seguridad reales: para producción se requieren datos STEAD/locales validados,
calibración por estación, redundancia de sensores, control de deriva y
certificación sismológica.
