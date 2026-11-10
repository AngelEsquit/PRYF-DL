# Guía Completa de Entrenamiento en Windows (NVIDIA RTX 5050)

Esta guía detalla paso a paso cómo migrar, configurar y entrenar el sistema **AlertaP** en **Windows**, aprovechando el espacio de disco disponible en tu partición de Windows y la aceleración por hardware de tu tarjeta gráfica **NVIDIA GeForce RTX 5050 (8 GB VRAM)**.

---

## 1. Requisitos Previos en Windows

1. **Python instalado:**
   - Asegúrate de tener instalado Python (3.10, 3.11, 3.12 o posterior) desde [python.org](https://www.python.org/downloads/).
   - ⚠️ **Importante:** Al instalar, marca la casilla **"Add python.exe to PATH"**.
2. **Git para Windows:**
   - Descarga desde [git-scm.com](https://git-scm.com/) si aún no lo tienes.
3. **Drivers NVIDIA:**
   - Tener instalados los controladores GeForce Game Ready o Studio más recientes desde la aplicación NVIDIA App / GeForce Experience.

---

## 2. Clonar o Actualizar el Repositorio en Windows

Abre una terminal (**PowerShell** o **Terminal de Windows**) y ubícate en la carpeta donde tengas espacio en disco (por ejemplo en tu disco `D:\` o `C:\Users\tu_usuario\Proyectos`):

```powershell
git clone https://github.com/AngelEsquit/PRYF-DL.git
cd PRYF-DL
git checkout main
git pull origin main
```

*(Si compartes partición en el mismo equipo, puedes simplemente acceder a la carpeta del repositorio directamente).*

---

## 3. Crear Entorno Virtual e Instalar Dependencias con CUDA

Abre **PowerShell** dentro de la carpeta del proyecto `PRYF-DL` y ejecuta:

```powershell
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Si PowerShell bloquea scripts, habilita la ejecución en la sesión actual:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 4. Actualizar pip e instalar dependencias básicas
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Instalar PyTorch con aceleración CUDA (para RTX serie 50 / Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Comprobar que PyTorch detecta tu RTX 5050 en Windows:
Ejecuta en la terminal:
```powershell
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No detectada')"
```
Debe responder:
```text
CUDA disponible: True
GPU: NVIDIA GeForce RTX 5050
```

---

## 4. Descarga y Preparación del Dataset STEAD

STEAD (**Stanford Earthquake Dataset**) pesa aproximadamente **85 a 90 GB** descomprimido.

### Pasos de descarga:
1. Ve al repositorio oficial: **[GitHub - smousavi05/STEAD](https://github.com/smousavi05/STEAD)**.
2. En la sección **Data Description & Download Links**, encontrarás los enlaces a **Box / Google Drive / Zenodo**.
3. Descarga en un disco con espacio (por ejemplo `D:\Datos\STEAD\`):
   - `merged.csv` (~300 MB): archivo con metadatos y llegadas P/S.
   - `merged.hdf5` (~85-90 GB): archivo HDF5 con las trazas de 3 componentes.
   *(Si el autor lo distribuye en `chunk1.hdf5` a `chunk6.hdf5`, sigue las instrucciones del repositorio para fusionarlos con el script `merge.py` proporcionado allí).*

> [!TIP]
> **Ahorro de espacio (Alternativa de Caché):**
> No necesitas conservar los 90 GB para siempre. Una vez que corras el script de entrenamiento por primera vez, el sistema extrae las 6,000 ventanas necesarias y crea el archivo compacto **`data\stead_subset.npz`** (~200 MB). Una vez generado ese archivo `.npz`, puedes mover o eliminar el archivo de 90 GB.

---

## 5. Comandos para Entrenar (Tolerante a Apagados)

El pipeline de entrenamiento está programado con **guardado atómico de checkpoints** y **Automatic Mixed Precision (AMP)** para no saturar los 8 GB de VRAM.

### Opción A: Entrenar con dataset real STEAD (en tu disco de Windows)
Suponiendo que guardaste STEAD en `D:\Datos\STEAD`:
```powershell
.\.venv\Scripts\python.exe src\train.py --source stead --stead-dir D:\Datos\STEAD --model conv1d --epochs 50 --batch-size 64
```

### Opción B: Entrenar con datos sintéticos controlados
Si quieres entrenar de inmediato sin esperar a descargar STEAD:
```powershell
.\.venv\Scripts\python.exe src\train.py --model conv1d --epochs 50 --batch-size 64
```

### Opción C: Ejecución rápida con doble clic
Puedes simplemente hacer doble clic sobre el archivo **`train_windows.bat`** en el Explorador de Archivos de Windows.

---

## 6. Cómo Pausar y Reanudar si Apagas la Computadora

Como no puedes tener la computadora prendida siempre, el sistema maneja la interrupción sin pérdidas:

1. **Para pausar antes de apagar la PC:**
   - Presiona `Ctrl + C` en la terminal de PowerShell.
   - Verás el mensaje:
     ```text
     [INTERRUPT] Señal de interrupción detectada (Ctrl+C o apagado de sistema).
     [INTERRUPT] Checkpoint guardado en: checkpoints\checkpoint_interrupt.pt
     ```
   *(Incluso si la computadora se apaga de golpe, la última época completada queda intacta en `checkpoints\checkpoint_last.pt` gracias al guardado atómico).*

2. **Para reanudar al volver a encender la PC:**
   Ejecuta exactamente el mismo comando agregando `--resume-auto` (activado por defecto) o especificando el archivo:
   ```powershell
   .\.venv\Scripts\python.exe src\train.py --resume checkpoints\checkpoint_last.pt
   ```
   El sistema restaurará la época exacta, pesos, estado de AdamW, LR scheduler y seguirá entrenando desde donde quedó.

---

## 7. Archivos Generados durante el Entrenamiento

En la carpeta `checkpoints\` se generan automáticamente:
- `checkpoint_last.pt`: Estado completo para reanudar.
- `best_model.pt`: El modelo con mejor **valor de negocio ($)** obtenido en validación.
- `history.csv`: Registro tabla por época de: Loss, F1, Error de Picking (s), Valor Neto (USD).
- `history.png`: Gráficos de curvas de entrenamiento listos para insertar en el informe final.

---

## 8. Evaluación y Dashboard en Windows

Una vez finalizado el entrenamiento:

### 1. Evaluar el mejor modelo y generar métricas finales:
```powershell
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint checkpoints\best_model.pt
```
Esto actualiza `results\metrics.json` y la gráfica de valor neto `results\value_curve.png`.

### 2. Abrir la interfaz web interactiva (Streamlit):
```powershell
.\.venv\Scripts\streamlit.exe run src\app.py
```
Se abrirá automáticamente en tu navegador `http://localhost:8501`. En la barra lateral podrás seleccionar tu modelo PyTorch entrenado en la RTX 5050 y probarlo en tiempo real con diferentes umbrales.
