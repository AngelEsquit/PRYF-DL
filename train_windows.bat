@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo    AlertaP - Entrenamiento con Datos Reales STEAD (RTX 5050 CUDA)
echo ======================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en .venv\
    pause
    exit /b 1
)

set PYTHONPATH=src
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Ejecutar con datos reales STEAD (20,000 ventanas reales de la base de 91 GB)
.venv\Scripts\python.exe src\train.py --source stead --stead-dir data --model conv1d --epochs 60 --batch-size 64 --n-train-events 10000 --n-train-noise 10000 --stead-cache data\stead_subset_10000ev.npz --checkpoint-dir checkpoints_stead --resume-auto %*

echo.
echo ======================================================================
echo Proceso finalizado.
echo ======================================================================
pause
