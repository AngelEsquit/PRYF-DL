@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo    AlertaP - Lanzador de Entrenamiento PyTorch (Windows / RTX 5050)
echo ======================================================================

REM Verificar entorno virtual
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en .venv\
    echo Por favor crea el entorno con:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo     .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)

REM Comprobar si se pasaron argumentos adicionales
set ARGS=%*
if "%ARGS%"=="" (
    echo [*] No se pasaron argumentos. Usando configuracion recomendada:
    echo     Modelo: conv1d, Epocas: 50, Batch: 64, Resume Automatico: Activado
    echo.
    .venv\Scripts\python.exe src\train.py --model conv1d --epochs 50 --batch-size 64 --resume-auto
) else (
    echo [*] Ejecutando con argumentos personalizados: %ARGS%
    echo.
    .venv\Scripts\python.exe src\train.py %ARGS%
)

echo.
echo ======================================================================
echo Proceso finalizado.
echo ======================================================================
pause
