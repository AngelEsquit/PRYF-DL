# Script de PowerShell para entrenar AlertaP en Windows con GPU RTX 5050
param (
    [string]$model = "conv1d",
    [int]$epochs = 50,
    [int]$batchSize = 64,
    [string]$source = "synthetic",
    [string]$steadDir = "",
    [switch]$resume
)

$venvPython = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] No se encontró el entorno virtual en .venv" -ForegroundColor Red
    Write-Host "Crea el entorno con:"
    Write-Host "    python -m venv .venv"
    Write-Host "    .\.venv\Scripts\pip install -r requirements.txt"
    Write-Host "    .\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128"
    exit 1
}

$cmdArgs = @("src\train.py", "--model", $model, "--epochs", $epochs, "--batch-size", $batchSize, "--source", $source, "--resume-auto")

if ($steadDir -ne "") {
    $cmdArgs += @("--stead-dir", $steadDir)
}

if ($resume) {
    $cmdArgs += @("--resume", "checkpoints\checkpoint_last.pt")
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  AlertaP - Iniciando Entrenamiento en Windows (RTX 5050 CUDA)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Comando: $venvPython $($cmdArgs -join ' ')" -ForegroundColor Yellow
Write-Host ""

& $venvPython @cmdArgs
