# Script de PowerShell para entrenar AlertaP en Windows con GPU RTX 5050 y datos reales STEAD
param (
    [string]$model = "conv1d",
    [int]$epochs = 60,
    [int]$batchSize = 64,
    [string]$source = "stead",
    [string]$steadDir = "data",
    [int]$nTrainEvents = 10000,
    [int]$nTrainNoise = 10000,
    [string]$checkpointDir = "checkpoints_stead",
    [switch]$resume
)

$venvPython = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] No se encontró el entorno virtual en .venv" -ForegroundColor Red
    exit 1
}

$cachePath = "data\stead_subset_${nTrainEvents}ev.npz"

$cmdArgs = @(
    "src\train.py",
    "--model", $model,
    "--source", $source,
    "--stead-dir", $steadDir,
    "--stead-cache", $cachePath,
    "--n-train-events", $nTrainEvents,
    "--n-train-noise", $nTrainNoise,
    "--epochs", $epochs,
    "--batch-size", $batchSize,
    "--checkpoint-dir", $checkpointDir,
    "--resume-auto"
)

if ($resume) {
    $cmdArgs += @("--resume", "$checkpointDir\checkpoint_last.pt")
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  AlertaP - Entrenamiento con Datos Reales STEAD (RTX 5050 CUDA)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Comando: $venvPython $($cmdArgs -join ' ')" -ForegroundColor Yellow
Write-Host ""

& $venvPython @cmdArgs
