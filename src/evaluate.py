"""Ejecuta la evaluación técnica y de negocio del sistema."""

import json
from pathlib import Path

from seismic import evaluate_baseline, evaluate_detector


if __name__ == "__main__":
    result = {"STA/LTA": evaluate_baseline(), "DeepPhase": evaluate_detector()}
    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
