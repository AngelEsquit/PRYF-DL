"""Script de entrenamiento robusto en PyTorch para AlertaP con soporte CUDA y checkpoints tolerantes a fallos.

Optimizado para GPUs NVIDIA (como RTX 5050) y CPU.
Permite pausar y reanudar el entrenamiento en cualquier momento sin perder progreso,
guardando atómicamente el estado del modelo, optimizador, scheduler y AMP scaler.

Uso:
    # 1. Iniciar entrenamiento desde cero (datos sintéticos):
    python src/train.py --model conv1d --epochs 50 --batch-size 64

    # 2. Reanudar automáticamente si se apagó la computadora:
    python src/train.py --resume checkpoints/checkpoint_last.pt

    # 3. Entrenar con dataset real STEAD:
    python src/train.py --source stead --stead-dir RUTA/A/STEAD --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from .models import ConvPhaseNet1D, DeepMLP, build_model
    from .seismic import (
        DEFAULT_VALUE_PARAMS,
        MEAN_S_MINUS_P_S,
        PICK_TOLERANCE_S,
        SAMPLE_RATE,
        WARMUP_SAMPLES,
        SignalExample,
        _features,
        business_value,
        generate_example,
        make_eval_set,
    )
except ImportError:
    from models import ConvPhaseNet1D, DeepMLP, build_model
    from seismic import (
        DEFAULT_VALUE_PARAMS,
        MEAN_S_MINUS_P_S,
        PICK_TOLERANCE_S,
        SAMPLE_RATE,
        WARMUP_SAMPLES,
        SignalExample,
        _features,
        business_value,
        generate_example,
        make_eval_set,
    )


class SeismicWindowDataset(Dataset):
    """Dataset de ventanas sísmicas adaptado para PyTorch."""

    def __init__(
        self,
        examples: list[SignalExample],
        model_type: str = "conv1d",
        sigma_samples: float = 8.0,
    ) -> None:
        self.examples = examples
        self.model_type = model_type.lower()
        self.sigma = sigma_samples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        signal = ex.signal  # shape (1200, 3)
        n = len(signal)

        # Generar target temporal gaussiano centrado en p_arrival
        target = np.zeros(n, dtype=np.float32)
        if ex.label and ex.p_arrival is not None:
            t = np.arange(n)
            target = np.exp(-0.5 * ((t - ex.p_arrival) / self.sigma) ** 2).astype(np.float32)

        if self.model_type in ("conv1d", "phasenet", "convphasenet", "convphasenet1d"):
            # Conv1D espera entrada de forma (3, T)
            feat = np.ascontiguousarray(signal.T, dtype=np.float32)
        else:
            # MLP espera las 4 características temporales por muestra (T, 4)
            feat = _features(signal).astype(np.float32)

        return {
            "x": torch.from_numpy(feat),
            "target": torch.from_numpy(target),
            "label": torch.tensor(ex.label, dtype=torch.float32),
            "p_arrival": torch.tensor(ex.p_arrival if ex.p_arrival is not None else -1, dtype=torch.long),
            "s_arrival": torch.tensor(ex.s_arrival if ex.s_arrival is not None else -1, dtype=torch.long),
        }


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    best_metric: float,
    best_epoch: int,
    history: list[dict[str, Any]],
    args_dict: dict[str, Any],
) -> None:
    """Guarda un checkpoint de forma atómica (usando archivo temporal y rename).

    Esto garantiza que si el usuario apaga la computadora o se corta la energía
    justo mientras se escribe el disco, el archivo previo no quede corrupto.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    payload = {
        "epoch": epoch,
        "model_name": getattr(model, "model_name", args_dict.get("model", "conv1d")),
        "model_kwargs": getattr(model, "init_kwargs", {}),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "history": history,
        "config": args_dict,
        "timestamp": time.time(),
    }

    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def evaluate_model_on_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Evalúa métricas técnicas (Loss, F1, MAE picking) y valor de negocio ($)."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))

    total_loss = 0.0
    total_samples = 0
    tp = fp = fn = 0
    picking_errors = []
    advances = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            p_arrivals = batch["p_arrival"].numpy()
            s_arrivals = batch["s_arrival"].numpy()
            labels = batch["label"].numpy()

            logits = model(x)
            loss = criterion(logits, target)
            total_loss += loss.item() * len(x)
            total_samples += len(x)

            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(probs)):
                p_arr = p_arrivals[i]
                s_arr = s_arrivals[i]
                is_event = bool(labels[i])
                prob_seq = probs[i]

                # Detección: primera muestra que supera el umbral tras WARMUP
                candidates = np.flatnonzero(prob_seq >= threshold)
                candidates = candidates[candidates >= WARMUP_SAMPLES]
                detected_sample = int(candidates[0]) if len(candidates) > 0 else None

                if is_event:
                    if detected_sample is not None and p_arr >= 0:
                        picking_errors.append(abs(detected_sample - p_arr) / SAMPLE_RATE)

                    earliest = p_arr - PICK_TOLERANCE_S * SAMPLE_RATE
                    if detected_sample is not None and earliest <= detected_sample < s_arr:
                        tp += 1
                        adv = (s_arr - detected_sample) / SAMPLE_RATE
                        max_adv = (s_arr - p_arr) / SAMPLE_RATE
                        advances.append(float(min(adv, max_adv)))
                    else:
                        fn += 1
                else:
                    if detected_sample is not None:
                        fp += 1

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0
    mean_error = round(float(np.mean(picking_errors)), 4) if picking_errors else 999.0
    mean_adv = float(np.mean(advances)) if advances else MEAN_S_MINUS_P_S

    val_net = business_value(
        tp=tp,
        fp=fp,
        fn=fn,
        n_windows=total_samples,
        seconds_advance=mean_adv,
        params=DEFAULT_VALUE_PARAMS,
    )

    return {
        "val_loss": round(total_loss / max(1, total_samples), 5),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "picking_error_s": mean_error,
        "business_value": round(val_net, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(path: Path, history: list[dict[str, Any]]) -> None:
    """Genera gráficos de pérdida, F1 y valor neto por época."""
    if len(history) < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        val_f1 = [h["val_f1"] for h in history]
        val_val = [h["val_business_value"] for h in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        # 1. Pérdida
        axes[0].plot(epochs, train_loss, label="Train Loss", color="royalblue")
        axes[0].plot(epochs, val_loss, label="Val Loss", color="coral", linestyle="--")
        axes[0].set_title("Pérdida (BCE Loss)")
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 2. F1 Score
        axes[1].plot(epochs, val_f1, label="Val F1", color="forestgreen", marker="o", markersize=3)
        axes[1].set_title("F1-Score en Validación")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("F1")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 3. Valor de Negocio
        axes[2].plot(epochs, val_val, label="Valor Neto ($)", color="purple", marker="s", markersize=3)
        axes[2].axhline(0, color="gray", linewidth=0.8, linestyle=":")
        axes[2].set_title("Valor Neto de Negocio (USD)")
        axes[2].set_xlabel("Época")
        axes[2].set_ylabel("USD")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] No se pudo guardar gráfico de entrenamiento: {e}")


def train(args: argparse.Namespace) -> None:
    # 1. Dispositivo y aceleración
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print(" AlertaP - Entrenamiento PyTorch con Checkpointing Tolerante a Fallos")
    print("=" * 70)
    print(f"[*] Dispositivo seleccionado: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"[*] Tarjeta Gráfica: {gpu_name} ({total_mem:.2f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
    print(f"[*] Modelo: {args.model}")
    print(f"[*] Directorio de checkpoints: {args.checkpoint_dir}")
    print(f"[*] Precision mixta (AMP): {args.amp and device.type == 'cuda'}")
    print("=" * 70)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = checkpoint_dir / "checkpoint_last.pt"
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    history_csv_path = checkpoint_dir / "history.csv"
    history_plot_path = checkpoint_dir / "history.png"

    # 2. Cargar o generar datos
    print(f"[*] Cargando datos ({args.source})...")
    if args.source == "synthetic":
        rng_train = np.random.default_rng(23361)
        train_examples = [
            generate_example(rng_train, event=ev)
            for ev in [True] * args.n_train_events + [False] * args.n_train_noise
        ]
        val_examples = make_eval_set(seed=777, n_events=args.n_val_events, n_noise=args.n_val_noise)
    else:
        from stead import SteadConfig, load_stead

        splits = load_stead(
            args.stead_dir,
            backend="hdf5",
            config=SteadConfig(n_events=args.n_train_events, n_noise=args.n_train_noise),
            cache=args.stead_cache,
        )
        train_examples = splits["train"]
        val_examples = splits["dev"]

    print(f"    - Entrenamiento: {len(train_examples)} ventanas")
    print(f"    - Validación:    {len(val_examples)} ventanas")

    train_ds = SeismicWindowDataset(train_examples, model_type=args.model)
    val_ds = SeismicWindowDataset(val_examples, model_type=args.model)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Inicializar modelo
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    pos_weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start_epoch = 1
    best_metric = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    # 4. Manejo de reanudación (Resume)
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
    elif args.resume_auto and last_ckpt_path.exists():
        resume_path = last_ckpt_path

    if resume_path and resume_path.exists():
        print(f"\n[RESUME] Cargando estado previo desde: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(f"[RESUME] Reanudando a partir de la época {start_epoch} (Mejor época previa: {best_epoch}, valor: ${best_metric:,.2f})\n")

    if start_epoch > args.epochs:
        print(f"[INFO] El entrenamiento ya alcanzó {args.epochs} épocas. Nada que hacer.")
        return

    # 5. Handler de interrupción limpia (Ctrl+C / SIGINT)
    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        print("\n\n[INTERRUPT] Señal de interrupción detectada (Ctrl+C o apagado de sistema).")
        print("[INTERRUPT] Guardando checkpoint de emergencia antes de salir...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, handle_sigint)
        except (ValueError, OSError):
            pass

    # 6. Bucle de entrenamiento por épocas
    print(f"\nIniciando entrenamiento: épocas {start_epoch} -> {args.epochs}...\n")
    print(f"{'Época':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val F1':^8} | {'Pick Err (s)':^12} | {'Valor Neto ($)':^15} | {'Tiempo':^8}")
    print("-" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        if interrupted:
            break

        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if interrupted:
                break
            x = batch["x"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        if interrupted:
            # Guardar estado de emergencia
            emer_path = checkpoint_dir / "checkpoint_interrupt.pt"
            save_checkpoint(
                emer_path,
                epoch=epoch - 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                history=history,
                args_dict=vars(args),
            )
            print(f"[INTERRUPT] Checkpoint guardado en: {emer_path}")
            print(f"[INTERRUPT] Puedes reanudar en cualquier momento ejecutando:")
            print(f"            python src/train.py --resume {emer_path}")
            sys.exit(0)

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)

        # Validación
        val_metrics = evaluate_model_on_dataset(model, val_loader, device)
        elapsed = time.perf_counter() - epoch_start

        current_val_val = val_metrics["business_value"]
        is_best = current_val_val > best_metric
        if is_best:
            best_metric = current_val_val
            best_epoch = epoch

        # Registro
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": val_metrics["val_loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_picking_error_s": val_metrics["picking_error_s"],
            "val_business_value": val_metrics["business_value"],
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            "epoch_sec": round(elapsed, 2),
        }
        history.append(row)

        star = " ★ BEST" if is_best else ""
        print(
            f"{epoch:^7d} | {train_loss:^10.4f} | {val_metrics['val_loss']:^10.4f} | "
            f"{val_metrics['f1']:^8.4f} | {val_metrics['picking_error_s']:^12.4f} | "
            f"${val_metrics['business_value']:>12,.2f} | {elapsed:^7.1f}s{star}"
        )

        # 7. Guardar checkpoints de forma atómica
        # Siempre se actualiza checkpoint_last.pt
        save_checkpoint(
            last_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
            args_dict=vars(args),
        )

        # Guardar mejor modelo
        if is_best:
            save_checkpoint(
                best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                history=history,
                args_dict=vars(args),
            )

        # Guardar checkpoint periódico si aplica
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(
                periodic_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                history=history,
                args_dict=vars(args),
            )

        # Guardar historial CSV y gráfica tras cada época
        save_history_csv(history_csv_path, history)
        plot_training_curves(history_plot_path, history)

    print("\n" + "=" * 70)
    print(" Entrenamiento finalizado exitosamente")
    print(f"[*] Mejor modelo alcanzado en época {best_epoch} con valor neto ${best_metric:,.2f}")
    print(f"[*] Checkpoint último: {last_ckpt_path}")
    print(f"[*] Mejor modelo:     {best_ckpt_path}")
    print(f"[*] Historial CSV:    {history_csv_path}")
    print(f"[*] Gráfica curvas:   {history_plot_path}")
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrenamiento PyTorch de AlertaP con soporte de reanudación y RTX 5050",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Modelo y arquitectura
    p.add_argument("--model", choices=["conv1d", "mlp"], default="conv1d", help="Arquitectura del modelo")

    # Datos
    p.add_argument("--source", choices=["synthetic", "stead"], default="synthetic", help="Fuente de datos")
    p.add_argument("--stead-dir", type=str, default=None, help="Directorio con merged.csv y merged.hdf5 de STEAD")
    p.add_argument("--stead-cache", type=str, default="data/stead_subset.npz", help="Ruta de caché NPZ")
    p.add_argument("--n-train-events", type=int, default=1500, help="Eventos para entrenamiento")
    p.add_argument("--n-train-noise", type=int, default=1500, help="Ventanas de ruido para entrenamiento")
    p.add_argument("--n-val-events", type=int, default=300, help="Eventos para validación")
    p.add_argument("--n-val-noise", type=int, default=300, help="Ventanas de ruido para validación")

    # Hiperparámetros
    p.add_argument("--epochs", type=int, default=40, help="Número total de épocas a entrenar")
    p.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote (ideal 64 o 128 para RTX 5050 8GB)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate inicial")
    default_workers = 0 if sys.platform == "win32" else 2
    p.add_argument("--workers", type=int, default=default_workers, help="Subprocesos para DataLoader (0 en Windows)")

    # Hardware y optimización
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo de cómputo")
    p.add_argument("--amp", action="store_true", default=True, help="Usar Automatic Mixed Precision (AMP)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Desactivar AMP")

    # Checkpoints y Reanudación
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directorio para guardar checkpoints")
    p.add_argument("--save-freq", type=int, default=10, help="Guardar checkpoint cada N épocas (0 = solo last y best)")
    p.add_argument("--resume", type=str, default=None, help="Ruta al checkpoint .pt desde el cual reanudar")
    p.add_argument("--resume-auto", action="store_true", default=True, help="Reanudar automáticamente si existe checkpoint_last.pt")
    p.add_argument("--no-resume-auto", dest="resume_auto", action="store_false", help="Forzar entrenamiento desde época 1 ignorando checkpoints previos")

    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)
