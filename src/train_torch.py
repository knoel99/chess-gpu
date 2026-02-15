#!/usr/bin/env python3
"""
Entraîne un réseau de neurones profond (PyTorch) pour prédire le prochain
coup aux échecs à partir de la position du plateau.

Architecture : entrée → 1024 (BN+ReLU+Dropout) → 512 (BN+ReLU+Dropout)
                     → 256 (BN+ReLU+Dropout) → n_classes (softmax)

Optimizer  : Adam avec cosine annealing
Régularisation : Dropout + BatchNorm + early stopping

Usage:
    python train_torch.py data/training_data.npz [data/model.npz]
"""

import sys
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Modèle
# ---------------------------------------------------------------------------

class ChessNet(nn.Module):
    """Réseau profond pour la prédiction de coups."""

    def __init__(self, n_features, n_classes, hidden=(1024, 512, 256),
                 dropout=0.3):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Métriques
# ---------------------------------------------------------------------------

def accuracy_topk(logits, y, k=1):
    """Top-k accuracy."""
    with torch.no_grad():
        if k == 1:
            preds = logits.argmax(dim=1)
            return (preds == y).float().mean().item()
        else:
            _, topk = logits.topk(k, dim=1)
            correct = topk.eq(y.unsqueeze(1).expand_as(topk)).any(dim=1)
            return correct.float().mean().item()


# ---------------------------------------------------------------------------
# 3. Affichage
# ---------------------------------------------------------------------------

def format_time(seconds):
    if seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# 4. Graphique live
# ---------------------------------------------------------------------------

def update_plot(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    if len(epochs) < 1:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Entraînement PyTorch — Epoch {epochs[-1]} — "
        f"top1={history['top1'][-1]:.1f}% — "
        f"top5={history['top5'][-1]:.1f}%",
        fontsize=14, fontweight="bold"
    )

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train", color="#2196F3", markersize=4)
    ax.plot(epochs, history["val_loss"], "s-", label="Val", color="#F44336", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["top1"], "o-", label="Top-1", color="#4CAF50", markersize=4)
    ax.plot(epochs, history["top5"], "s-", label="Top-5", color="#FF9800", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Précision (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history["lr"], "o-", color="#9C27B0", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate (cosine)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Overfitting gap
    ax = axes[1, 1]
    gap = [t - v for t, v in zip(history["val_loss"], history["train_loss"])]
    ax.plot(epochs, gap, "o-", color="#E91E63", markersize=4, label="val - train")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overfitting gap")
    ax.set_title("Overfitting")
    ax.legend()
    ax.grid(True, alpha=0.3)

    total_time = sum(history["time"])
    fig.text(0.5, 0.01,
             f"Temps total : {format_time(total_time)} │ "
             f"Meilleur top-1 : {max(history['top1']):.1f}% "
             f"(epoch {history['top1'].index(max(history['top1']))+1}) │ "
             f"Meilleur top-5 : {max(history['top5']):.1f}% "
             f"(epoch {history['top5'].index(max(history['top5']))+1})",
             ha="center", fontsize=10, style="italic", color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Entraînement
# ---------------------------------------------------------------------------

def train(X, y, n_classes, config, plot_path="data/training_curves.png"):
    """Boucle d'entraînement PyTorch."""
    N, D = X.shape
    hidden = config.get("hidden", (1024, 512, 256))
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 512)
    epochs = config.get("epochs", 50)
    dropout = config.get("dropout", 0.3)
    patience = config.get("patience", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  🟢 GPU : {gpu_name} ({vram:.1f} Go)")
    else:
        print(f"  🔴 CPU (pas de GPU)")

    # Split train/val (90/10)
    indices = np.random.permutation(N)
    split = int(0.9 * N)
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]

    # Charger tout sur GPU si assez de VRAM (sinon DataLoader classique)
    data_bytes = (X_train.nbytes + X_val.nbytes) * 2 + (y_train.nbytes + y_val.nbytes)
    use_gpu_data = (device.type == "cuda" and
                    data_bytes < torch.cuda.get_device_properties(0).total_memory * 0.7)

    if use_gpu_data:
        print(f"  ⚡ Chargement complet sur GPU ({data_bytes / 1024**3:.1f} Go)")
        X_train_t = torch.from_numpy(X_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).long().to(device)
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).long().to(device)
        del X_train, X_val, y_train, y_val
        n_train = X_train_t.shape[0]
        n_val_total = X_val_t.shape[0]
        n_train_batches = (n_train + batch_size - 1) // batch_size
        n_val_batches = (n_val_total + batch_size * 2 - 1) // (batch_size * 2)
        train_loader = None
        val_loader = None
    else:
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        n_workers = min(8, os.cpu_count() or 2)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=n_workers, pin_memory=True,
                                  drop_last=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=n_workers, pin_memory=True,
                                persistent_workers=True)

    # Modèle
    model = ChessNet(D, n_classes, hidden=hidden, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Compiler le modèle pour plus de vitesse (PyTorch 2.0+)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            model = torch.compile(model)
            print("  ⚡ torch.compile activé")
        except Exception:
            pass

    # AMP (Mixed Precision) pour accélérer sur GPU
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Optimizer + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                      eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    arch_str = f"{D}"
    for h in hidden:
        arch_str += f" → {h} (BN+ReLU+Drop)"
    arch_str += f" → {n_classes}"

    print(f"\n{'='*70}")
    print(f"  ♟  Entraînement PyTorch ({len(hidden)+1} couches)")
    print(f"{'='*70}")
    print(f"  Exemples train : {len(y_train):,}")
    print(f"  Exemples val   : {len(y_val):,}")
    print(f"  Architecture   : {arch_str}")
    print(f"  Paramètres     : {n_params:,} ({n_params*4/1024/1024:.1f} Mo)")
    print(f"  Optimizer      : Adam (lr={lr}, wd=1e-5)")
    print(f"  Scheduler      : CosineAnnealing → {1e-6}")
    print(f"  Dropout        : {dropout}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Epochs         : {epochs} (patience={patience})")
    print(f"  Graphique live : {plot_path}")
    print(f"  Device         : {device}" +
          (f" ({gpu_name})" if gpu_name else ""))
    print(f"{'='*70}\n")

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "top1": [], "top5": [], "time": [], "lr": [],
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    total_start = time.time()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_gpu_data:
            perm = torch.randperm(n_train, device=device)
            total_batches = n_train_batches
            for i in range(total_batches):
                start = i * batch_size
                end = min(start + batch_size, n_train)
                idx = perm[start:end]
                xb, yb = X_train_t[idx], y_train_t[idx]

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if (i + 1) % 50 == 0 or i == total_batches - 1:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (total_batches - i - 1) / rate if rate > 0 else 0
                    pct = (i + 1) / total_batches
                    bar_len = 30
                    filled = int(bar_len * pct)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  Epoch {epoch:3d}/{epochs} │{bar}│ "
                          f"{i+1:>5d}/{total_batches} │ "
                          f"loss={loss.item():.4f} │ "
                          f"{format_time(elapsed)} │ ETA {format_time(eta)}",
                          end="", flush=True)
        else:
            total_batches = len(train_loader)
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if (i + 1) % 20 == 0 or i == total_batches - 1:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (total_batches - i - 1) / rate if rate > 0 else 0
                    pct = (i + 1) / total_batches
                    bar_len = 30
                    filled = int(bar_len * pct)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  Epoch {epoch:3d}/{epochs} │{bar}│ "
                          f"{i+1:>4d}/{total_batches} │ "
                          f"loss={loss.item():.4f} │ "
                          f"{format_time(elapsed)} │ ETA {format_time(eta)}",
                          end="", flush=True)

        epoch_loss /= n_batches
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        val_top5 = 0.0
        n_val = 0

        with torch.no_grad():
            if use_gpu_data:
                vbs = batch_size * 2
                for i in range(n_val_batches):
                    start = i * vbs
                    end = min(start + vbs, n_val_total)
                    xb, yb = X_val_t[start:end], y_val_t[start:end]
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            logits = model(xb)
                            val_loss += criterion(logits, yb).item() * len(yb)
                    else:
                        logits = model(xb)
                        val_loss += criterion(logits, yb).item() * len(yb)
                    val_top1 += accuracy_topk(logits, yb, k=1) * len(yb)
                    val_top5 += accuracy_topk(logits, yb, k=5) * len(yb)
                    n_val += len(yb)
            else:
                for xb, yb in val_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            logits = model(xb)
                            val_loss += criterion(logits, yb).item() * len(yb)
                    else:
                        logits = model(xb)
                        val_loss += criterion(logits, yb).item() * len(yb)
                    val_top1 += accuracy_topk(logits, yb, k=1) * len(yb)
                    val_top5 += accuracy_topk(logits, yb, k=5) * len(yb)
                    n_val += len(yb)

        val_loss /= n_val
        val_top1 = val_top1 / n_val * 100
        val_top5 = val_top5 / n_val * 100

        dt = time.time() - t0
        epoch_times.append(dt)

        # Historique
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["top1"].append(val_top1)
        history["top5"].append(val_top5)
        history["time"].append(dt)
        history["lr"].append(current_lr)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " ★"
        else:
            epochs_no_improve += 1
            marker = ""

        avg_t = sum(epoch_times) / len(epoch_times)
        total_eta = (epochs - epoch) * avg_t

        print(f"\r  Epoch {epoch:3d}/{epochs} │ "
              f"train={epoch_loss:.4f} val={val_loss:.4f} │ "
              f"top1={val_top1:.1f}% top5={val_top5:.1f}% │ "
              f"lr={current_lr:.1e} │ {dt:.1f}s │ "
              f"restant: {format_time(total_eta)}{marker}     ")
        print()

        update_plot(history, plot_path)

        if epochs_no_improve >= patience:
            print(f"\n  ⏹ Early stopping (pas d'amélioration depuis {patience} epochs)")
            break

    # Restaurer les meilleurs poids
    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ✓ Entraînement terminé en {format_time(total_time)}")
    print(f"    Meilleur top-1 : {max(history['top1']):.1f}% "
          f"(epoch {history['top1'].index(max(history['top1']))+1})")
    print(f"    Meilleur top-5 : {max(history['top5']):.1f}% "
          f"(epoch {history['top5'].index(max(history['top5']))+1})")
    print(f"    Meilleure val loss : {best_val_loss:.4f}")
    print(f"{'='*70}")

    return model, history


# ---------------------------------------------------------------------------
# 6. Export vers format NumPy (compatible evaluate.py)
# ---------------------------------------------------------------------------

def export_to_npz(model, move_tokens, out_path):
    """Exporte le modèle PyTorch en .npz compatible avec evaluate.py.
    
    Le réseau profond est exporté couche par couche :
    W1, b1 = première couche linéaire
    W2, b2 = dernière couche linéaire (output)
    W_hidden_i, b_hidden_i = couches intermédiaires
    """
    state = model.state_dict()

    # Extraire les couches linéaires (ignorer BN, Dropout)
    linear_layers = []
    keys = list(state.keys())
    for k in keys:
        if k.endswith(".weight") and "bn" not in k.lower():
            # Vérifier que c'est bien une couche linéaire (2D)
            if state[k].dim() == 2:
                bias_key = k.replace(".weight", ".bias")
                linear_layers.append((state[k].numpy(), state[bias_key].numpy()))

    # Sauvegarder aussi les paramètres BatchNorm
    bn_layers = []
    for k in keys:
        if "weight" in k and state[k].dim() == 1:
            prefix = k.rsplit(".", 1)[0]
            bn_keys = [f"{prefix}.weight", f"{prefix}.bias",
                       f"{prefix}.running_mean", f"{prefix}.running_var"]
            if all(bk in state for bk in bn_keys):
                bn_layers.append({
                    "weight": state[bn_keys[0]].numpy(),
                    "bias": state[bn_keys[1]].numpy(),
                    "running_mean": state[bn_keys[2]].numpy(),
                    "running_var": state[bn_keys[3]].numpy(),
                })

    save_dict = {"move_tokens": move_tokens, "n_layers": len(linear_layers)}
    for i, (w, b) in enumerate(linear_layers):
        save_dict[f"W{i}"] = w
        save_dict[f"b{i}"] = b
    for i, bn in enumerate(bn_layers):
        for k, v in bn.items():
            save_dict[f"bn{i}_{k}"] = v

    np.savez_compressed(out_path, **save_dict)

    total_params = sum(w.size + b.size for w, b in linear_layers)
    print(f"\n  Modèle exporté : {out_path}")
    print(f"    {len(linear_layers)} couches linéaires, {total_params:,} paramètres")
    print(f"    {len(bn_layers)} couches BatchNorm")
    print(f"    Taille : {os.path.getsize(out_path) / 1024**2:.1f} Mo")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def run(data_path, model_path="data/model.npz"):
    """Lance l'entraînement. Retourne le chemin du modèle."""
    print(f"Chargement de {data_path}...")
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    move_tokens = data["move_tokens"]

    n_classes = len(move_tokens)
    print(f"  → {X.shape[0]:,} exemples, {X.shape[1]} features, {n_classes} classes")

    # Auto-tune batch_size selon VRAM disponible
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb >= 40:
            bs = 8192
        elif vram_gb >= 16:
            bs = 4096
        elif vram_gb >= 8:
            bs = 2048
        else:
            bs = 1024
    else:
        bs = 512

    config = {
        "lr": 1e-3,
        "batch_size": bs,
        "epochs": 50,
        "hidden": (1024, 512, 256),
        "dropout": 0.3,
        "patience": 10,
    }

    plot_path = os.path.splitext(model_path)[0] + "_curves.png"
    model, history = train(X, y, n_classes, config, plot_path=plot_path)

    # Export en .npz compatible evaluate.py
    export_to_npz(model, move_tokens, model_path)
    return model_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <training_data.npz> [model_output.npz]")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/model.npz")


if __name__ == "__main__":
    main()
