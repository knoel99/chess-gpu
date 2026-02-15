#!/usr/bin/env python3
"""
Entraîne un modèle linéaire (softmax regression) pour prédire le prochain
coup aux échecs à partir de la position du plateau.

    p = softmax(W · x + b)

Utilise CuPy (GPU) si disponible, sinon NumPy (CPU).
Le graphique de suivi est mis à jour en direct à chaque epoch.

Usage:
    python train.py data/training_data.npz
"""

import sys
import os
import time

# Auto-détection GPU : CuPy si disponible, sinon NumPy
try:
    import cupy as xp
    GPU = True
    gpu_name = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"  🟢 GPU détecté : {gpu_name}")
    print(f"     VRAM : {xp.cuda.runtime.memGetInfo()[1] / 1024**3:.1f} Go")
except (ImportError, Exception):
    import numpy as xp
    GPU = False
    print(f"  🔴 Pas de GPU — utilisation de NumPy (CPU)")

import numpy as np  # toujours disponible pour l'I/O et matplotlib


# ---------------------------------------------------------------------------
# 1. Softmax stable
# ---------------------------------------------------------------------------

def softmax(z):
    """Softmax sur le dernier axe, numériquement stable.
    z: (N, C) → retourne (N, C) avec lignes sommant à 1."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = xp.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Cross-entropy loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(P, y):
    """Loss moyenne sur le batch.
    P: (N, C) probabilités, y: (N,) indices des classes correctes."""
    N = len(y)
    p_correct = P[xp.arange(N), y].clip(min=1e-12)
    return float(-xp.log(p_correct).mean())


# ---------------------------------------------------------------------------
# 3. Forward + backward en une passe
# ---------------------------------------------------------------------------

def forward_backward(X, y, W1, b1, W2, b2, n_classes):
    """
    Forward pass + calcul du gradient (réseau 2 couches avec ReLU).

    X:  (N, D) entrées
    y:  (N,) labels
    W1: (H, D) poids couche 1
    b1: (H,) biais couche 1
    W2: (C, H) poids couche 2
    b2: (C,) biais couche 2

    Retourne: loss, dW1, db1, dW2, db2, P
    """
    N = X.shape[0]

    # Couche 1 : linéaire + ReLU
    Z1 = X @ W1.T + b1                 # (N, H)
    A1 = xp.maximum(Z1, 0)             # (N, H) — ReLU

    # Couche 2 : linéaire + softmax
    Z2 = A1 @ W2.T + b2                # (N, C)
    P = softmax(Z2)                    # (N, C)

    loss = cross_entropy_loss(P, y)

    # Gradient couche 2
    E = P.copy()                       # (N, C)
    E[xp.arange(N), y] -= 1.0

    dW2 = (E.T @ A1) / N               # (C, H)
    db2 = E.mean(axis=0)               # (C,)

    # Backprop vers couche 1
    dA1 = E @ W2                       # (N, H)
    dZ1 = dA1 * (Z1 > 0).astype(xp.float32)  # ReLU gradient

    dW1 = (dZ1.T @ X) / N              # (H, D)
    db1 = dZ1.mean(axis=0)             # (H,)

    return loss, dW1, db1, dW2, db2, P


# ---------------------------------------------------------------------------
# 4. Métriques
# ---------------------------------------------------------------------------

def accuracy_topk(P, y, k=1):
    """Précision top-k."""
    if k == 1:
        preds = P.argmax(axis=1)
        return float((preds == y).mean())
    else:
        topk = xp.argpartition(P, -k, axis=1)[:, -k:]
        # Ramener sur CPU pour la comparaison si nécessaire
        topk_cpu = topk if not GPU else topk.get()
        y_cpu = y if not GPU else y.get()
        return np.array([y_cpu[i] in topk_cpu[i] for i in range(len(y_cpu))]).mean()


# ---------------------------------------------------------------------------
# 5. Affichage progression
# ---------------------------------------------------------------------------

def format_time(seconds):
    """Formate un nombre de secondes en HH:MM:SS ou MM:SS."""
    if seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_progress_bar(batch, n_batches, batch_loss, epoch, epochs, elapsed, eta):
    """Affiche une barre de progression pour le batch courant."""
    pct = (batch + 1) / n_batches
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)

    line = (f"\r  Epoch {epoch:3d}/{epochs} │{bar}│ "
            f"{batch+1:>4d}/{n_batches} batches "
            f"│ loss={batch_loss:.4f} "
            f"│ {format_time(elapsed)} elapsed "
            f"│ ETA {format_time(eta)}")
    print(line, end="", flush=True)


def print_epoch_summary(epoch, epochs, metrics, dt, total_elapsed, total_eta):
    """Affiche le résumé de l'epoch sur une nouvelle ligne."""
    print(f"\r  Epoch {epoch:3d}/{epochs} │ "
          f"train={metrics['train_loss']:.4f} "
          f"val={metrics['val_loss']:.4f} │ "
          f"top1={metrics['top1']:.1f}% "
          f"top5={metrics['top5']:.1f}% │ "
          f"‖∇‖={metrics['grad_norm']:.4f} │ "
          f"{dt:.1f}s │ "
          f"restant: {format_time(total_eta)}     ")


# ---------------------------------------------------------------------------
# 6. Graphique live
# ---------------------------------------------------------------------------

def update_plot(history, out_path):
    """Met à jour le graphique de suivi (appelé à chaque epoch)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    if len(epochs) < 1:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Entraînement — Epoch {epochs[-1]} — "
        f"top1={history['top1'][-1]:.1f}% — "
        f"top5={history['top5'][-1]:.1f}%",
        fontsize=14, fontweight="bold"
    )

    # --- 1. Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train", color="#2196F3", markersize=4)
    ax.plot(epochs, history["val_loss"], "s-", label="Val", color="#F44336", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss (cross-entropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for series, color in [("train_loss", "#2196F3"), ("val_loss", "#F44336")]:
        ax.annotate(f" {history[series][-1]:.4f}",
                    xy=(epochs[-1], history[series][-1]),
                    fontsize=9, color=color, fontweight="bold")

    # --- 2. Accuracy ---
    ax = axes[0, 1]
    ax.plot(epochs, history["top1"], "o-", label="Top-1", color="#4CAF50", markersize=4)
    ax.plot(epochs, history["top5"], "s-", label="Top-5", color="#FF9800", markersize=4)
    ax.axhline(y=100/1968*100, color="gray", linestyle="--", alpha=0.5, label="Aléatoire")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Précision de prédiction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    for series, color in [("top1", "#4CAF50"), ("top5", "#FF9800")]:
        ax.annotate(f" {history[series][-1]:.1f}%",
                    xy=(epochs[-1], history[series][-1]),
                    fontsize=9, color=color, fontweight="bold")

    # --- 3. Gradient norm ---
    ax = axes[1, 0]
    ax.plot(epochs, history["grad_norm"], "o-", color="#9C27B0", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("‖∇L‖ (norme L2 moyenne)")
    ax.set_title("Norme du gradient")
    ax.grid(True, alpha=0.3)
    ax.annotate(f" {history['grad_norm'][-1]:.4f}",
                xy=(epochs[-1], history['grad_norm'][-1]),
                fontsize=9, color="#9C27B0", fontweight="bold")
    # Détecter convergence
    if len(epochs) > 3:
        recent = history["grad_norm"][-3:]
        if max(recent) - min(recent) < 0.001:
            ax.text(0.5, 0.9, "⚠ Gradient stable — convergence ?",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color="orange", fontweight="bold")

    # --- 4. Temps & overfitting ---
    ax = axes[1, 1]
    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    ax2 = ax.twinx()
    ax.bar(epochs, history["time"], alpha=0.3, color="#607D8B", label="Temps/epoch")
    ax2.plot(epochs, gap, "D-", color="#E91E63", markersize=4, label="Gap (val−train)")
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Temps (s)", color="#607D8B")
    ax2.set_ylabel("Overfitting gap", color="#E91E63")
    ax.set_title("Temps par epoch & détection d'overfitting")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    # Alerte overfitting
    if len(gap) > 2 and gap[-1] > 0.1 and gap[-1] > gap[-2]:
        ax.text(0.5, 0.9, "⚠ Overfitting détecté",
                transform=ax.transAxes, ha="center", fontsize=10,
                color="red", fontweight="bold")

    # Texte récapitulatif total
    total_time = sum(history["time"])
    fig.text(0.5, 0.01,
             f"Temps total : {format_time(total_time)} │ "
             f"Meilleur top-1 : {max(history['top1']):.1f}% (epoch {history['top1'].index(max(history['top1']))+1}) │ "
             f"Meilleur top-5 : {max(history['top5']):.1f}% (epoch {history['top5'].index(max(history['top5']))+1})",
             ha="center", fontsize=10, style="italic", color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Entraînement
# ---------------------------------------------------------------------------

def train(X, y, n_classes, config, plot_path="data/training_curves.png"):
    """Boucle d'entraînement avec progression live et graphique mis à jour."""
    N, D = X.shape
    H = config.get("hidden", 512)
    lr = config["lr"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    # Initialisation des poids (Xavier) — sur GPU si disponible
    scale1 = float(np.sqrt(2.0 / (D + H)))
    scale2 = float(np.sqrt(2.0 / (H + n_classes)))
    W1 = xp.array(np.random.randn(H, D).astype(np.float32) * scale1)
    b1 = xp.zeros(H, dtype=xp.float32)
    W2 = xp.array(np.random.randn(n_classes, H).astype(np.float32) * scale2)
    b2 = xp.zeros(n_classes, dtype=xp.float32)

    n_params = H * D + H + n_classes * H + n_classes

    # Split train/val (90/10)
    indices = np.random.permutation(N)
    split = int(0.9 * N)
    train_idx, val_idx = indices[:split], indices[split:]

    # Transférer les données sur GPU si disponible
    X_train = xp.array(X[train_idx])
    y_train = xp.array(y[train_idx])
    X_val = xp.array(X[val_idx])
    y_val = xp.array(y[val_idx])

    n_batches = len(y_train) // batch_size

    print(f"\n{'='*70}")
    print(f"  ♟  Entraînement du modèle (832 → {H} → {n_classes})")
    print(f"{'='*70}")
    print(f"  Exemples train : {len(y_train):,}")
    print(f"  Exemples val   : {len(y_val):,}")
    print(f"  Architecture   : {D} → {H} (ReLU) → {n_classes} (softmax)")
    print(f"  Paramètres     : {n_params:,} ({n_params*4/1024/1024:.1f} Mo)")
    print(f"  Learning rate  : {lr}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Epochs         : {epochs}")
    print(f"  Batches/epoch  : {n_batches}")
    print(f"  Graphique live : {plot_path}")
    print(f"  Backend calcul : {'🟢 GPU (CuPy)' if GPU else '🔴 CPU (NumPy)'}")
    print(f"{'='*70}\n")

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "top1": [], "top5": [], "time": [],
        "grad_norm": [], "lr": [],
    }

    total_start = time.time()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Shuffle (indices sur CPU, CuPy accepte l'indexation par array numpy)
        perm = np.random.permutation(len(y_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            Xb = X_train[start:end].astype(xp.float32)
            yb = y_train[start:end]

            loss, dW1, db1_grad, dW2, db2_grad, _ = forward_backward(
                Xb, yb, W1, b1, W2, b2, n_classes)
            epoch_loss += float(loss)
            epoch_grad_norm += float(xp.sqrt(
                (dW1 ** 2).sum() + (db1_grad ** 2).sum() +
                (dW2 ** 2).sum() + (db2_grad ** 2).sum()))

            W1 -= lr * dW1
            b1 -= lr * db1_grad
            W2 -= lr * dW2
            b2 -= lr * db2_grad

            # Barre de progression toutes les 5 batches
            if (i + 1) % 5 == 0 or i == n_batches - 1:
                batch_elapsed = time.time() - t0
                batch_rate = (i + 1) / batch_elapsed if batch_elapsed > 0 else 0
                batch_eta = (n_batches - i - 1) / batch_rate if batch_rate > 0 else 0
                print_progress_bar(i, n_batches, float(loss), epoch, epochs, batch_elapsed, batch_eta)

        epoch_loss /= n_batches
        epoch_grad_norm /= n_batches
        dt = time.time() - t0
        epoch_times.append(dt)

        # Validation
        val_sample = min(2000, len(y_val))
        Xv = X_val[:val_sample].astype(xp.float32)
        yv = y_val[:val_sample]
        Zv1 = Xv @ W1.T + b1
        Av1 = xp.maximum(Zv1, 0)
        Zv2 = Av1 @ W2.T + b2
        Pv = softmax(Zv2)
        val_loss = cross_entropy_loss(Pv, yv)
        val_top1 = accuracy_topk(Pv, yv, k=1)
        val_top5 = accuracy_topk(Pv, yv, k=5)

        # Historique (tout en float Python pour matplotlib)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(epoch_loss))
        history["val_loss"].append(float(val_loss))
        history["top1"].append(float(val_top1 * 100))
        history["top5"].append(float(val_top5 * 100))
        history["time"].append(float(dt))
        history["grad_norm"].append(float(epoch_grad_norm))
        history["lr"].append(float(lr))

        # ETA global
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        total_eta = remaining_epochs * avg_epoch_time

        # Résumé de l'epoch
        metrics = {
            "train_loss": epoch_loss, "val_loss": val_loss,
            "top1": val_top1 * 100, "top5": val_top5 * 100,
            "grad_norm": epoch_grad_norm,
        }
        print_epoch_summary(epoch, epochs, metrics, dt,
                           time.time() - total_start, total_eta)
        print()  # nouvelle ligne

        # Mise à jour du graphique live
        update_plot(history, plot_path)

    # Résumé final
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ✓ Entraînement terminé en {format_time(total_time)}")
    print(f"    Meilleur top-1 : {max(history['top1']):.1f}% "
          f"(epoch {history['top1'].index(max(history['top1']))+1})")
    print(f"    Meilleur top-5 : {max(history['top5']):.1f}% "
          f"(epoch {history['top5'].index(max(history['top5']))+1})")
    print(f"    Loss finale    : train={history['train_loss'][-1]:.4f}, "
          f"val={history['val_loss'][-1]:.4f}")
    print(f"{'='*70}")

    # Ramener les poids sur CPU pour la sauvegarde
    W1_cpu = W1.get() if GPU else W1
    b1_cpu = b1.get() if GPU else b1
    W2_cpu = W2.get() if GPU else W2
    b2_cpu = b2.get() if GPU else b2

    return {"W1": W1_cpu, "b1": b1_cpu, "W2": W2_cpu, "b2": b2_cpu}, history


# ---------------------------------------------------------------------------
# 8. Main
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

    config = {
        "lr": 0.01,
        "batch_size": 256,
        "epochs": 20,
        "hidden": 512,
    }

    plot_path = os.path.splitext(model_path)[0] + "_curves.png"
    weights, history = train(X, y, n_classes, config, plot_path=plot_path)

    np.savez_compressed(model_path, **weights, move_tokens=move_tokens)
    print(f"\nModèle sauvegardé dans {model_path}")
    print(f"  → W1: {weights['W1'].shape}, W2: {weights['W2'].shape}")
    print(f"  → Taille : {os.path.getsize(model_path) / (1024**2):.1f} Mo")
    return model_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <training_data.npz> [model_output.npz]")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/model.npz")


if __name__ == "__main__":
    main()
