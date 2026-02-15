#!/usr/bin/env python3
"""
Entraîne un Transformer pour prédire le prochain coup aux échecs
à partir d'une séquence de positions précédentes.

Architecture :
    séquence (seq_len × 846) → projection (d_model=256)
    → positional embedding → 4× TransformerEncoderLayer
    → dernier token → Linear → n_classes (softmax)

Le mécanisme d'attention permet au réseau de capturer les relations
temporelles entre les coups (ex: sacrifice → attaque, ouverture → plan).

Usage:
    python train_transformer.py data/top_players_seq.npz [data/model_transformer.npz]
"""

import sys
import os
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. Modèle Transformer
# ---------------------------------------------------------------------------

class ChessTransformer(nn.Module):
    """Transformer encoder pour la prédiction de coups d'échecs.

    Prend en entrée une séquence de positions encodées et prédit
    le prochain coup via un mécanisme d'attention multi-tête.
    """

    def __init__(self, n_features, n_classes, seq_len=16,
                 d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Projection linéaire des features vers d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional embedding (appris, pas sinusoïdal)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN pour stabilité
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Normalisation finale
        self.norm = nn.LayerNorm(d_model)

        # Tête de classification
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier pour convergence rapide."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, padding_mask=None):
        """
        x: (batch, seq_len, n_features)
        padding_mask: (batch, seq_len) — True = position paddée à ignorer
        """
        # Projection + positional embedding
        x = self.input_proj(x) + self.pos_embedding

        # Masque causal : chaque position ne voit que les précédentes + elle-même
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.seq_len, device=x.device, dtype=x.dtype
        )

        # Transformer encoder
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Prendre le dernier token (position courante)
        x = x[:, -1, :]

        # Normalisation + classification
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2. Métriques
# ---------------------------------------------------------------------------

def accuracy_topk(logits, y, k=1):
    """Top-k accuracy."""
    with torch.no_grad():
        if k == 1:
            return (logits.argmax(dim=1) == y).float().mean().item()
        else:
            _, topk = logits.topk(k, dim=1)
            return topk.eq(y.unsqueeze(1)).any(dim=1).float().mean().item()


# ---------------------------------------------------------------------------
# 3. Affichage
# ---------------------------------------------------------------------------

def format_time(seconds):
    if seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


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
        f"Transformer — Epoch {epochs[-1]} — "
        f"top1={history['top1'][-1]:.1f}% — "
        f"top5={history['top5'][-1]:.1f}%",
        fontsize=14, fontweight="bold"
    )

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train", color="#2196F3", ms=4)
    ax.plot(epochs, history["val_loss"], "s-", label="Val", color="#F44336", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["top1"], "o-", label="Top-1", color="#4CAF50", ms=4)
    ax.plot(epochs, history["top5"], "s-", label="Top-5", color="#FF9800", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Précision")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["lr"], "o-", color="#9C27B0", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.set_title("LR (cosine)")
    ax.grid(True, alpha=0.3); ax.set_yscale("log")

    ax = axes[1, 1]
    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    ax.plot(epochs, gap, "o-", color="#E91E63", ms=4)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap"); ax.set_title("Overfitting")
    ax.grid(True, alpha=0.3)

    total_time = sum(history["time"])
    fig.text(0.5, 0.01,
             f"Temps total : {format_time(total_time)} │ "
             f"Meilleur top-1 : {max(history['top1']):.1f}% "
             f"(epoch {history['top1'].index(max(history['top1']))+1}) │ "
             f"Meilleur top-5 : {max(history['top5']):.1f}%",
             ha="center", fontsize=10, style="italic", color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Entraînement
# ---------------------------------------------------------------------------

class ChessSequenceDataset(torch.utils.data.Dataset):
    """Dataset qui construit les séquences à la volée par slicing.

    Stocke les positions plates (N, 846) en RAM (~56 Go),
    et construit la fenêtre glissante (seq_len, 846) au __getitem__.
    game_start_of[i] est pré-calculé pour chaque exemple (évite searchsorted).
    """

    def __init__(self, positions, y, game_starts, seq_len):
        self.positions = positions   # (N, D) float32
        self.y = y                   # (N,) int32
        self.seq_len = seq_len
        # Pré-calculer le début de partie pour chaque exemple (O(N) une fois)
        self.game_start_of = np.empty(len(y), dtype=np.int64)
        game_indices = np.searchsorted(game_starts, np.arange(len(y)),
                                       side="right") - 1
        self.game_start_of[:] = game_starts[game_indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        game_start = self.game_start_of[idx]
        n_avail = idx - game_start + 1

        if n_avail >= self.seq_len:
            seq = self.positions[idx - self.seq_len + 1: idx + 1]
        else:
            seq = np.zeros((self.seq_len, self.positions.shape[1]),
                           dtype=np.float32)
            seq[self.seq_len - n_avail:] = self.positions[game_start: idx + 1]

        return torch.from_numpy(np.ascontiguousarray(seq)), int(self.y[idx])


def train(positions, y, game_starts, n_classes, config,
          plot_path="data/transformer_curves.png"):
    """Boucle d'entraînement du Transformer."""
    N = len(y)
    S = config["seq_len"]
    D = config["n_features"]
    d_model = config.get("d_model", 256)
    nhead = config.get("nhead", 8)
    num_layers = config.get("num_layers", 4)
    dim_ff = config.get("dim_feedforward", 1024)
    lr = config.get("lr", 3e-4)
    batch_size = config.get("batch_size", 512)
    epochs = config.get("epochs", 50)
    dropout = config.get("dropout", 0.1)
    patience = config.get("patience", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  🟢 GPU : {gpu_name} ({vram:.1f} Go)")
    else:
        print(f"  🔴 CPU (pas de GPU)")

    # Dataset + split train/val (90/10)
    dataset = ChessSequenceDataset(positions, y, game_starts, S)

    indices = np.random.permutation(N)
    split = int(0.9 * N)
    train_idx = indices[:split]
    val_idx = indices[split:]
    n_train = len(train_idx)
    n_val_total = len(val_idx)

    n_cpu = os.cpu_count() or 1
    n_workers_dl = min(12, n_cpu)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        num_workers=n_workers_dl, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size * 2,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        num_workers=n_workers_dl, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # Créer le padding mask (positions où toutes les features sont 0)
    def make_padding_mask(X_batch):
        """True = position paddée (à ignorer dans l'attention)."""
        return (X_batch.abs().sum(dim=-1) == 0)

    # Modèle
    model = ChessTransformer(
        n_features=D, n_classes=n_classes, seq_len=S,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_ff, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            model = torch.compile(model)
            print("  ⚡ torch.compile activé")
        except Exception:
            pass

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Optimizer : AdamW (meilleur pour Transformers que Adam)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.98))
    # Warmup + cosine decay
    warmup_epochs = min(5, epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*70}")
    print(f"  ♟  Entraînement Transformer")
    print(f"{'='*70}")
    print(f"  Exemples train : {n_train:,}")
    print(f"  Exemples val   : {n_val_total:,}")
    print(f"  Séquence       : {S} positions × {D} features")
    print(f"  Architecture   : {num_layers}L × {nhead}H × d={d_model} ff={dim_ff}")
    print(f"  Paramètres     : {n_params:,} ({n_params*4/1024/1024:.1f} Mo)")
    print(f"  Optimizer      : AdamW (lr={lr}, wd=0.01)")
    print(f"  Scheduler      : warmup {warmup_epochs}ep + cosine")
    print(f"  Dropout        : {dropout}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Epochs         : {epochs} (patience={patience})")
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

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pad_mask = make_padding_mask(xb)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb, padding_mask=pad_mask)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb, padding_mask=pad_mask)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (i + 1) % 50 == 0 or i == n_train_batches - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_train_batches - i - 1) / rate if rate > 0 else 0
                pct = (i + 1) / n_train_batches
                filled = int(30 * pct)
                bar = "█" * filled + "░" * (30 - filled)
                print(f"\r  Epoch {epoch:3d}/{epochs} │{bar}│ "
                      f"{i+1:>5d}/{n_train_batches} │ "
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
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pad_mask = make_padding_mask(xb)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb, padding_mask=pad_mask)
                        val_loss += criterion(logits, yb).item() * len(yb)
                else:
                    logits = model(xb, padding_mask=pad_mask)
                    val_loss += criterion(logits, yb).item() * len(yb)

                val_top1 += accuracy_topk(logits, yb, k=1) * len(yb)
                val_top5 += accuracy_topk(logits, yb, k=5) * len(yb)
                n_val += len(yb)

        val_loss /= n_val
        val_top1 = val_top1 / n_val * 100
        val_top5 = val_top5 / n_val * 100

        dt = time.time() - t0
        epoch_times.append(dt)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["top1"].append(val_top1)
        history["top5"].append(val_top5)
        history["time"].append(dt)
        history["lr"].append(current_lr)

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
            print(f"\n  ⏹ Early stopping ({patience} epochs sans amélioration)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ✓ Entraînement terminé en {format_time(total_time)}")
    print(f"    Meilleur top-1 : {max(history['top1']):.1f}%")
    print(f"    Meilleur top-5 : {max(history['top5']):.1f}%")
    print(f"    Meilleure val loss : {best_val_loss:.4f}")
    print(f"{'='*70}")

    return model, history


# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------

def export_model(model, move_tokens, config, out_path):
    """Sauvegarde le Transformer complet en format PyTorch natif.

    Le Transformer est trop complexe pour un export .npz couche par couche,
    on utilise donc le format .pt standard de PyTorch.
    """
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_type": "transformer",
        "state_dict": state,
        "move_tokens": move_tokens,
        "config": config,
    }, out_path)

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"\n  Modèle exporté : {out_path}")
    print(f"    {n_params:,} paramètres ({size_mb:.1f} Mo)")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def run(data_path, model_path="data/transformer_model.pt"):
    """Lance l'entraînement du Transformer."""
    print(f"Chargement de {data_path}...")
    data = np.load(data_path)
    positions = data["positions"]
    y = data["y"]
    game_starts = data["game_starts"]
    move_tokens = data["move_tokens"]
    seq_len = int(data["seq_len"])
    n_features = int(data["n_features"])

    n_classes = len(move_tokens)
    print(f"  → {len(y):,} exemples, {len(game_starts):,} parties")
    print(f"  → positions {positions.shape}, séquence={seq_len}×{n_features}, "
          f"{n_classes} classes")
    print(f"  → RAM positions: {positions.nbytes / 1024**3:.1f} Go")

    # Auto-tune batch_size pour maximiser l'utilisation GPU
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb >= 40:
            bs = 4096
        elif vram_gb >= 16:
            bs = 1024
        elif vram_gb >= 8:
            bs = 512
        else:
            bs = 256
    else:
        bs = 128

    config = {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "lr": 3e-4,
        "batch_size": bs,
        "epochs": 50,
        "dropout": 0.1,
        "patience": 10,
        "seq_len": seq_len,
        "n_features": n_features,
        "n_classes": n_classes,
    }

    plot_path = os.path.splitext(model_path)[0] + "_curves.png"
    model, history = train(positions, y, game_starts, n_classes, config,
                           plot_path=plot_path)

    export_model(model, move_tokens, config, model_path)
    return model_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <training_seq.npz> [model_output.pt]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "data/transformer_model.pt"
    run(sys.argv[1], out)


if __name__ == "__main__":
    main()
