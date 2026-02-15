#!/usr/bin/env python3
"""
Pipeline Phase 2 : téléchargement → préparation séquentielle → entraînement Transformer.

Réutilise le téléchargement de la phase 1 (PGN communs dans data/).
Génère des données séquentielles spécifiques au Transformer.

Usage:
    python main.py --top10
    python main.py magnuscarlsen
"""

import sys
import os

# Résoudre les chemins
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.download_data import run as download, run_multi as download_multi
from common.prepare_sequences import run as prepare_sequences
from phase2_transformer.train_transformer import run as train_transformer

SEQ_LEN = 16  # fenêtre de contexte (16 demi-coups = 8 coups complets)


def pipeline(pgn_path, seq_path, model_path, label=""):
    """Exécute préparation séquentielle + entraînement Transformer."""
    print(f"\n{'='*60}")
    print(f"  ♟  Pipeline Phase 2 — Transformer" +
          (f" — {label}" if label else ""))
    print(f"{'='*60}")
    print(f"  PGN       : {pgn_path}")
    print(f"  Séquences : {seq_path}")
    print(f"  Modèle    : {model_path}")
    print(f"  Contexte  : {SEQ_LEN} positions")
    print(f"{'='*60}\n")

    # --- Préparation séquentielle ---
    if os.path.exists(seq_path):
        size = os.path.getsize(seq_path) / (1024 * 1024)
        print(f"⏭  Préparation : {seq_path} existe déjà ({size:.1f} Mo)\n")
    else:
        print("🔄 Préparation des séquences\n")
        prepare_sequences(pgn_path, seq_path, seq_len=SEQ_LEN)
        print()

    # --- Entraînement ---
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"⏭  Entraînement : {model_path} existe déjà ({size:.1f} Mo)\n")
    else:
        print("🧠 Entraînement du Transformer\n")
        train_transformer(seq_path, model_path)
        print()

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline Phase 2 terminée → {model_path}")
    print(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <username|--top10> [année] [mois]")
        sys.exit(1)

    year = sys.argv[2] if len(sys.argv) > 2 else None
    month = sys.argv[3] if len(sys.argv) > 3 else None

    if sys.argv[1] == "--top10":
        pgn_path = os.path.join(DATA_DIR, "top_players.pgn")
        seq_path = os.path.join(DATA_DIR, "top_players_seq.npz")
        model_path = os.path.join(DATA_DIR, "transformer_model.pt")

        if os.path.exists(pgn_path):
            size = os.path.getsize(pgn_path) / (1024 * 1024)
            print(f"⏭  Téléchargement : {pgn_path} existe déjà ({size:.1f} Mo)\n")
        else:
            print("📥 Téléchargement des parties (top 10 joueurs)\n")
            download_multi(year=year, month=month, out_path=pgn_path)

        pipeline(pgn_path, seq_path, model_path, label="Top 10 GM")
    else:
        username = sys.argv[1].lower()
        pgn_path = os.path.join(DATA_DIR, username)
        if year:
            pgn_path += f"_{year}"
        if month:
            pgn_path += f"_{month.zfill(2)}"
        pgn_path += ".pgn"
        seq_path = pgn_path.replace(".pgn", "_seq.npz")
        model_path = pgn_path.replace(".pgn", "_transformer.pt")

        if os.path.exists(pgn_path):
            size = os.path.getsize(pgn_path) / 1024
            print(f"⏭  Téléchargement : {pgn_path} existe déjà ({size:.0f} Ko)\n")
        else:
            print("📥 Téléchargement des parties\n")
            result = download(username, year, month)
            if result is None:
                print("Erreur lors du téléchargement.")
                sys.exit(1)

        pipeline(pgn_path, seq_path, model_path, label=username)


if __name__ == "__main__":
    main()
