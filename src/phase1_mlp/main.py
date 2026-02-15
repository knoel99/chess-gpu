#!/usr/bin/env python3
"""
Pipeline complète : téléchargement → préparation → entraînement.

Chaque étape est sautée si le fichier de sortie existe déjà.

Usage:
    python main.py <username> [année] [mois]
    python main.py magnuscarlsen 2025 01
    python main.py --top10                    # top 10 joueurs mondiaux
    python main.py --top10 2025               # top 10, année 2025
"""

import sys
import os

# Résoudre les chemins : data/ est à la racine du repo
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Ajouter src/ au path pour les imports common et phase1
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.download_data import run as download, run_multi as download_multi
from common.prepare_data import run as prepare
try:
    from phase1_mlp.train_torch import run as train
except ImportError:
    from phase1_mlp.train import run as train


def pipeline(pgn_path, npz_path, model_path, label=""):
    """Exécute préparation + entraînement sur un fichier PGN."""
    print(f"\n{'='*60}")
    print(f"  ♟  Pipeline chess-gpu" + (f" — {label}" if label else ""))
    print(f"{'='*60}")
    print(f"  PGN     : {pgn_path}")
    print(f"  Données : {npz_path}")
    print(f"  Modèle  : {model_path}")
    print(f"{'='*60}\n")

    # --- Préparation ---
    if os.path.exists(npz_path):
        size = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"⏭  Préparation : {npz_path} existe déjà ({size:.1f} Mo)\n")
    else:
        print("🔄 Préparation des données\n")
        prepare(pgn_path, npz_path)
        print()

    # --- Entraînement ---
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"⏭  Entraînement : {model_path} existe déjà ({size:.1f} Mo)\n")
    else:
        print("🧠 Entraînement du modèle\n")
        train(npz_path, model_path)
        print()

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline terminée → {model_path}")
    print(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <username|--top10> [année] [mois]")
        sys.exit(1)

    year = sys.argv[2] if len(sys.argv) > 2 else None
    month = sys.argv[3] if len(sys.argv) > 3 else None

    if sys.argv[1] == "--top10":
        pgn_path = os.path.join(DATA_DIR, "top_players.pgn")
        npz_path = os.path.join(DATA_DIR, "top_players.npz")
        model_path = os.path.join(DATA_DIR, "top_players_model.npz")

        # Téléchargement
        if os.path.exists(pgn_path):
            size = os.path.getsize(pgn_path) / (1024 * 1024)
            print(f"⏭  Téléchargement : {pgn_path} existe déjà ({size:.1f} Mo)\n")
        else:
            print("📥 Téléchargement des parties (top 10 joueurs)\n")
            download_multi(year=year, month=month, out_path=pgn_path)

        pipeline(pgn_path, npz_path, model_path, label="Top 10 GM")
    else:
        username = sys.argv[1].lower()

        pgn_path = os.path.join(DATA_DIR, username)
        if year:
            pgn_path += f"_{year}"
        if month:
            pgn_path += f"_{month.zfill(2)}"
        pgn_path += ".pgn"
        npz_path = pgn_path.replace(".pgn", ".npz")
        model_path = pgn_path.replace(".pgn", "_model.npz")

        # Téléchargement
        if os.path.exists(pgn_path):
            size = os.path.getsize(pgn_path) / 1024
            print(f"⏭  Téléchargement : {pgn_path} existe déjà ({size:.0f} Ko)\n")
        else:
            print("📥 Téléchargement des parties\n")
            result = download(username, year, month)
            if result is None:
                print("Erreur lors du téléchargement.")
                sys.exit(1)

        pipeline(pgn_path, npz_path, model_path, label=username)


if __name__ == "__main__":
    main()
