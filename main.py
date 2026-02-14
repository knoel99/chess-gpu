#!/usr/bin/env python3
"""
Pipeline complète : téléchargement → préparation → entraînement.

Chaque étape est sautée si le fichier de sortie existe déjà.

Usage:
    python main.py <username> [année] [mois]
    python main.py magnuscarlsen 2025 01
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from download_data import run as download
from prepare_data import run as prepare
from train import run as train


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <username> [année] [mois]")
        sys.exit(1)

    username = sys.argv[1].lower()
    year = sys.argv[2] if len(sys.argv) > 2 else None
    month = sys.argv[3] if len(sys.argv) > 3 else None

    # --- Chemins ---
    pgn_path = f"data/{username}"
    if year:
        pgn_path += f"_{year}"
    if month:
        pgn_path += f"_{month.zfill(2)}"
    pgn_path += ".pgn"

    npz_path = pgn_path.replace(".pgn", ".npz")
    model_path = pgn_path.replace(".pgn", "_model.npz")

    print(f"\n{'='*60}")
    print(f"  ♟  Pipeline chess-gpu")
    print(f"{'='*60}")
    print(f"  Joueur  : {username}")
    print(f"  PGN     : {pgn_path}")
    print(f"  Données : {npz_path}")
    print(f"  Modèle  : {model_path}")
    print(f"{'='*60}\n")

    # --- 1. Téléchargement ---
    if os.path.exists(pgn_path):
        size = os.path.getsize(pgn_path) / 1024
        print(f"⏭  Téléchargement : {pgn_path} existe déjà ({size:.0f} Ko)\n")
    else:
        print("📥 Étape 1/3 : Téléchargement des parties\n")
        result = download(username, year, month)
        if result is None:
            print("Erreur lors du téléchargement.")
            sys.exit(1)
        print()

    # --- 2. Préparation ---
    if os.path.exists(npz_path):
        size = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"⏭  Préparation : {npz_path} existe déjà ({size:.1f} Mo)\n")
    else:
        print("🔄 Étape 2/3 : Préparation des données\n")
        prepare(pgn_path, npz_path)
        print()

    # --- 3. Entraînement ---
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"⏭  Entraînement : {model_path} existe déjà ({size:.1f} Mo)\n")
    else:
        print("🧠 Étape 3/3 : Entraînement du modèle\n")
        train(npz_path, model_path)
        print()

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline terminée")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
