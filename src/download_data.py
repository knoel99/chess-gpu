#!/usr/bin/env python3
"""
Télécharge les parties d'un joueur depuis l'API Chess.com au format PGN.

Usage:
    python download_data.py <username> [année] [mois]
    python download_data.py magnuscarlsen              # toutes les parties
    python download_data.py magnuscarlsen 2025         # année 2025
    python download_data.py magnuscarlsen 2025 01      # janvier 2025
"""

import sys
import os
import json
import urllib.request
import time

HEADERS = {"User-Agent": "chess-gpu-training/1.0"}
BASE_URL = "https://api.chess.com/pub/player"


def api_get(url):
    """Requête GET avec User-Agent et gestion d'erreurs."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  ⚠ Erreur HTTP {e.code} pour {url}")
        return None


def get_archives(username):
    """Récupère la liste des archives mensuelles d'un joueur."""
    data = api_get(f"{BASE_URL}/{username}/games/archives")
    if data is None:
        print(f"Joueur '{username}' introuvable.")
        sys.exit(1)
    return data["archives"]


def download_pgn(url):
    """Télécharge le PGN d'un mois."""
    pgn_url = url + "/pgn"
    req = urllib.request.Request(pgn_url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode()


def run(username, year=None, month=None):
    """Télécharge les parties et retourne le chemin du fichier PGN."""
    username = username.lower()
    month = month.zfill(2) if month else None

    print(f"♟  Téléchargement des parties de '{username}'...")

    archives = get_archives(username)
    print(f"  {len(archives)} mois d'archives disponibles")

    if year:
        archives = [a for a in archives if f"/{year}/" in a]
        if month:
            archives = [a for a in archives if a.endswith(f"/{month}")]

    if not archives:
        print("Aucune archive ne correspond aux filtres.")
        return None

    print(f"  {len(archives)} mois à télécharger")

    os.makedirs("data", exist_ok=True)
    out_path = f"data/{username}"
    if year:
        out_path += f"_{year}"
    if month:
        out_path += f"_{month}"
    out_path += ".pgn"

    total_games = 0
    with open(out_path, "w") as f:
        for i, url in enumerate(archives):
            parts = url.split("/")
            y, m = parts[-2], parts[-1]
            print(f"  [{i+1}/{len(archives)}] {y}/{m}...", end=" ", flush=True)

            pgn = download_pgn(url)
            games_count = pgn.count('[Event ')
            total_games += games_count
            f.write(pgn)
            f.write("\n")

            print(f"{games_count} parties")
            time.sleep(0.3)

    print(f"\n✓ {total_games} parties sauvegardées dans {out_path}")
    print(f"  Taille : {os.path.getsize(out_path) / 1024 / 1024:.1f} Mo")
    return out_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <username> [année] [mois]")
        sys.exit(1)
    run(sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else None)


if __name__ == "__main__":
    main()
