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
    results = {}

    # Téléchargement parallèle en fonction des ressources
    n_archives = len(archives)
    max_workers = min(n_archives, max(1, min(os.cpu_count() or 1, 8)))

    if max_workers > 1 and n_archives > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"  Workers   : {max_workers} (parallèle)")

        def _dl(idx_url):
            idx, url = idx_url
            parts = url.split("/")
            y, m = parts[-2], parts[-1]
            pgn = download_pgn(url)
            count = pgn.count('[Event ')
            return idx, y, m, pgn, count

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_dl, (i, url)): i
                       for i, url in enumerate(archives)}
            for future in as_completed(futures):
                idx, y, m, pgn, count = future.result()
                results[idx] = pgn
                total_games += count
                done = len(results)
                print(f"  [{done}/{n_archives}] {y}/{m} → {count} parties")
                sys.stdout.flush()

        # Écrire dans l'ordre
        with open(out_path, "w") as f:
            for idx in range(n_archives):
                f.write(results[idx])
                f.write("\n")
    else:
        with open(out_path, "w") as f:
            for i, url in enumerate(archives):
                parts = url.split("/")
                y, m = parts[-2], parts[-1]
                print(f"  [{i+1}/{n_archives}] {y}/{m}...", end=" ", flush=True)

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


# Top 10 joueurs d'échecs (comptes Chess.com)
TOP_PLAYERS = [
    "magnuscarlsen",
    "hikaru",
    "faaborovsky",       # Fabiano Caruana
    "lachesisq",         # Ian Nepomniachtchi
    "firouzja2003",      # Alireza Firouzja
    "duhless",           # Sergey Karjakin (ancien compte)
    "gmwso",             # Wesley So
    "leinier",           # Leinier Dominguez
    "rpragchess",        # Praggnanandhaa
    "daniil_dubov",      # Daniil Dubov
]


def run_multi(players=None, year=None, month=None, out_path=None):
    """Télécharge les parties de plusieurs joueurs et fusionne en un seul PGN.
    
    Args:
        players: liste de usernames (défaut: TOP_PLAYERS)
        year/month: filtres temporels
        out_path: chemin de sortie (défaut: data/top_players.pgn)
    
    Returns: chemin du fichier PGN fusionné
    """
    if players is None:
        players = TOP_PLAYERS
    if out_path is None:
        out_path = "data/top_players.pgn"

    print(f"\n{'='*60}")
    print(f"  ♟  Téléchargement multi-joueurs ({len(players)} joueurs)")
    print(f"{'='*60}")
    for p in players:
        print(f"    • {p}")
    print(f"{'='*60}\n")

    os.makedirs("data", exist_ok=True)
    all_paths = []
    total_games = 0

    for i, player in enumerate(players):
        print(f"\n── [{i+1}/{len(players)}] {player} ──")
        try:
            path = run(player, year=year, month=month)
            if path:
                all_paths.append(path)
        except Exception as e:
            print(f"  ⚠ Erreur pour {player}: {e}")

    # Fusionner tous les PGN
    print(f"\n🔗 Fusion de {len(all_paths)} fichiers...")
    with open(out_path, "w") as out_f:
        for path in all_paths:
            with open(path, "r") as in_f:
                content = in_f.read()
                games = content.count("[Event ")
                total_games += games
                out_f.write(content)
                out_f.write("\n")

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"  ✓ {total_games} parties fusionnées dans {out_path}")
    print(f"    Joueurs : {len(all_paths)}/{len(players)}")
    print(f"    Taille  : {size_mb:.1f} Mo")
    print(f"{'='*60}")
    return out_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <username|--top10> [année] [mois]")
        sys.exit(1)

    year = sys.argv[2] if len(sys.argv) > 2 else None
    month = sys.argv[3] if len(sys.argv) > 3 else None

    if sys.argv[1] == "--top10":
        run_multi(year=year, month=month)
    else:
        run(sys.argv[1], year, month)


if __name__ == "__main__":
    main()
