#!/usr/bin/env python3
"""Script tout-en-un pour Google Colab — lancer depuis le terminal :
    cd /content && python chess-gpu/run_colab.py
"""
import os, sys, subprocess

def sh(cmd):
    print(f"\n{'='*60}")
    print(f"  ▶ {cmd}")
    print('='*60, flush=True)
    r = subprocess.call(cmd, shell=True)
    if r != 0:
        print(f"⚠ Exit code {r}")
    return r

# ── 0. Stockfish ──
if not os.path.exists("/usr/games/stockfish"):
    sh("apt install -y stockfish -qq")

# ── 1. Repo ──
if not os.path.exists("/content/chess-gpu"):
    sh("git clone https://github.com/knoel99/chess-gpu.git /content/chess-gpu")
    sh("cd /content/chess-gpu && pip install -r requirements.txt -q")
else:
    sh("cd /content/chess-gpu && git pull && pip install -r requirements.txt -q")

os.chdir("/content/chess-gpu")
os.environ["PYTHONUNBUFFERED"] = "1"

# ── 2. Download + Prepare + Train ──
sh("python src/phase1_mlp/main.py --top10")

# ── 3. Évaluations (seulement si modèle existe) ──
MODEL = "data/top_players_model.npz"
if os.path.exists(MODEL):
    sh(f"python src/phase1_mlp/evaluate.py {MODEL} --games 10")
    sh(f"python src/phase1_mlp/evaluate.py {MODEL} --games 10 --max-depth 4 --max-nodes 1000")
    sh(f"python src/phase1_mlp/evaluate.py {MODEL} --games 10 --benchmark --max-depth 4 --max-nodes 1000")

    # ── 4. Résumé des résultats ──
    print("\n" + "="*60)
    print("  📊 Fichiers générés")
    print("="*60, flush=True)
    sh("ls -lh data/top_players_model_runs/ 2>/dev/null || echo 'Aucun résultat'")
else:
    print(f"\n⚠ Modèle {MODEL} introuvable — entraînement échoué ?")
