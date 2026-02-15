#!/usr/bin/env python3
"""Script tout-en-un pour Google Colab — lancer depuis le terminal :
    cd /content && python chess-gpu/run_colab.py            # phase 1 (MLP)
    cd /content && python chess-gpu/run_colab.py --phase2   # phase 2 (Transformer)
    cd /content && python chess-gpu/run_colab.py --all      # les deux
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

phase = "all" if "--all" in sys.argv else "phase2" if "--phase2" in sys.argv else "phase1"

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

# ── 2. Phase 1 — MLP ──
if phase in ("phase1", "all"):
    sh("python src/phase1_mlp/main.py --top10")
    MODEL1 = "data/top_players_model.npz"
    if os.path.exists(MODEL1):
        sh(f"python src/phase1_mlp/evaluate.py {MODEL1} --games 10")
        sh(f"python src/phase1_mlp/evaluate.py {MODEL1} --games 10 --max-depth 4 --max-nodes 1000")

# ── 3. Phase 2 — Transformer ──
if phase in ("phase2", "all"):
    sh("python src/phase2_transformer/main.py --top10")
    MODEL2 = "data/transformer_model.pt"
    if os.path.exists(MODEL2):
        sh(f"python src/phase2_transformer/evaluate.py {MODEL2} --games 10")

# ── 4. Résumé ──
print("\n" + "="*60)
print("  📊 Fichiers générés")
print("="*60, flush=True)
sh("ls -lh data/top_players_model_runs/ 2>/dev/null || echo 'Aucun résultat MLP'")
sh("ls -lh data/transformer_model_runs/ 2>/dev/null || echo 'Aucun résultat Transformer'")
