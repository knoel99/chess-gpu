#!/usr/bin/env python3
"""
Fait jouer le modèle linéaire contre Stockfish sur des parties complètes.

Stockfish doit être installé :
    apt install stockfish          # Linux / Colab
    brew install stockfish         # macOS

Usage:
    python evaluate.py data/magnuscarlsen_2025_01_model.npz [--games 10] [--stockfish-elo 800]
"""

import sys
import os
import argparse
import numpy as np
import chess
import chess.engine

# Auto-détection GPU
try:
    import cupy as xp
    GPU = True
except (ImportError, Exception):
    import numpy as xp
    GPU = False


# ── Modèle ──────────────────────────────────────────────────────────────────

def board_to_vector(board):
    """Encode le plateau en vecteur one-hot (832 features)."""
    PIECE_MAP = {
        (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
        (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
        (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
        (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8,
        (chess.BISHOP, chess.BLACK): 9, (chess.ROOK, chess.BLACK): 10,
        (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
    }
    vec = np.zeros(832, dtype=np.float32)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            idx = sq * 13  # case vide = catégorie 0
        else:
            cat = PIECE_MAP[(piece.piece_type, piece.color)]
            idx = sq * 13 + cat
        vec[idx] = 1.0
    return vec


def load_model(model_path):
    """Charge le modèle (W, b, move_tokens)."""
    data = np.load(model_path)
    W = xp.array(data["W"])
    b = xp.array(data["b"])
    move_tokens = data["move_tokens"]  # (N, 3) : from, to, promotion
    return W, b, move_tokens


def build_token_to_move(move_tokens):
    """Construit le mapping index → chess.Move."""
    moves = {}
    for i, (f, t, p) in enumerate(move_tokens):
        promo = int(p) if int(p) != 0 else None
        moves[i] = chess.Move(int(f), int(t), promotion=promo)
    return moves


def predict_move(board, W, b, token_to_move):
    """Prédit le meilleur coup légal."""
    x = board_to_vector(board)
    x_gpu = xp.array(x.reshape(1, -1))

    logits = x_gpu @ W.T + b
    # Softmax
    logits = logits - logits.max()
    probs = xp.exp(logits)
    probs = probs / probs.sum()

    if GPU:
        probs = probs.get()
    probs = probs.flatten()

    # Trier par probabilité décroissante, prendre le premier coup légal
    legal_moves = set(board.legal_moves)
    sorted_indices = np.argsort(probs)[::-1]

    for idx in sorted_indices:
        move = token_to_move.get(int(idx))
        if move and move in legal_moves:
            return move, probs[idx]

    # Fallback : coup aléatoire (ne devrait jamais arriver)
    import random
    return random.choice(list(legal_moves)), 0.0


# ── Partie ──────────────────────────────────────────────────────────────────

def play_game(W, b, token_to_move, engine, model_color, time_limit=0.1):
    """Joue une partie complète. Retourne le résultat du point de vue du modèle."""
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < 200:
        if board.turn == model_color:
            move, prob = predict_move(board, W, b, token_to_move)
        else:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            move = result.move

        board.push(move)
        move_count += 1

    result = board.result()

    if result == "1-0":
        return 1.0 if model_color == chess.WHITE else 0.0
    elif result == "0-1":
        return 0.0 if model_color == chess.WHITE else 1.0
    elif result == "1/2-1/2":
        return 0.5
    else:
        return 0.5  # partie non terminée


# ── Main ────────────────────────────────────────────────────────────────────

def find_stockfish():
    """Cherche le binaire Stockfish."""
    import shutil
    path = shutil.which("stockfish")
    if path:
        return path
    candidates = [
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run(model_path, n_games=10, stockfish_elo=800):
    """Lance l'évaluation contre Stockfish."""
    print(f"\n{'='*60}")
    print(f"  ♟  Évaluation : Modèle vs Stockfish (Elo ~{stockfish_elo})")
    print(f"{'='*60}")

    # Charger modèle
    print(f"  Modèle    : {model_path}")
    W, b, move_tokens = load_model(model_path)
    token_to_move = build_token_to_move(move_tokens)
    print(f"  Tokens    : {len(token_to_move)} coups")
    print(f"  Parties   : {n_games}")
    print(f"  Backend   : {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}")

    # Trouver Stockfish
    sf_path = find_stockfish()
    if sf_path is None:
        print("\n❌ Stockfish non trouvé. Installe-le :")
        print("   apt install stockfish    # Linux/Colab")
        print("   brew install stockfish   # macOS")
        sys.exit(1)
    print(f"  Stockfish : {sf_path}")
    print(f"{'='*60}\n")

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    # Limiter le niveau de Stockfish
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": max(stockfish_elo, 1350)})

    results = {"win": 0, "draw": 0, "loss": 0}
    scores = []

    for i in range(n_games):
        # Alterner les couleurs
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        color_str = "Blancs" if model_color == chess.WHITE else "Noirs"

        score = play_game(W, b, token_to_move, engine, model_color)
        scores.append(score)

        if score == 1.0:
            results["win"] += 1
            symbol = "✅"
        elif score == 0.0:
            results["loss"] += 1
            symbol = "❌"
        else:
            results["draw"] += 1
            symbol = "🤝"

        total = i + 1
        win_rate = sum(scores) / total * 100
        print(f"  Partie {total:2d}/{n_games} │ {color_str:7s} │ {symbol} │ "
              f"Score: {results['win']}W-{results['draw']}D-{results['loss']}L │ "
              f"Win rate: {win_rate:.0f}%")

    engine.quit()

    # Résumé
    total_score = sum(scores)
    win_rate = total_score / n_games * 100
    print(f"\n{'='*60}")
    print(f"  Résultat final : {results['win']}W - {results['draw']}D - {results['loss']}L")
    print(f"  Score          : {total_score:.1f}/{n_games} ({win_rate:.0f}%)")
    print(f"  Elo Stockfish  : ~{stockfish_elo}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation modèle vs Stockfish")
    parser.add_argument("model", help="Chemin vers le modèle .npz")
    parser.add_argument("--games", type=int, default=10, help="Nombre de parties (défaut: 10)")
    parser.add_argument("--stockfish-elo", type=int, default=1350, help="Elo de Stockfish (défaut: 1350, min: 1350)")
    args = parser.parse_args()
    run(args.model, args.games, args.stockfish_elo)


if __name__ == "__main__":
    main()
