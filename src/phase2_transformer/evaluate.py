#!/usr/bin/env python3
"""
Évalue le Transformer contre Stockfish en parties complètes.

Le Transformer maintient un historique des positions pour chaque partie,
permettant à l'attention de capturer le contexte des coups précédents.

Usage:
    python evaluate.py data/transformer_model.pt [--games 10] [--stockfish-elo 1350]
"""

import sys
import os
import argparse
import numpy as np
import chess

import torch

# Ajouter src/ au path pour réutiliser le moteur d'évaluation de la phase 1
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phase2_transformer.train_transformer import ChessTransformer
from common.prepare_sequences import board_to_vector_enriched, N_FEATURES


# ---------------------------------------------------------------------------
# Wrapper : adapter le Transformer à l'interface de evaluate.py phase 1
# ---------------------------------------------------------------------------

class TransformerWrapper:
    """Encapsule le Transformer pour le rendre compatible avec evaluate.py.

    Maintient un historique des positions et construit la séquence
    à chaque appel de prédiction.
    """

    def __init__(self, model, move_tokens, seq_len, device):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.move_tokens = move_tokens
        self.token_to_move = {}
        for i, (f, t, p) in enumerate(move_tokens):
            promo = int(p) if int(p) != 0 else None
            self.token_to_move[i] = chess.Move(int(f), int(t), promotion=promo)

        # Historique des positions pour la partie en cours
        self.position_history = []

    def reset(self):
        """Réinitialiser pour une nouvelle partie."""
        self.position_history = []

    def get_top_moves(self, board, k=10):
        """Retourne les k meilleurs coups légaux avec probabilités."""
        # Encoder la position courante et l'ajouter à l'historique
        vec = board_to_vector_enriched(board)
        self.position_history.append(vec)

        # Construire la séquence
        n_pos = len(self.position_history)
        if n_pos >= self.seq_len:
            seq = np.stack(self.position_history[-self.seq_len:])
        else:
            pad = np.zeros((self.seq_len - n_pos, N_FEATURES), dtype=np.float32)
            seq = np.vstack([pad, np.stack(self.position_history)])

        # Inférence
        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            pad_mask = (x.abs().sum(dim=-1) == 0)

            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(x, padding_mask=pad_mask)
            else:
                logits = self.model(x, padding_mask=pad_mask)

            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

        # Filtrer les coups légaux
        legal_moves = set(board.legal_moves)
        sorted_indices = np.argsort(probs)[::-1]

        top = []
        for idx in sorted_indices:
            move = self.token_to_move.get(int(idx))
            if move and move in legal_moves:
                top.append((move, float(probs[idx])))
                if len(top) >= k:
                    break
        return top

    def record_opponent_move(self, board):
        """Enregistre la position après un coup adverse."""
        vec = board_to_vector_enriched(board)
        self.position_history.append(vec)


def load_transformer(model_path):
    """Charge le Transformer depuis un fichier .pt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    move_tokens = checkpoint["move_tokens"]

    model = ChessTransformer(
        n_features=config["n_features"],
        n_classes=config["n_classes"],
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=0.0,  # pas de dropout en inférence
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    wrapper = TransformerWrapper(model, move_tokens, config["seq_len"], device)
    return wrapper


# ---------------------------------------------------------------------------
# Évaluation (reprise du moteur phase 1 avec adaptation Transformer)
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def material_detail(board):
    w, b = 0, 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                w += val
            else:
                b += val
    return w, b, w - b


def find_stockfish():
    import shutil
    for p in ["/usr/games/stockfish", "/usr/local/bin/stockfish",
              "/usr/bin/stockfish"]:
        if os.path.exists(p):
            return p
    return shutil.which("stockfish")


def play_game(wrapper, engine, model_color, sf_time=0.1,
              game_id=0, log_file=None):
    """Joue une partie complète avec le Transformer."""
    import chess.pgn
    import threading

    _log_lock = threading.Lock()

    wrapper.reset()
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "Transformer" if model_color == chess.WHITE else "Stockfish"
    game.headers["Black"] = "Stockfish" if model_color == chess.WHITE else "Transformer"
    node = game
    move_count = 0
    mat_history = []

    while not board.is_game_over() and move_count < 200:
        if board.turn == model_color:
            import time as _time
            t0 = _time.time()
            top = wrapper.get_top_moves(board, k=1)
            elapsed = _time.time() - t0

            if top:
                move, prob = top[0]
                info = {"prob": prob, "time": elapsed}
            else:
                import random
                move = random.choice(list(board.legal_moves))
                info = {"prob": 0.0, "time": elapsed}

            node = node.add_variation(move)
            node.comment = f"p={info['prob']:.3f} t={info['time']:.3f}s"
            who = "🤖"
        else:
            result = engine.play(board, chess.engine.Limit(time=sf_time))
            move = result.move
            node = node.add_variation(move)
            who = "♟"
            info = None

        board.push(move)
        move_count += 1

        # Enregistrer position adverse pour le contexte
        if who == "♟":
            wrapper.record_opponent_move(board)

        mw, mb, diff = material_detail(board)
        nn_diff = diff if model_color == chess.WHITE else -diff
        mat_history.append((move_count, nn_diff, None, who))

        if log_file is not None:
            move_num = (move_count + 1) // 2
            dot = "." if board.turn == chess.BLACK else "..."
            legal = board.legal_moves.count()
            if info:
                search_str = (f"p={info['prob']:.3f} t={info['time']:.2f}s")
            else:
                search_str = "stockfish"
            with _log_lock:
                log_file.write(
                    f"G{game_id+1:02d} │ {move_num:3d}{dot}{move.uci():6s} {who} │ "
                    f"mat ⬜{mw:2d} ⬛{mb:2d} Δ{diff:+3d} │ "
                    f"legal={legal:3d} │ {search_str}\n"
                )
                log_file.flush()

    result = board.result()
    game.headers["Result"] = result

    if result == "1-0":
        score = 1.0 if model_color == chess.WHITE else 0.0
    elif result == "0-1":
        score = 0.0 if model_color == chess.WHITE else 1.0
    else:
        score = 0.5

    return score, str(game), mat_history


def run(model_path, n_games=10, stockfish_elo=1350, n_workers=1):
    """Lance l'évaluation du Transformer contre Stockfish."""
    import chess.engine
    import json
    from datetime import datetime

    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_transformer"

    print(f"\n{'='*60}")
    print(f"  ♟  Évaluation Transformer vs Stockfish (Elo ~{stockfish_elo})")
    print(f"{'='*60}")

    wrapper = load_transformer(model_path)
    print(f"  Modèle    : {model_path}")
    print(f"  Séquence  : {wrapper.seq_len} positions")
    print(f"  Device    : {wrapper.device}")
    print(f"  Parties   : {n_games}")

    sf_path = find_stockfish()
    if sf_path is None:
        print("❌ Stockfish non trouvé.")
        sys.exit(1)
    print(f"  Stockfish : {sf_path}")

    base = os.path.splitext(model_path)[0]
    out_dir = f"{base}_runs"
    os.makedirs(out_dir, exist_ok=True)

    log_path = f"{out_dir}/{tag}.log"
    log_file = open(log_path, "w")
    log_file.write(f"# Transformer vs Stockfish (Elo ~{stockfish_elo})\n")
    log_file.write(f"# Modèle : {model_path}\n")
    log_file.write(f"{'─'*80}\n")
    log_file.flush()
    print(f"  Log       : {log_path}")
    print(f"{'='*60}\n")

    results = {"win": 0, "draw": 0, "loss": 0}
    scores = []
    pgns = []

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})

    for i in range(n_games):
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        color_str = "blancs" if model_color == chess.WHITE else "noirs"
        print(f"  Partie {i+1}/{n_games} ({color_str})...", end=" ", flush=True)

        score, pgn_str, mat_history = play_game(
            wrapper, engine, model_color,
            sf_time=0.1, game_id=i, log_file=log_file
        )

        scores.append(score)
        pgns.append(pgn_str)

        if score == 1.0:
            results["win"] += 1
            res_str = "✅ Victoire"
        elif score == 0.0:
            results["loss"] += 1
            res_str = "❌ Défaite"
        else:
            results["draw"] += 1
            res_str = "🤝 Nulle"

        print(res_str)

    engine.quit()
    log_file.write(f"{'─'*80}\n")
    log_file.write(f"# Résultat : {results['win']}W - {results['draw']}D - "
                   f"{results['loss']}L\n")
    total_score = sum(scores)
    pct = total_score / n_games * 100
    elo_diff = 400 * (total_score / n_games - 0.5)
    estimated_elo = stockfish_elo + int(elo_diff)
    log_file.write(f"# Score    : {total_score}/{n_games} ({pct:.0f}%)\n")
    log_file.write(f"# Elo estimé : ~{estimated_elo}\n")
    log_file.close()

    print(f"\n{'='*60}")
    print(f"  Résultat : {results['win']}W - {results['draw']}D - {results['loss']}L")
    print(f"  Score    : {total_score}/{n_games} ({pct:.0f}%)")
    print(f"  Elo estimé : ~{estimated_elo}")
    print(f"{'='*60}")

    summary = {
        "model": model_path,
        "model_type": "transformer",
        "stockfish_elo": stockfish_elo,
        "estimated_elo": estimated_elo,
        "n_games": n_games,
        "wins": results["win"],
        "draws": results["draw"],
        "losses": results["loss"],
        "score": total_score,
        "win_rate": total_score / n_games,
        "tag": tag,
    }
    summary_path = f"{out_dir}/{tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  📄 Résumé : {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Évaluer le Transformer vs Stockfish")
    parser.add_argument("model", help="Chemin du modèle .pt")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--stockfish-elo", type=int, default=1350)
    args = parser.parse_args()
    run(args.model, n_games=args.games, stockfish_elo=args.stockfish_elo)


if __name__ == "__main__":
    main()
