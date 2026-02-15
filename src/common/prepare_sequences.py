#!/usr/bin/env python3
"""
Convertit des fichiers PGN en données de positions pour le Transformer.

Stocke les positions plates (N, 846) avec les limites de parties,
permettant de construire les séquences à la volée dans le DataLoader
(évite l'explosion mémoire ×16 des fenêtres glissantes).

Format de sortie :
  - positions : (n_examples, n_features)  — positions enrichies
  - y         : (n_examples,)             — index du coup à prédire
  - game_starts : (n_games,)              — indice de début de chaque partie

Usage:
    python prepare_sequences.py data/top_players.pgn data/top_players_seq.npz [--seq-len 16]
"""

import sys
import os
import numpy as np
import chess
import chess.pgn
import io

# Réutiliser le dictionnaire de coups de la phase 1
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prepare_data import build_move_dict, move_to_token, _split_pgn, _parse_game

# ---------------------------------------------------------------------------
# 1. Encodage enrichi d'une position (846 features)
# ---------------------------------------------------------------------------

PIECE_TO_IDX = {
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7, (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9, (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11, (chess.KING, chess.BLACK): 12,
}

N_BOARD_FEATURES = 832  # 64 × 13
N_EXTRA_FEATURES = 14   # turn(1) + castling(4) + en_passant(8) + move_number(1)
N_FEATURES = N_BOARD_FEATURES + N_EXTRA_FEATURES  # 846


def board_to_vector_enriched(board):
    """Encode un chess.Board en vecteur enrichi (846,) float32.

    832 features one-hot pour les pièces +
    1 feature pour le tour (1=blanc, 0=noir) +
    4 features pour les droits de roque +
    8 features one-hot pour la colonne d'en-passant +
    1 feature pour le numéro de coup normalisé
    """
    x = np.zeros(N_FEATURES, dtype=np.float32)

    # Pièces (832 features)
    for sq in range(64):
        x[sq * 13] = 1.0  # case vide par défaut
    for sq, piece in board.piece_map().items():
        idx = PIECE_TO_IDX[(piece.piece_type, piece.color)]
        x[sq * 13] = 0.0
        x[sq * 13 + idx] = 1.0

    offset = N_BOARD_FEATURES

    # Tour (1 feature)
    x[offset] = 1.0 if board.turn == chess.WHITE else 0.0
    offset += 1

    # Droits de roque (4 features)
    x[offset] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    x[offset + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    x[offset + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    x[offset + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    offset += 4

    # En-passant (8 features one-hot pour la colonne)
    ep = board.ep_square
    if ep is not None:
        col = chess.square_file(ep)
        x[offset + col] = 1.0
    offset += 8

    # Numéro de coup normalisé (1 feature)
    x[offset] = min(board.fullmove_number / 100.0, 1.0)

    return x


# ---------------------------------------------------------------------------
# 2. Parsing d'une partie en positions plates
# ---------------------------------------------------------------------------

def _parse_game_enriched(game_text, move_to_idx):
    """Parse une partie PGN et retourne les positions enrichies + labels.

    Retourne (X_list, y_list, n_skipped) — positions plates, pas de séquences.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
    except Exception:
        return [], [], 0

    if game is None:
        return [], [], 0

    variant = game.headers.get("Variant", "Standard")
    if variant not in ("Standard", "standard", ""):
        return [], [], 0

    board = game.board()
    X_list = []
    y_list = []
    n_skipped = 0

    for move in game.mainline_moves():
        vec = board_to_vector_enriched(board)
        token = move_to_token(move, move_to_idx)
        if token is not None:
            X_list.append(vec)
            y_list.append(token)
        else:
            n_skipped += 1
        board.push(move)

    return X_list, y_list, n_skipped


def _parse_batch_enriched(args):
    """Parse un batch de parties en positions plates.
    Module-level pour ProcessPoolExecutor."""
    batch, move_to_idx = args
    all_x, all_y = [], []
    game_lengths = []
    n_skipped = 0
    for text in batch:
        xl, yl, sk = _parse_game_enriched(text, move_to_idx)
        if yl:
            all_x.extend(xl)
            all_y.extend(yl)
            game_lengths.append(len(yl))
        n_skipped += sk
    return all_x, all_y, game_lengths, len(batch), n_skipped


# ---------------------------------------------------------------------------
# 3. Parsing PGN complet
# ---------------------------------------------------------------------------

def parse_pgn_positions(pgn_path, move_to_idx):
    """Parse un fichier PGN et retourne positions plates + game_starts.

    Retourne (X, y, game_starts, n_games, n_skipped).
      X : (N, 846) float32
      y : (N,) int32
      game_starts : (n_valid_games,) int64 — indice de début de chaque partie
    """
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t0 = _time.time()
    print(f"  Découpage du PGN...", end=" ", flush=True)
    game_texts = _split_pgn(pgn_path)
    n_total = len(game_texts)
    print(f"{n_total} parties trouvées ({_time.time()-t0:.1f}s)")

    n_workers = max(1, min(os.cpu_count() or 1, 8))
    batch_size = max(100, n_total // (n_workers * 4))
    batches = [game_texts[i:i+batch_size] for i in range(0, n_total, batch_size)]
    print(f"  Parsing positions avec {n_workers} workers...", flush=True)

    X_list = []
    y_list = []
    game_lengths = []
    n_games = 0
    n_skipped_moves = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_parse_batch_enriched, (b, move_to_idx)): i
            for i, b in enumerate(batches)
        }
        done = 0
        for future in as_completed(futures):
            bx, by, gl, ng, sk = future.result()
            X_list.extend(bx)
            y_list.extend(by)
            game_lengths.extend(gl)
            n_games += ng
            n_skipped_moves += sk
            done += 1
            elapsed = _time.time() - t0
            pct = done / len(batches) * 100
            print(f"  [{done}/{len(batches)}] {pct:.0f}% │ "
                  f"{n_games} parties │ {len(y_list):,} exemples │ "
                  f"{elapsed:.0f}s", flush=True)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    game_starts = np.zeros(len(game_lengths), dtype=np.int64)
    cumsum = 0
    for i, gl in enumerate(game_lengths):
        game_starts[i] = cumsum
        cumsum += gl

    return X, y, game_starts, n_games, n_skipped_moves


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def run(pgn_path, out_path, seq_len=16):
    """Convertit un PGN en données de positions .npz."""
    print(f"Construction du dictionnaire de coups...")
    move_to_idx, idx_to_move = build_move_dict()
    print(f"  → {len(move_to_idx)} tokens")

    print(f"Parsing de {pgn_path}...")
    X, y, game_starts, n_games, n_skipped = parse_pgn_positions(
        pgn_path, move_to_idx)
    print(f"  → {n_games} parties, {len(y):,} exemples, {n_skipped} coups ignorés")
    print(f"  → X shape: {X.shape}, y shape: {y.shape}")
    print(f"  → {len(game_starts)} parties avec coups valides")

    move_tokens = np.array(
        [(f, t, p if p else 0) for (f, t, p) in
         [idx_to_move[i] for i in range(len(idx_to_move))]],
        dtype=np.int32
    )

    np.savez_compressed(out_path, positions=X, y=y,
                        game_starts=game_starts,
                        move_tokens=move_tokens,
                        seq_len=seq_len, n_features=N_FEATURES)
    print(f"  → Sauvegardé dans {out_path}")

    file_size = os.path.getsize(out_path)
    print(f"  → Taille : {file_size / (1024**3):.2f} Go")
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", help="Fichier PGN d'entrée")
    parser.add_argument("output", help="Fichier .npz de sortie")
    parser.add_argument("--seq-len", type=int, default=16,
                        help="Longueur de la séquence (défaut: 16)")
    args = parser.parse_args()
    run(args.pgn, args.output, args.seq_len)


if __name__ == "__main__":
    main()
