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
import chess.pgn

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
    """Charge le modèle 2 couches (W1, b1, W2, b2, move_tokens)."""
    data = np.load(model_path)
    W1 = xp.array(data["W1"])
    b1 = xp.array(data["b1"])
    W2 = xp.array(data["W2"])
    b2 = xp.array(data["b2"])
    move_tokens = data["move_tokens"]
    return W1, b1, W2, b2, move_tokens


def build_token_to_move(move_tokens):
    """Construit le mapping index → chess.Move."""
    moves = {}
    for i, (f, t, p) in enumerate(move_tokens):
        promo = int(p) if int(p) != 0 else None
        moves[i] = chess.Move(int(f), int(t), promotion=promo)
    return moves


def predict_move(board, W1, b1, W2, b2, token_to_move):
    """Prédit le meilleur coup légal (réseau 2 couches)."""
    x = board_to_vector(board)
    x_gpu = xp.array(x.reshape(1, -1))

    # Couche 1 : ReLU
    z1 = x_gpu @ W1.T + b1
    a1 = xp.maximum(z1, 0)

    # Couche 2 : softmax
    logits = a1 @ W2.T + b2
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

def play_game(W1, b1, W2, b2, token_to_move, engine, model_color, time_limit=0.1):
    """Joue une partie complète. Retourne (score, pgn_string)."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "Modèle" if model_color == chess.WHITE else f"Stockfish"
    game.headers["Black"] = "Stockfish" if model_color == chess.WHITE else "Modèle"
    node = game
    move_count = 0

    while not board.is_game_over() and move_count < 200:
        if board.turn == model_color:
            move, prob = predict_move(board, W1, b1, W2, b2, token_to_move)
            node = node.add_variation(move)
            node.comment = f"p={prob:.3f}"
        else:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            move = result.move
            node = node.add_variation(move)

        board.push(move)
        move_count += 1

    result = board.result()
    game.headers["Result"] = result

    if result == "1-0":
        score = 1.0 if model_color == chess.WHITE else 0.0
    elif result == "0-1":
        score = 0.0 if model_color == chess.WHITE else 1.0
    elif result == "1/2-1/2":
        score = 0.5
    else:
        score = 0.5

    return score, str(game)


# ── Main ────────────────────────────────────────────────────────────────────

def generate_html(pgns, results, stockfish_elo, out_path):
    """Génère un fichier HTML interactif avec échiquier navigable."""
    import json
    pgns_json = json.dumps(pgns)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>♟ Modèle vs Stockfish (Elo {stockfish_elo})</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
         display: flex; flex-direction: column; align-items: center; padding: 20px; }}
  h1 {{ margin-bottom: 10px; }}
  .score {{ font-size: 1.2em; margin-bottom: 20px; color: #aaa; }}
  .game-select {{ margin-bottom: 15px; }}
  select, button {{ font-size: 16px; padding: 8px 16px; border-radius: 6px;
                    border: none; cursor: pointer; margin: 0 4px; }}
  select {{ background: #16213e; color: #eee; }}
  button {{ background: #0f3460; color: #eee; }}
  button:hover {{ background: #533483; }}
  button:disabled {{ opacity: 0.3; cursor: default; }}
  #board {{ font-size: 0; margin: 10px 0; }}
  .row {{ display: flex; }}
  .sq {{ width: 60px; height: 60px; display: flex; align-items: center;
         justify-content: center; font-size: 40px; user-select: none; }}
  .light {{ background: #e8d5b5; }}
  .dark {{ background: #b58863; }}
  .highlight {{ box-shadow: inset 0 0 0 3px #f5c542; }}
  .info {{ margin-top: 15px; font-size: 14px; color: #aaa; text-align: center; }}
  .move-list {{ max-width: 500px; margin-top: 10px; font-family: monospace;
                font-size: 13px; color: #ccc; line-height: 1.6; word-wrap: break-word;
                text-align: center; }}
  .move-list .current {{ color: #f5c542; font-weight: bold; }}
</style>
</head>
<body>
<h1>♟ Modèle vs Stockfish</h1>
<div class="score">{results['win']}W - {results['draw']}D - {results['loss']}L │ Elo Stockfish: ~{stockfish_elo}</div>

<div class="game-select">
  <select id="gameSelect" onchange="loadGame(this.value)"></select>
</div>
<div id="board"></div>
<div>
  <button onclick="goTo(0)">⏮</button>
  <button onclick="step(-1)">◀</button>
  <button id="playBtn" onclick="togglePlay()">▶ Play</button>
  <button onclick="step(1)">▶</button>
  <button onclick="goTo(positions.length-1)">⏭</button>
</div>
<div class="info" id="info"></div>
<div class="move-list" id="moveList"></div>

<script>
const PIECES = {{
  'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔',
  'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'
}};
const pgns = {pgns_json};
let positions = [], moves = [], currentPos = 0, playInterval = null;

// Parse PGN simplifié
function parsePGN(pgn) {{
  const lines = pgn.split('\\n');
  let moveText = '';
  for (const line of lines) {{
    if (!line.startsWith('[')) moveText += ' ' + line;
  }}
  moveText = moveText.replace(/\\{{[^}}]*\\}}/g, '').replace(/\\d+\\./g, '').trim();
  return moveText.split(/\\s+/).filter(t => t && !['1-0','0-1','1/2-1/2','*'].includes(t));
}}

function fenToBoard(fen) {{
  const rows = fen.split(' ')[0].split('/');
  const board = [];
  for (const row of rows) {{
    const r = [];
    for (const c of row) {{
      if (c >= '1' && c <= '8') for (let i = 0; i < +c; i++) r.push('');
      else r.push(c);
    }}
    board.push(r);
  }}
  return board;
}}

// Mini chess engine pour rejouer les coups
function initBoard() {{
  return 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
}}

function loadGame(idx) {{
  // Parse via le PGN et générer les positions FEN avec un mini-moteur
  // On utilise une approche simple : on lit le PGN move par move
  const pgn = pgns[idx];
  const moveTokens = parsePGN(pgn);

  // Reset — on simule via fetch vers une micro lib embarquée
  positions = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'];
  moves = moveTokens;

  // Pour jouer les coups, on utilise chess.js embarqué minimal
  if (typeof Chess !== 'undefined') {{
    const game = new Chess();
    for (const m of moveTokens) {{
      const result = game.move(m);
      if (!result) break;
      positions.push(game.fen());
    }}
  }}

  currentPos = 0;
  render();
  renderMoveList();

  // Headers
  const headers = {{}};
  pgn.split('\\n').forEach(l => {{
    const m = l.match(/^\\[(\w+)\\s+"(.+)"\\]/);
    if (m) headers[m[1]] = m[2];
  }});
  document.getElementById('info').textContent =
    `${{headers.White || '?'}} vs ${{headers.Black || '?'}} — ${{headers.Result || '?'}}`;
}}

function render() {{
  const fen = positions[currentPos] || positions[0];
  const board = fenToBoard(fen);
  let html = '', lastFrom = -1, lastTo = -1;

  for (let r = 0; r < 8; r++) {{
    html += '<div class="row">';
    for (let c = 0; c < 8; c++) {{
      const light = (r + c) % 2 === 0;
      const piece = board[r][c];
      html += `<div class="sq ${{light?'light':'dark'}}">${{piece ? PIECES[piece] || '' : ''}}</div>`;
    }}
    html += '</div>';
  }}
  document.getElementById('board').innerHTML = html;
  renderMoveList();
}}

function renderMoveList() {{
  let html = '';
  for (let i = 0; i < moves.length; i++) {{
    if (i % 2 === 0) html += `${{Math.floor(i/2)+1}}. `;
    const cls = i === currentPos - 1 ? 'current' : '';
    html += `<span class="${{cls}}" style="cursor:pointer" onclick="goTo(${{i+1}})">${{moves[i]}}</span> `;
  }}
  document.getElementById('moveList').innerHTML = html;
}}

function step(dir) {{
  const next = currentPos + dir;
  if (next >= 0 && next < positions.length) {{ currentPos = next; render(); }}
}}

function goTo(idx) {{
  if (idx >= 0 && idx < positions.length) {{ currentPos = idx; render(); }}
}}

function togglePlay() {{
  if (playInterval) {{
    clearInterval(playInterval);
    playInterval = null;
    document.getElementById('playBtn').textContent = '▶ Play';
  }} else {{
    document.getElementById('playBtn').textContent = '⏸ Pause';
    playInterval = setInterval(() => {{
      if (currentPos >= positions.length - 1) {{ togglePlay(); return; }}
      step(1);
    }}, 800);
  }}
}}

// Populate game selector
pgns.forEach((pgn, i) => {{
  const opt = document.createElement('option');
  const headers = {{}};
  pgn.split('\\n').forEach(l => {{
    const m = l.match(/^\\[(\w+)\\s+"(.+)"\\]/);
    if (m) headers[m[1]] = m[2];
  }});
  opt.value = i;
  opt.textContent = `Partie ${{i+1}}: ${{headers.White||'?'}} vs ${{headers.Black||'?'}} (${{headers.Result||'?'}})`;
  document.getElementById('gameSelect').appendChild(opt);
}});

// Load chess.js from CDN for move replay
const s = document.createElement('script');
s.src = 'https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js';
s.onload = () => loadGame(0);
document.head.appendChild(s);
</script>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n  🌐 Visualisation sauvegardée : {out_path}")


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
    W1, b1, W2, b2, move_tokens = load_model(model_path)
    token_to_move = build_token_to_move(move_tokens)
    print(f"  Tokens    : {len(token_to_move)} coups")
    print(f"  Arch.     : {W1.shape[1]} → {W1.shape[0]} (ReLU) → {W2.shape[0]} (softmax)")
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
    pgns = []

    for i in range(n_games):
        # Alterner les couleurs
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        color_str = "Blancs" if model_color == chess.WHITE else "Noirs"

        score, pgn = play_game(W1, b1, W2, b2, token_to_move, engine, model_color)
        scores.append(score)
        pgns.append(pgn)

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

    # Générer le HTML interactif
    html_path = os.path.splitext(model_path)[0] + "_games.html"
    generate_html(pgns, results, stockfish_elo, html_path)

    # Résumé
    total_score = sum(scores)
    win_rate = total_score / n_games * 100
    print(f"\n{'='*60}")
    print(f"  Résultat final : {results['win']}W - {results['draw']}D - {results['loss']}L")
    print(f"  Score          : {total_score:.1f}/{n_games} ({win_rate:.0f}%)")
    print(f"  Elo Stockfish  : ~{stockfish_elo}")
    print(f"  Visualisation  : {html_path}")
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
