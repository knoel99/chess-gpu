"""UI web FastAPI du moteur d'échecs matriciel.

- Partie à 0/1/2 joueurs humains (l'autre camp : Stockfish, ou aléatoire en repli)
- Panneau latéral : chaque coup avec sa matrice T (entrées de T − I), l'opérateur
  global G = T_n···T_1 (rang, heatmap, vérification G·b0 = b_n)
- Undo matriciel : application de R = I − E_tt − E_ff + (1/k)·E_ft + correction
  des prises ; retour possible jusqu'à la position initiale
- Bac à sable : placer des pièces sur un plateau vide et visualiser les masques
  matriciels de mouvement/prise et la matrice T de chaque coup candidat

Lancer : venv/bin/python -m src.matrix_chess.webapp  →  http://127.0.0.1:8000
"""

import os
import random
import shutil

import chess
import chess.engine
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .engine import (Game, WHITE, in_check, move_matrix, reverse_matrix, sq,
                     to_uci)

app = FastAPI(title="chess-matrix")

STOCKFISH = shutil.which("stockfish") or os.path.expanduser("~/.local/bin/stockfish")
if not os.path.exists(STOCKFISH):
    STOCKFISH = None
_sf = None


def stockfish():
    global _sf
    if _sf is None and STOCKFISH:
        _sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    return _sf


SQ = [f"{'abcdefgh'[i % 8]}{8 - i // 8}" for i in range(64)]


def formula(move):
    f, t, promo, ec, also = move
    s = f"T = I − E[{SQ[f]},{SQ[f]}] − E[{SQ[t]},{SQ[t]}] + " \
        + (f"{promo}·" if promo else "") + f"E[{SQ[t]},{SQ[f]}]"
    if ec is not None:
        s += f" − E[{SQ[ec]},{SQ[ec]}]"
    if also is not None:
        f2, t2 = also
        s += f" − E[{SQ[f2]},{SQ[f2]}] − E[{SQ[t2]},{SQ[t2]}] + E[{SQ[t2]},{SQ[f2]}]"
    return s


def entries(M):
    """Entrées non nulles de M − I : [[ligne, colonne, valeur], ...]"""
    D = M - np.eye(64)
    return [[int(r), int(c), float(D[r, c])] for r, c in zip(*np.nonzero(D))]


class State:
    def __init__(self, white="human", black="stockfish"):
        self.game = Game()
        self.ref = chess.Board()
        self.players = {"w": white, "b": black}
        self.history = []  # {uci, T(np), caps, prev_castling, prev_ep, record(json)}

    def push_uci(self, uci):
        mv = next(m for m in self.game.legal_moves() if to_uci(m) == uci)
        flat = self.game.board.reshape(64)
        caps = [(mv[1], int(flat[mv[1]]))]
        if mv[3] is not None:
            caps.append((mv[3], int(flat[mv[3]])))
        prev = (self.game.castling, self.game.ep_square)
        san = self.ref.san(chess.Move.from_uci(uci))
        T = self.game.push(mv)
        self.ref.push_uci(uci)
        R = reverse_matrix(T)
        self.history.append({
            "uci": uci, "T": T, "caps": caps, "prev": prev,
            "record": {
                "uci": uci, "san": san, "formula": formula(mv),
                "t_entries": entries(T),
                "r_formula": r_formula(mv),
                "r_entries": entries(R),
                "caps": [[i, v] for i, v in caps if v],
            },
        })

    def undo(self):
        if not self.history:
            return None
        h = self.history.pop()
        R = reverse_matrix(h["T"])
        b = R @ self.game.board.reshape(64).astype(float)
        for idx, val in h["caps"]:
            b[idx] = val
        self.game.board = np.rint(b).astype(np.int64).reshape(8, 8)
        self.game.castling, self.game.ep_square = h["prev"]
        self.game.turn = -self.game.turn
        self.ref.pop()
        return h["record"]

    def global_op(self):
        G = np.eye(64, dtype=np.int64)
        for h in self.history:
            G = h["T"] @ G
        return G


def r_formula(move):
    f, t, promo, ec, also = move
    s = f"R = I − E[{SQ[t]},{SQ[t]}] − E[{SQ[f]},{SQ[f]}] + " \
        + (f"(1/{promo})·" if promo else "") + f"E[{SQ[f]},{SQ[t]}]"
    if also is not None:
        f2, t2 = also
        s += f" − E[{SQ[t2]},{SQ[t2]}] − E[{SQ[f2]},{SQ[f2]}] + E[{SQ[f2]},{SQ[t2]}]"
    if ec is not None:
        s += f"  (+ correction : pièce capturée restaurée en {SQ[ec]})"
    return s


S = State()  # ponytail: une seule partie en mémoire, multi-session si besoin un jour


@app.post("/api/new")
def new(cfg: dict):
    global S
    S = State(cfg.get("white", "human"), cfg.get("black", "stockfish"))
    return state()


@app.get("/api/state")
def state():
    g, b0 = S.game, Game.START
    G = S.global_op()
    ok = bool(np.array_equal((G @ b0.reshape(64)).reshape(8, 8), g.board))
    status = g.status()
    return {
        "board": g.board.tolist(),
        "turn": "w" if g.turn == WHITE else "b",
        "status": status,
        "check": bool(in_check(g.board, g.turn)) if status == "ongoing" else status == "checkmate",
        "players": S.players,
        "legal": [to_uci(m) for m in g.legal_moves()],
        "history": [h["record"] for h in S.history],
        "global": {
            "n": len(S.history),
            "formula": ("G = " + " · ".join(f"T{i}" for i in range(len(S.history), 0, -1)))
                       if S.history else "G = I",
            "rank": int(np.linalg.matrix_rank(G.astype(float))),
            "entries": entries(G),
            "verified": ok,
        },
        "stockfish": STOCKFISH is not None,
    }


@app.post("/api/move")
def move(body: dict):
    uci = body["uci"]
    legal = {to_uci(m) for m in S.game.legal_moves()}
    if uci not in legal and uci + "q" in legal:
        uci += "q"  # ponytail: auto-promotion dame côté humain
    if uci not in legal:
        return {"error": f"coup illégal : {uci}", **state()}
    S.push_uci(uci)
    return state()


@app.post("/api/engine-step")
def engine_step():
    side = "w" if S.game.turn == WHITE else "b"
    kind = S.players[side]
    if kind == "human" or S.game.status() != "ongoing":
        return state()
    legal = sorted(to_uci(m) for m in S.game.legal_moves())
    if kind == "stockfish" and stockfish():
        result = stockfish().play(S.ref, chess.engine.Limit(time=0.1))
        uci = result.move.uci()
    else:
        uci = random.choice(legal)
    S.push_uci(uci)
    return state()


@app.post("/api/undo")
def undo():
    rec = S.undo()
    return {"undone": rec, **state()}


@app.post("/api/rewind")
def rewind():
    undone = []
    while S.history:
        undone.append(S.undo())
    return {"undone": undone, **state()}


@app.post("/api/sandbox")
def sandbox(body: dict):
    """Masques matriciels de la pièce en (r, c) sur un plateau arbitraire."""
    from .engine import piece_moves
    board = np.array(body["board"], dtype=np.int64)
    r, c = body["r"], body["c"]
    if board[r, c] == 0:
        return {"moves": [], "captures": [], "targets": []}
    mask = piece_moves(board, r, c)
    color = 1 if board[r, c] > 0 else -1
    targets = []
    for tr, tc in zip(*np.nonzero(mask)):
        f, t = sq(r, c), sq(int(tr), int(tc))
        mv = (f, t, None, None, None)
        T = move_matrix(f, t)
        targets.append({
            "r": int(tr), "c": int(tc),
            "capture": bool(board[tr, tc] * color < 0),
            "formula": formula(mv), "t_entries": entries(T),
        })
    return {
        "moves": [[t["r"], t["c"]] for t in targets if not t["capture"]],
        "captures": [[t["r"], t["c"]] for t in targets if t["capture"]],
        "targets": targets,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/pieces", response_class=HTMLResponse)
def pieces():
    from . import pieces_doc
    return pieces_doc.PAGE


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>♟ Échecs matriciels</title>
<style>
:root { --sq: 56px; }
body { font-family: system-ui, sans-serif; margin: 16px; background: #14151a; color: #e8e8ea; }
h1 { font-size: 18px; margin: 0 0 10px; }
.tabs button { background:#2a2c34; color:#e8e8ea; border:0; padding:8px 16px; cursor:pointer; border-radius:6px 6px 0 0; margin-right:4px; }
.tabs button.active { background:#3d5afe; }
.layout { display:flex; gap:20px; align-items:flex-start; margin-top:12px; flex-wrap:wrap; }
#board, #sbboard { display:grid; grid-template-columns:repeat(8,var(--sq)); border:2px solid #444; width:fit-content; user-select:none; }
.cell { width:var(--sq); height:var(--sq); display:flex; align-items:center; justify-content:center; font-size:38px; cursor:pointer; position:relative; }
.light { background:#e9d7b7; } .dark { background:#a97d55; }
.cell.sel { outline:3px solid #3d5afe; outline-offset:-3px; }
.cell.move::after { content:""; width:16px; height:16px; border-radius:50%; background:rgba(30,90,220,.55); position:absolute; }
.cell.capt { outline:3px solid #e53935; outline-offset:-3px; }
.wp { color:#fafafa; text-shadow:0 0 2px #000, 0 0 2px #000; } .bp { color:#111; }
.panel { background:#1d1f26; border-radius:8px; padding:12px; max-width:520px; min-width:380px; }
.panel h2 { font-size:14px; margin:0 0 8px; color:#9aa0b4; text-transform:uppercase; letter-spacing:.05em; }
select, .panel button { background:#2a2c34; color:#e8e8ea; border:1px solid #444; border-radius:6px; padding:6px 10px; cursor:pointer; }
#moves { max-height:260px; overflow-y:auto; font-family:ui-monospace,monospace; font-size:12px; }
.mv { padding:4px 6px; border-radius:4px; cursor:pointer; white-space:nowrap; overflow-x:auto; }
.mv:hover, .mv.sel { background:#2a2c34; }
.mv .f { color:#8ab4f8; }
canvas { image-rendering:pixelated; border:1px solid #444; border-radius:4px; }
.small { font-size:12px; color:#9aa0b4; }
.mono { font-family:ui-monospace,monospace; font-size:12px; }
#palette { display:flex; gap:4px; flex-wrap:wrap; margin-bottom:8px; }
#palette span { font-size:30px; cursor:pointer; padding:2px 6px; border-radius:6px; background:#2a2c34; }
#palette span.sel { background:#3d5afe; }
.status { font-weight:600; margin:8px 0; }
.ok { color:#66bb6a; } .bad { color:#ef5350; }
</style></head><body>
<h1>♟ Échecs matriciels — chaque coup est une matrice 64×64
<a href="/pieces" target="_blank" style="font-size:13px;color:#8ab4f8;font-weight:400;margin-left:12px">📄 les matrices de chaque pièce (LaTeX)</a></h1>
<div class="tabs">
  <button id="tab-game" class="active" onclick="showTab('game')">Partie</button>
  <button id="tab-sb" onclick="showTab('sb')">Bac à sable</button>
</div>

<div id="view-game" class="layout">
  <div>
    <div style="margin-bottom:8px">
      Blancs <select id="pw"><option value="human">Humain</option><option value="stockfish">Stockfish</option><option value="random">Aléatoire</option></select>
      Noirs <select id="pb"><option value="human">Humain</option><option value="stockfish" selected>Stockfish</option><option value="random">Aléatoire</option></select>
      <button onclick="newGame()">Nouvelle partie</button>
    </div>
    <div id="board"></div>
    <div class="status" id="status"></div>
    <div>
      <button onclick="undo()">↩ Undo (applique R)</button>
      <button onclick="rewind()">⏮ Retour à b₀ (R·R·…·R)</button>
    </div>
    <div class="small" id="undoinfo" style="max-width:460px"></div>
  </div>
  <div class="panel">
    <h2>Coups et matrices T</h2>
    <div id="moves"></div>
    <h2 style="margin-top:12px">Matrice sélectionnée (T − I sur fond de I)</h2>
    <div class="mono" id="tformula"></div>
    <canvas id="tcanvas" width="256" height="256"></canvas>
  </div>
  <div class="panel">
    <h2>Opérateur global de la partie</h2>
    <div class="mono" id="gformula"></div>
    <div class="small" id="ginfo"></div>
    <canvas id="gcanvas" width="256" height="256"></canvas>
    <div class="small">G · b₀ = plateau courant : <span id="gverif"></span><br>
    Diagonale grise = I (cases inchangées), vert = +k (pièce déposée), rouge = −1 (case vidée/écrasée).</div>
  </div>
</div>

<div id="view-sb" class="layout" style="display:none">
  <div>
    <div id="palette"></div>
    <div id="sbboard"></div>
    <div class="small">Choisis une pièce dans la palette puis clique une case pour la poser
    (re-cliquer la même pièce = effacer). Sans sélection de palette, clique une pièce posée
    pour voir ses <b>mouvements matriciels</b> (points bleus) et <b>prises</b> (cadres rouges).</div>
  </div>
  <div class="panel">
    <h2>Coups candidats de la pièce (pseudo-légaux)</h2>
    <div id="sbtargets" class="mono" style="max-height:300px;overflow-y:auto"></div>
    <h2 style="margin-top:12px">Matrice T du coup survolé/cliqué</h2>
    <div class="mono" id="sbformula"></div>
    <canvas id="sbcanvas" width="256" height="256"></canvas>
  </div>
</div>

<script>
const GLYPH = {0:"",1:"♙",2:"♘",3:"♗",4:"♖",5:"♕",6:"♔","-1":"♟","-2":"♞","-3":"♝","-4":"♜","-5":"♛","-6":"♚"};
let ST = null, sel = null, selMove = -1, engineTimer = null;

function showTab(t) {
  document.getElementById("view-game").style.display = t === "game" ? "flex" : "none";
  document.getElementById("view-sb").style.display = t === "sb" ? "flex" : "none";
  document.getElementById("tab-game").classList.toggle("active", t === "game");
  document.getElementById("tab-sb").classList.toggle("active", t === "sb");
}
async function api(path, body) {
  const r = await fetch("/api/" + path, body !== undefined
    ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)}
    : undefined);
  return r.json();
}
function sqName(r, c) { return "abcdefgh"[c] + (8 - r); }

function drawBoard() {
  const el = document.getElementById("board");
  el.innerHTML = "";
  const legalFrom = new Set(ST.legal.map(u => u.slice(0, 2)));
  const targets = sel ? new Set(ST.legal.filter(u => u.startsWith(sel)).map(u => u.slice(2, 4))) : new Set();
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const d = document.createElement("div");
    const v = ST.board[r][c], name = sqName(r, c);
    d.className = "cell " + ((r + c) % 2 ? "dark" : "light") + (v > 0 ? " wp" : v < 0 ? " bp" : "");
    d.textContent = GLYPH[v];
    if (sel === name) d.classList.add("sel");
    if (targets.has(name)) d.classList.add(ST.board[r][c] !== 0 ? "capt" : "move");
    d.onclick = () => clickCell(r, c, name, legalFrom);
    el.appendChild(d);
  }
  const s = document.getElementById("status");
  const human = ST.players[ST.turn] === "human";
  s.textContent = ST.status === "checkmate" ? "ÉCHEC ET MAT — " + (ST.turn === "w" ? "les noirs" : "les blancs") + " gagnent"
    : ST.status === "stalemate" ? "PAT — nulle"
    : (ST.turn === "w" ? "Trait aux blancs" : "Trait aux noirs") + (human ? "" : " (" + ST.players[ST.turn] + "…)");
}
async function clickCell(r, c, name, legalFrom) {
  if (ST.players[ST.turn] !== "human" || ST.status !== "ongoing") return;
  if (sel && sel !== name) {
    const uci = sel + name;
    if (ST.legal.includes(uci) || ST.legal.includes(uci + "q")) {
      sel = null; ST = await api("move", {uci}); render(); maybeEngine(); return;
    }
  }
  sel = legalFrom.has(name) ? name : null;
  drawBoard();
}
function drawMatrix(canvasId, ents) {
  const cv = document.getElementById(canvasId), ctx = cv.getContext("2d"), s = 4;
  ctx.fillStyle = "#101116"; ctx.fillRect(0, 0, 256, 256);
  ctx.fillStyle = "#3a3d46";
  for (let i = 0; i < 64; i++) ctx.fillRect(i * s, i * s, s, s);   // I
  for (const [r, c, v] of ents) {
    ctx.fillStyle = v > 0 ? "#43a047" : "#e53935";
    ctx.fillRect(c * s, r * s, s, s);
    if (r === c && v === -1) ctx.fillStyle = "#e53935", ctx.fillRect(c*s, r*s, s, s);
  }
}
function drawMoves() {
  const el = document.getElementById("moves");
  el.innerHTML = "";
  ST.history.forEach((h, i) => {
    const d = document.createElement("div");
    d.className = "mv" + (i === selMove ? " sel" : "");
    d.innerHTML = `<b>T${i + 1}</b> ${h.san} (${h.uci}) — <span class="f">${h.formula}</span>`
      + (h.caps.length ? ` <span style="color:#ef5350">prise!</span>` : "");
    d.onclick = () => { selMove = i; drawMoves(); showT(h); };
    el.appendChild(d);
  });
  if (ST.history.length && (selMove < 0 || selMove >= ST.history.length)) {
    selMove = ST.history.length - 1;
    el.children[selMove].classList.add("sel");
    showT(ST.history[selMove]);
  }
  if (!ST.history.length) { selMove = -1; document.getElementById("tformula").textContent = "";
    drawMatrix("tcanvas", []); }
  el.scrollTop = el.scrollHeight;
}
function showT(h) {
  document.getElementById("tformula").innerHTML = h.formula + "<br>" + h.r_formula;
  drawMatrix("tcanvas", h.t_entries);
}
function render() {
  drawBoard(); drawMoves();
  document.getElementById("gformula").textContent = ST.global.formula;
  document.getElementById("ginfo").textContent =
    `rang(G) = ${ST.global.rank}/64` + (ST.global.rank < 64
      ? " → singulière (les prises/écrasements détruisent l'information : l'inverse exact n'existe pas, d'où l'undo affine R + correction)" : " → inversible (aucune prise)");
  document.getElementById("gverif").innerHTML = ST.global.verified ? '<span class="ok">✓ vérifié</span>' : '<span class="bad">✗</span>';
  drawMatrix("gcanvas", ST.global.entries);
}
function maybeEngine() {
  clearTimeout(engineTimer);
  if (ST.status === "ongoing" && ST.players[ST.turn] !== "human")
    engineTimer = setTimeout(async () => { ST = await api("engine-step", {}); render(); maybeEngine(); }, 350);
}
async function newGame() {
  sel = null; selMove = -1;
  ST = await api("new", {white: document.getElementById("pw").value, black: document.getElementById("pb").value});
  document.getElementById("undoinfo").textContent = "";
  render(); maybeEngine();
}
async function undo() {
  clearTimeout(engineTimer);
  const res = await api("undo", {});
  ST = res; render();
  if (res.undone) document.getElementById("undoinfo").innerHTML =
    "Annulé " + res.undone.san + " : " + res.undone.r_formula +
    (res.undone.caps.length ? "<br>correction : b[" + res.undone.caps.map(x => sqName(Math.floor(x[0]/8), x[0]%8) + "]=" + x[1]) : "");
}
async function rewind() {
  clearTimeout(engineTimer);
  const res = await api("rewind", {});
  ST = res; render();
  document.getElementById("undoinfo").textContent =
    res.undone.length + " matrices R appliquées (avec corrections de prises) — retour exact à b₀.";
}

/* ---------- bac à sable ---------- */
let SB = Array.from({length: 8}, () => Array(8).fill(0)), pal = null, sbSel = null;
const PIECES = [6,5,4,3,2,1,-1,-2,-3,-4,-5,-6];
function drawPalette() {
  const el = document.getElementById("palette");
  el.innerHTML = "";
  PIECES.forEach(v => {
    const s = document.createElement("span");
    s.textContent = GLYPH[v];
    if (pal === v) s.classList.add("sel");
    s.onclick = () => { pal = pal === v ? null : v; drawPalette(); };
    el.appendChild(s);
  });
  const e = document.createElement("span");
  e.textContent = "🗑"; e.title = "vider le plateau";
  e.onclick = () => { SB = Array.from({length:8},()=>Array(8).fill(0)); sbSel = null; drawSB([], []); };
  el.appendChild(e);
}
function drawSB(moves, captures) {
  const el = document.getElementById("sbboard");
  el.innerHTML = "";
  const mset = new Set(moves.map(m => m[0] * 8 + m[1])), cset = new Set(captures.map(m => m[0] * 8 + m[1]));
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const d = document.createElement("div");
    const v = SB[r][c];
    d.className = "cell " + ((r + c) % 2 ? "dark" : "light") + (v > 0 ? " wp" : v < 0 ? " bp" : "");
    d.textContent = GLYPH[v];
    if (sbSel && sbSel[0] === r && sbSel[1] === c) d.classList.add("sel");
    if (mset.has(r * 8 + c)) d.classList.add("move");
    if (cset.has(r * 8 + c)) d.classList.add("capt");
    d.onclick = () => sbClick(r, c);
    el.appendChild(d);
  }
}
async function sbClick(r, c) {
  if (pal !== null) { SB[r][c] = SB[r][c] === pal ? 0 : pal; drawSB([], []); return; }
  if (!SB[r][c]) return;
  sbSel = [r, c];
  const res = await api("sandbox", {board: SB, r, c});
  drawSB(res.moves, res.captures);
  const el = document.getElementById("sbtargets");
  el.innerHTML = "";
  res.targets.forEach(t => {
    const d = document.createElement("div");
    d.className = "mv";
    d.innerHTML = (t.capture ? "⚔ " : "→ ") + sqName(t.r, t.c) + " — <span class='f'>" + t.formula + "</span>";
    d.onclick = () => { document.getElementById("sbformula").textContent = t.formula; drawMatrix("sbcanvas", t.t_entries); };
    el.appendChild(d);
  });
  document.getElementById("sbformula").textContent = "";
  drawMatrix("sbcanvas", []);
}
drawPalette(); drawSB([], []);
(async () => { ST = await api("state"); render(); maybeEngine(); })();
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
