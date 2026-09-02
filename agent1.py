"""The submission entrypoint. The platform imports this file and calls get_move."""

import time

import chess
import chess.polyglot

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

INF = 10 ** 9
EXACT, LOWER, UPPER = 0, 1, 2

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables, indexed a1..h8 (python-chess square order), White's perspective.
# Black looks itself up with chess.square_mirror. Values are centipawns.

PAWN_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10, -20, -20,  10,  10,   5,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    0,   0,   0,  20,  20,   0,   0,   0,
    5,   5,  10,  25,  25,  10,   5,   5,
    10,  10,  20,  30,  30,  20,  10,  10,
    50,  50,  50,  50,  50,  50,  50,  50,
    0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
    0,   0,   0,   5,   5,   0,   0,   0,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    5,  10,  10,  10,  10,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0,
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
    0,   0,   5,   5,   5,   5,   0,  -5,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

# Middlegame table: rewards a castled, sheltered king. No separate endgame table yet
# (see docs/IDEAS.md-style follow-ups: king should centralise once material thins out).
KING_PST = [
    20,  30,  10,   0,   0,  10,  30,  20,
    20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}

# Transposition table, keyed by zobrist hash -> (depth, score, flag, best_move).
# Module-level and never reset mid-game, so it stays warm across your own moves;
# capped so a long game can't grow it without bound.
_TT: dict[int, tuple[int, int, int, "chess.Move | None"]] = {}
_TT_MAX_ENTRIES = 2_000_000


def _evaluate(board: chess.Board) -> int:
    """Static material + piece-square evaluation, in centipawns, from the mover's side.

    Callers are expected to have already handled checkmate/stalemate — this assumes
    the position is "alive".
    """
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        index = square if piece.color == chess.WHITE else chess.square_mirror(square)
        value += PST[piece.piece_type][index]
        score += value if piece.color == chess.WHITE else -value
    return score if board.turn == chess.WHITE else -score


def _order_moves(board: chess.Board, moves, tt_move=None):
    """TT move first, then captures by MVV-LVA, then everything else."""

    def priority(move: chess.Move) -> int:
        if tt_move is not None and move == tt_move:
            return 1_000_000
        if board.is_capture(move):
            attacker = board.piece_type_at(move.from_square)
            captured = board.piece_type_at(move.to_square) or chess.PAWN  # en passant
            return PIECE_VALUES[captured] * 10 - PIECE_VALUES[attacker]
        return 0

    return sorted(moves, key=priority, reverse=True)


def _quiescence(board: chess.Board, alpha: int, beta: int, deadline: float) -> int:
    """Search captures only, until the position is quiet, to avoid evaluating
    mid-exchange and misjudging a position as safe or lost."""
    if time.monotonic() > deadline:
        raise TimeoutError

    moves = list(board.legal_moves)
    if not moves:
        return -INF if board.is_check() else 0

    stand_pat = _evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = _order_moves(board, [m for m in moves if board.is_capture(m)])
    for move in captures:
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, deadline)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _search(board: chess.Board, depth: int, alpha: int, beta: int, deadline: float) -> int:
    """Fail-soft negamax alpha-beta with a transposition table."""
    if time.monotonic() > deadline:
        raise TimeoutError

    key = chess.polyglot.zobrist_hash(board)
    alpha_orig = alpha
    tt_move = None
    entry = _TT.get(key)
    if entry is not None:
        tt_depth, tt_score, tt_flag, tt_move = entry
        if tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_score
            if tt_flag == LOWER and tt_score > alpha:
                alpha = tt_score
            elif tt_flag == UPPER and tt_score < beta:
                beta = tt_score
            if alpha >= beta:
                return tt_score

    moves = list(board.legal_moves)
    if not moves:
        return -INF if board.is_check() else 0  # checkmate or stalemate

    if depth == 0:
        return _quiescence(board, alpha, beta, deadline)

    best_score = -INF
    best_move = None
    for move in _order_moves(board, moves, tt_move):
        board.push(move)
        score = -_search(board, depth - 1, -beta, -alpha, deadline)
        board.pop()
        if score > best_score:
            best_score, best_move = score, move
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    flag = EXACT
    if best_score <= alpha_orig:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    _TT[key] = (depth, best_score, flag, best_move)
    if len(_TT) > _TT_MAX_ENTRIES:
        _TT.clear()

    return best_score


def _time_budget_s(board: chess.Board, time_left_ms: int) -> float:
    """Conservative per-move budget: assume ~40 moves left, never spend more than a
    third of what's on the clock on one move, and always keep a safety floor so we
    can never flag."""
    moves_left = max(10, 40 - board.fullmove_number)
    budget_ms = time_left_ms / moves_left
    budget_ms = min(budget_ms, time_left_ms * 0.3)
    budget_ms = max(budget_ms, 50.0)
    return budget_ms / 1000.0


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds

    returns "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"  # shouldn't be asked to move here, but never crash

    best_move = legal_moves[0]
    deadline = time.monotonic() + _time_budget_s(board, time_left_ms)

    # Iterative deepening: always keep the last fully-completed depth's best move,
    # so a TimeoutError mid-search never leaves us without an answer.
    depth = 1
    try:
        while depth <= 64:
            best_score = -INF
            current_best = None
            for move in _order_moves(board, legal_moves):
                board.push(move)
                score = -_search(board, depth - 1, -INF, INF, deadline)
                board.pop()
                if score > best_score:
                    best_score, current_best = score, move
            best_move = current_best
            depth += 1
    except TimeoutError:
        pass

    return best_move.uci()

