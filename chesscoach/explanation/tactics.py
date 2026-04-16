"""Rule-based tactic detection using python-chess attack maps.

All detectors work by inspecting the board *after* the move is pushed.
Conservative by design — prefers false negatives over false positives.
"""

from __future__ import annotations

import chess

from chesscoach.explanation.models import TacticInfo

# Piece values in centipawns (used to judge whether an attack is meaningful).
_PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20_000,
}

_MIN_FORK_VALUE = 100  # both attacked pieces must be worth at least a pawn


def _piece_name(piece_type: chess.PieceType) -> str:
    return chess.piece_name(piece_type).capitalize()


def _square_name(sq: chess.Square) -> str:
    return chess.square_name(sq)


def detect_tactics(board: chess.Board, move: chess.Move) -> list[TacticInfo]:
    """Detect tactical motifs available after *move* is played on *board*.

    The board is NOT modified; a copy is used internally.

    Args:
        board: Position *before* the move.
        move: The move to evaluate (must be pseudo-legal on *board*).

    Returns:
        List of detected :class:`TacticInfo` instances. Empty if none found.
    """
    after = board.copy()
    after.push(move)

    tactics: list[TacticInfo] = []
    tactics.extend(_detect_check(after))
    tactics.extend(_detect_hanging_pieces(after))
    tactics.extend(_detect_fork(after, move))
    tactics.extend(_detect_pin(after))
    tactics.extend(_detect_discovered_attack(board, after, move))
    return tactics


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def _detect_check(board: chess.Board) -> list[TacticInfo]:
    """Detect if the side to move is in check (i.e., the previous move gave check)."""
    if not board.is_check():
        return []
    # The side that just moved is `not board.turn`.
    return [TacticInfo(name="check", description="The move gives check.")]


def _detect_hanging_pieces(board: chess.Board) -> list[TacticInfo]:
    """Detect opponent pieces that are attacked but insufficiently defended."""
    # After the move, it's the opponent's turn.  We look for their pieces
    # that our side (not board.turn) can capture for free or with gain.
    our_color = not board.turn  # we just moved
    opp_color = board.turn

    results: list[TacticInfo] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != opp_color:
            continue
        if piece.piece_type == chess.KING:
            continue  # king can't be "hanging" in a normal sense

        attackers = board.attackers(our_color, sq)
        if not attackers:
            continue

        defenders = board.attackers(opp_color, sq)
        piece_val = _PIECE_VALUE[piece.piece_type]

        # Simplistic SEE: if undefended, it's hanging.
        if not defenders:
            results.append(
                TacticInfo(
                    name="hanging_piece",
                    description=(
                        f"Opponent's {_piece_name(piece.piece_type)} on "
                        f"{_square_name(sq)} is undefended and can be captured."
                    ),
                )
            )
            continue

        # If the cheapest attacker is worth less than the piece, it's still
        # a favourable capture even after recapture.
        min_attacker_val = min(
            _PIECE_VALUE[board.piece_at(a).piece_type]  # type: ignore[union-attr]
            for a in attackers
            if board.piece_at(a) is not None
        )
        if min_attacker_val < piece_val:
            results.append(
                TacticInfo(
                    name="hanging_piece",
                    description=(
                        f"Opponent's {_piece_name(piece.piece_type)} on "
                        f"{_square_name(sq)} can be captured with material gain."
                    ),
                )
            )

    return results


def _detect_fork(
    board: chess.Board,
    move: chess.Move,
) -> list[TacticInfo]:
    """Detect if the moved piece attacks two or more valuable opponent pieces."""
    our_color = not board.turn  # we just moved
    opp_color = board.turn

    # Which piece moved (it may have promoted, so check destination).
    moved_piece = board.piece_at(move.to_square)
    if moved_piece is None or moved_piece.color != our_color:
        return []

    attacked_valuable: list[tuple[chess.Square, chess.PieceType]] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != opp_color:
            continue
        if _PIECE_VALUE.get(piece.piece_type, 0) < _MIN_FORK_VALUE:
            continue
        if move.to_square in board.attackers(our_color, sq):
            attacked_valuable.append((sq, piece.piece_type))

    if len(attacked_valuable) >= 2:
        targets = " and ".join(
            f"{_piece_name(pt)} on {_square_name(sq)}" for sq, pt in attacked_valuable
        )
        return [
            TacticInfo(
                name="fork",
                description=(
                    f"{_piece_name(moved_piece.piece_type)} on "
                    f"{_square_name(move.to_square)} forks {targets}."
                ),
            )
        ]
    return []


def _detect_pin(board: chess.Board) -> list[TacticInfo]:
    """Detect opponent pieces pinned against their king by our sliding pieces."""
    our_color = not board.turn
    opp_color = board.turn

    opp_king_sq = board.king(opp_color)
    if opp_king_sq is None:
        return []

    results: list[TacticInfo] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != our_color:
            continue
        if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue

        # Get the squares between this piece and the opponent king.
        between = chess.SquareSet(chess.between(sq, opp_king_sq))
        if not between:
            continue
        # Exactly one opponent piece must be between us and the king.
        between_pieces = [
            (s, board.piece_at(s))
            for s in between
            if board.piece_at(s) is not None
        ]
        if len(between_pieces) != 1:
            continue
        pinned_sq, pinned_piece = between_pieces[0]
        if pinned_piece is None or pinned_piece.color != opp_color:
            continue

        # Verify the ray actually connects (bishop on diagonals, rook on ranks/files).
        if not _ray_connects(piece.piece_type, sq, opp_king_sq):
            continue

        results.append(
            TacticInfo(
                name="pin",
                description=(
                    f"Opponent's {_piece_name(pinned_piece.piece_type)} on "
                    f"{_square_name(pinned_sq)} is pinned against their king."
                ),
            )
        )

    return results


def _detect_discovered_attack(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> list[TacticInfo]:
    """Detect if moving a piece revealed an attack from a piece behind it."""
    our_color = not board_after.turn  # we just moved

    results: list[TacticInfo] = []
    for sq in chess.SQUARES:
        attacker_piece = board_after.piece_at(sq)
        if attacker_piece is None or attacker_piece.color != our_color:
            continue
        if attacker_piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        if sq == move.to_square:
            continue  # the moved piece itself — not a discovered attack

        # Squares this slider now attacks that it did NOT attack before.
        attacked_before = board_before.attacks(sq)
        attacked_after = board_after.attacks(sq)
        newly_attacked = attacked_after & ~attacked_before

        for target_sq in newly_attacked:
            target = board_after.piece_at(target_sq)
            if target is None or target.color == our_color:
                continue
            if _PIECE_VALUE.get(target.piece_type, 0) < _MIN_FORK_VALUE:
                continue
            results.append(
                TacticInfo(
                    name="discovered_attack",
                    description=(
                        f"Moving reveals a {_piece_name(attacker_piece.piece_type)} "
                        f"on {_square_name(sq)} attacking opponent's "
                        f"{_piece_name(target.piece_type)} on {_square_name(target_sq)}."
                    ),
                )
            )

    return results


def _ray_connects(piece_type: chess.PieceType, src: chess.Square, dst: chess.Square) -> bool:
    """Return True if *piece_type* can slide from *src* to *dst* on a straight ray."""
    src_file, src_rank = chess.square_file(src), chess.square_rank(src)
    dst_file, dst_rank = chess.square_file(dst), chess.square_rank(dst)

    same_file = src_file == dst_file
    same_rank = src_rank == dst_rank
    same_diag = abs(src_file - dst_file) == abs(src_rank - dst_rank)

    if piece_type == chess.ROOK:
        return same_file or same_rank
    if piece_type == chess.BISHOP:
        return same_diag
    if piece_type == chess.QUEEN:
        return same_file or same_rank or same_diag
    return False
