from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import chess
import pytest

from chesscoach.analysis.coach import ChessCoach
from chesscoach.analysis.models import MoveAnalysis

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

SAMPLE_MOVES = [
    MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5", "Nf3", "Nc6"]),
    MoveAnalysis("d4", "d2d4", 30, None, 20, ["d5", "c4", "e6"]),
    MoveAnalysis("Nf3", "g1f3", 28, None, 20, ["d5", "d4", "Nf6"]),
]


def make_coach(moves=None):
    engine = MagicMock()
    engine.get_best_moves.return_value = moves or SAMPLE_MOVES
    return ChessCoach(engine)


# ---------------------------------------------------------------------------
# parse_fen
# ---------------------------------------------------------------------------

def test_parse_fen_valid_starting_position():
    coach = make_coach()
    board = coach.parse_fen(STARTING_FEN)
    assert isinstance(board, chess.Board)


def test_parse_fen_valid_midgame():
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    coach = make_coach()
    board = coach.parse_fen(fen)
    assert isinstance(board, chess.Board)


def test_parse_fen_invalid_raises():
    coach = make_coach()
    with pytest.raises(ValueError, match="Invalid FEN"):
        coach.parse_fen("not_a_fen")


def test_parse_fen_empty_raises():
    coach = make_coach()
    with pytest.raises(ValueError):
        coach.parse_fen("")


# ---------------------------------------------------------------------------
# analyze_position
# ---------------------------------------------------------------------------

def test_analyze_position_calls_engine():
    coach = make_coach()
    results = coach.analyze_position(STARTING_FEN, n=3)
    assert results == SAMPLE_MOVES
    coach._engine.get_best_moves.assert_called_once()


def test_analyze_position_invalid_fen_raises():
    coach = make_coach()
    with pytest.raises(ValueError):
        coach.analyze_position("garbage fen")


# ---------------------------------------------------------------------------
# format_suggestions
# ---------------------------------------------------------------------------

def test_format_suggestions_contains_moves():
    coach = make_coach()
    output = coach.format_suggestions(STARTING_FEN, SAMPLE_MOVES)
    assert "e4" in output
    assert "d4" in output
    assert "Nf3" in output


def test_format_suggestions_contains_scores():
    coach = make_coach()
    output = coach.format_suggestions(STARTING_FEN, SAMPLE_MOVES)
    assert "+0.35" in output
    assert "+0.30" in output
    assert "+0.28" in output


def test_format_suggestions_contains_continuations():
    coach = make_coach()
    output = coach.format_suggestions(STARTING_FEN, SAMPLE_MOVES)
    assert "e5" in output
    assert "d5" in output


def test_format_suggestions_contains_fen():
    coach = make_coach()
    output = coach.format_suggestions(STARTING_FEN, SAMPLE_MOVES)
    assert STARTING_FEN in output


def test_format_suggestions_mate_score():
    moves = [MoveAnalysis("Qh5#", "d1h5", None, 1, 20, [])]
    coach = make_coach(moves)
    output = coach.format_suggestions(STARTING_FEN, moves)
    assert "#" in output


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_main_prints_output(capsys):
    with patch("chesscoach.analysis.coach.ChessCoach.analyze_position", return_value=SAMPLE_MOVES):
        with patch("sys.argv", ["chess-coach", STARTING_FEN]):
            from chesscoach.cli import main
            main()

    captured = capsys.readouterr()
    assert "e4" in captured.out
    assert "d4" in captured.out


def test_cli_main_invalid_fen_exits(capsys):
    with patch("sys.argv", ["chess-coach", "invalid_fen"]):
        from chesscoach.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1
