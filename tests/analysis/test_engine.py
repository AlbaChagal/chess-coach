from __future__ import annotations

from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from chesscoach.analysis.engine import ChessEngine
from chesscoach.analysis.models import MoveAnalysis

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_pov_score(cp: int) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)


def _make_pov_mate(mate: int) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Mate(mate), chess.WHITE)


def _make_info(pv_ucis: list[str], score: chess.engine.PovScore, depth: int = 20):
    board = chess.Board()
    pv = [chess.Move.from_uci(u) for u in pv_ucis]
    return {"pv": pv, "score": score, "depth": depth}


# ---------------------------------------------------------------------------
# Unit tests (mocked Stockfish)
# ---------------------------------------------------------------------------

@patch("chess.engine.SimpleEngine.popen_uci")
def test_get_best_moves_returns_three_analyses(mock_popen):
    mock_engine = MagicMock()
    mock_popen.return_value = mock_engine

    infos = [
        _make_info(["e2e4", "e7e5", "g1f3"], _make_pov_score(35)),
        _make_info(["d2d4", "d7d5", "c2c4"], _make_pov_score(30)),
        _make_info(["g1f3", "d7d5", "d2d4"], _make_pov_score(28)),
    ]
    mock_engine.analyse.return_value = infos

    engine = ChessEngine()
    board = chess.Board(STARTING_FEN)
    results = engine.get_best_moves(board, n=3)

    assert len(results) == 3
    assert all(isinstance(r, MoveAnalysis) for r in results)
    mock_engine.analyse.assert_called_once()
    mock_engine.quit.assert_called_once()


@patch("chess.engine.SimpleEngine.popen_uci")
def test_get_best_moves_centipawn_scores(mock_popen):
    mock_engine = MagicMock()
    mock_popen.return_value = mock_engine
    mock_engine.analyse.return_value = [
        _make_info(["e2e4", "e7e5"], _make_pov_score(42), depth=15),
    ]

    engine = ChessEngine()
    results = engine.get_best_moves(chess.Board(), n=1)

    assert results[0].score_cp == 42
    assert results[0].score_mate is None
    assert results[0].depth == 15


@patch("chess.engine.SimpleEngine.popen_uci")
def test_get_best_moves_mate_score(mock_popen):
    mock_engine = MagicMock()
    mock_popen.return_value = mock_engine
    mock_engine.analyse.return_value = [
        _make_info(["e2e4"], _make_pov_mate(3)),
    ]

    engine = ChessEngine()
    results = engine.get_best_moves(chess.Board(), n=1)

    assert results[0].score_mate == 3
    assert results[0].score_cp is None


@patch("chess.engine.SimpleEngine.popen_uci")
def test_get_best_moves_san_notation(mock_popen):
    mock_engine = MagicMock()
    mock_popen.return_value = mock_engine
    mock_engine.analyse.return_value = [
        _make_info(["e2e4", "e7e5", "g1f3"], _make_pov_score(35)),
    ]

    engine = ChessEngine()
    results = engine.get_best_moves(chess.Board(), n=1)

    assert results[0].move_san == "e4"
    assert results[0].move_uci == "e2e4"
    assert results[0].continuation == ["e5", "Nf3"]


@patch("chess.engine.SimpleEngine.popen_uci")
def test_context_manager_reuses_engine(mock_popen):
    mock_engine = MagicMock()
    mock_popen.return_value = mock_engine
    mock_engine.analyse.return_value = [
        _make_info(["e2e4"], _make_pov_score(35)),
    ]

    with ChessEngine() as engine:
        engine.get_best_moves(chess.Board(), n=1)
        engine.get_best_moves(chess.Board(), n=1)

    assert mock_popen.call_count == 1  # opened once
    mock_engine.quit.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests (real Stockfish required)
# ---------------------------------------------------------------------------

def _stockfish_available() -> bool:
    import shutil
    return shutil.which("stockfish") is not None


@pytest.mark.integration
@pytest.mark.skipif(not _stockfish_available(), reason="stockfish binary not found")
def test_integration_starting_position():
    engine = ChessEngine(depth=10)
    board = chess.Board(STARTING_FEN)
    results = engine.get_best_moves(board, n=3)

    assert len(results) == 3
    for r in results:
        assert r.move_san != "?"
        assert r.depth >= 10
        assert r.score_cp is not None or r.score_mate is not None


@pytest.mark.integration
@pytest.mark.skipif(not _stockfish_available(), reason="stockfish binary not found")
def test_integration_context_manager():
    board = chess.Board(STARTING_FEN)
    with ChessEngine(depth=10) as engine:
        results = engine.get_best_moves(board, n=3)
    assert len(results) == 3
