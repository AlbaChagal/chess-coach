import pytest
from chesscoach.analysis.models import MoveAnalysis


def make_move(**kwargs):
    defaults = dict(
        move_san="e4", move_uci="e2e4", score_cp=35, score_mate=None, depth=20
    )
    defaults.update(kwargs)
    return MoveAnalysis(**defaults)


def test_score_display_positive_centipawns():
    move = make_move(score_cp=35, score_mate=None)
    assert move.score_display() == "+0.35"


def test_score_display_negative_centipawns():
    move = make_move(score_cp=-120, score_mate=None)
    assert move.score_display() == "-1.20"


def test_score_display_zero():
    move = make_move(score_cp=0, score_mate=None)
    assert move.score_display() == "+0.00"


def test_score_display_mate_positive():
    move = make_move(score_cp=None, score_mate=3)
    assert move.score_display() == "#+3"


def test_score_display_mate_negative():
    move = make_move(score_cp=None, score_mate=-2)
    assert move.score_display() == "#-2"


def test_score_display_unknown():
    move = make_move(score_cp=None, score_mate=None)
    assert move.score_display() == "?"


def test_continuation_defaults_empty():
    move = MoveAnalysis(
        move_san="e4", move_uci="e2e4", score_cp=35, score_mate=None, depth=20
    )
    assert move.continuation == []


def test_continuation_stored():
    move = make_move(continuation=["e5", "Nf3", "Nc6"])
    assert move.continuation == ["e5", "Nf3", "Nc6"]
