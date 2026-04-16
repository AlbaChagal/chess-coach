"""Tests for move quality classification."""

from __future__ import annotations


from chesscoach.explanation.classifier import classify_move


def test_best_move_zero_loss() -> None:
    q = classify_move(played_cp=35, best_cp=35)
    assert q.label == "best"
    assert q.cp_loss == 0


def test_good_move_small_loss() -> None:
    q = classify_move(played_cp=25, best_cp=35)
    assert q.label == "good"
    assert q.cp_loss == 10


def test_inaccuracy_boundary() -> None:
    q = classify_move(played_cp=0, best_cp=50)
    assert q.label == "inaccuracy"
    assert q.cp_loss == 50


def test_inaccuracy_just_inside() -> None:
    q = classify_move(played_cp=0, best_cp=49)
    assert q.label == "inaccuracy"


def test_mistake_boundary() -> None:
    q = classify_move(played_cp=0, best_cp=150)
    assert q.label == "mistake"
    assert q.cp_loss == 150


def test_blunder_large_loss() -> None:
    q = classify_move(played_cp=-200, best_cp=100)
    assert q.label == "blunder"
    assert q.cp_loss == 300


def test_blunder_just_over_threshold() -> None:
    q = classify_move(played_cp=0, best_cp=151)
    assert q.label == "blunder"
    assert q.emoji == "??"


def test_negative_cp_loss_clamped_to_best() -> None:
    # Played move appears better than engine top (can happen with multipv quirks).
    q = classify_move(played_cp=50, best_cp=30)
    assert q.label == "best"
    assert q.cp_loss == 0


def test_played_forced_mate_is_best() -> None:
    # We play a move that mates in 1.
    q = classify_move(played_cp=None, best_cp=None, played_mate=1, best_mate=1)
    assert q.label == "best"


def test_missing_forced_mate_is_blunder() -> None:
    # Best move mates in 1; we played something else that doesn't mate.
    q = classify_move(played_cp=100, best_cp=None, played_mate=None, best_mate=1)
    assert q.label == "blunder"


def test_emoji_for_blunder() -> None:
    q = classify_move(played_cp=-300, best_cp=0)
    assert q.emoji == "??"


def test_emoji_for_mistake() -> None:
    q = classify_move(played_cp=0, best_cp=100)
    assert q.emoji == "?"


def test_emoji_for_inaccuracy() -> None:
    q = classify_move(played_cp=0, best_cp=30)
    assert q.emoji == "?!"


def test_emoji_for_best_is_empty() -> None:
    q = classify_move(played_cp=35, best_cp=35)
    assert q.emoji == ""
