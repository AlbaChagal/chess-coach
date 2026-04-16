from __future__ import annotations

import sys

from chesscoach.analysis.coach import ChessCoach
from chesscoach.analysis.engine import ChessEngine


def main() -> None:
    if len(sys.argv) >= 2:
        fen = " ".join(sys.argv[1:])
    else:
        fen = input("Enter FEN: ").strip()

    engine = ChessEngine()
    coach = ChessCoach(engine)

    try:
        moves = coach.analyze_position(fen)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(coach.format_suggestions(fen, moves))


if __name__ == "__main__":
    main()
