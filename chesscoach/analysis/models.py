from dataclasses import dataclass, field


@dataclass
class MoveAnalysis:
    move_san: str
    move_uci: str
    score_cp: int | None
    score_mate: int | None
    depth: int
    continuation: list[str] = field(default_factory=list)

    def score_display(self) -> str:
        if self.score_mate is not None:
            sign = "+" if self.score_mate > 0 else ""
            return f"#{sign}{self.score_mate}" if self.score_mate > 0 else f"#-{abs(self.score_mate)}"
        if self.score_cp is not None:
            pawns = self.score_cp / 100
            return f"{pawns:+.2f}"
        return "?"
