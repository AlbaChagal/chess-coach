export const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

export const PIECE_GLYPHS = {
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚"
};

export const ANALYSIS_ARROW_OPACITY = 0.7;

export const STARTING_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export function squareName(fileIndex, rankIndex) {
  return `${FILES[fileIndex]}${8 - rankIndex}`;
}

export function squareToIndices(square) {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]);
  return [8 - rank, file];
}

export function isPawnPiece(piece) {
  return piece === "P" || piece === "p";
}

export function parseFenPlacement(fenOrPlacement) {
  const placement = fenOrPlacement.split(" ")[0];
  return placement.split("/").map((rankText) => {
    const rank = [];
    for (const char of rankText) {
      if (char >= "1" && char <= "8") {
        for (let index = 0; index < Number(char); index += 1) {
          rank.push(null);
        }
      } else {
        rank.push(char);
      }
    }
    return rank;
  });
}

export function placementFromGrid(grid) {
  return grid
    .map((rank) => {
      let text = "";
      let empty = 0;
      for (const piece of rank) {
        if (piece === null) {
          empty += 1;
          continue;
        }
        if (empty > 0) {
          text += String(empty);
          empty = 0;
        }
        text += piece;
      }
      if (empty > 0) {
        text += String(empty);
      }
      return text;
    })
    .join("/");
}

export function fullFenFromParts({
  placement,
  sideToMove = "w",
  castlingRights = "-",
  enPassant = "-"
}) {
  return `${placement} ${sideToMove} ${castlingRights || "-"} ${enPassant || "-"} 0 1`;
}

export function boardStateFromFen(fen) {
  const parts = fen.split(" ");
  return {
    placement: parseFenPlacement(fen),
    turn: parts[1] || "w",
    castlingRights: parts[2] || "-",
    enPassant: parts[3] || "-"
  };
}

export function fenFromBoardState(boardState) {
  return fullFenFromParts({
    placement: placementFromGrid(boardState.placement),
    sideToMove: boardState.turn,
    castlingRights: boardState.castlingRights,
    enPassant: boardState.enPassant
  });
}

export function nextTurn(turn) {
  return turn === "w" ? "b" : "w";
}

export function applyUciMoveToFen(fen, moveUci) {
  const state = boardStateFromFen(fen);
  const nextState = applyUciMove(state, moveUci);
  return fenFromBoardState(nextState);
}

export function applyUciMove(boardState, moveUci) {
  const from = moveUci.slice(0, 2);
  const to = moveUci.slice(2, 4);
  const promotion = moveUci.slice(4, 5);
  const [fromRow, fromCol] = squareToIndices(from);
  const [toRow, toCol] = squareToIndices(to);
  const grid = boardState.placement.map((rank) => rank.slice());
  let piece = grid[fromRow]?.[fromCol] || null;
  if (!piece) {
    return {
      ...boardState,
      placement: grid,
      turn: nextTurn(boardState.turn),
      enPassant: "-"
    };
  }
  if (promotion) {
    piece = boardState.turn === "w" ? promotion.toUpperCase() : promotion;
  }
  grid[fromRow][fromCol] = null;
  grid[toRow][toCol] = piece;
  return {
    ...boardState,
    placement: grid,
    turn: nextTurn(boardState.turn),
    enPassant: "-"
  };
}

export function pieceAtSquare(fenOrPlacement, square) {
  const grid = parseFenPlacement(fenOrPlacement);
  const [row, col] = squareToIndices(square);
  return grid[row]?.[col] || null;
}

export function setPieceAtSquare(placement, square, piece) {
  const grid = parseFenPlacement(placement);
  const [row, col] = squareToIndices(square);
  if (!grid[row] || col < 0) {
    return placement;
  }
  grid[row][col] = piece;
  return placementFromGrid(grid);
}

export function validatePlacement(placement) {
  const grid = parseFenPlacement(placement);
  if (grid.length !== 8 || grid.some((rank) => rank.length !== 8)) {
    return "The board is malformed. Reset and try again.";
  }
  let whiteKings = 0;
  let blackKings = 0;
  for (const rank of grid) {
    for (const piece of rank) {
      if (piece === "K") {
        whiteKings += 1;
      }
      if (piece === "k") {
        blackKings += 1;
      }
    }
  }
  if (whiteKings !== 1) {
    return "The board must contain exactly one White king.";
  }
  if (blackKings !== 1) {
    return "The board must contain exactly one Black king.";
  }
  return null;
}
