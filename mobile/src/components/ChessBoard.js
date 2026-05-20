import { StyleSheet, Text, Pressable, View } from "react-native";
import Svg, { Circle, Polygon } from "react-native-svg";

import { ChessPieceIcon } from "./ChessPieceIcon";
import { colors } from "../theme";
import {
  ANALYSIS_ARROW_OPACITY,
  FILES,
  parseFenPlacement,
  squareName,
  squareToIndices
} from "../utils/chess";

const BOARD_VIEWBOX_SIZE = 320;
const ARROW_HEAD_LENGTH = 14;
const ARROW_HEAD_HALF_WIDTH = 8;
const ARROW_SHAFT_HALF_WIDTH = 4.5;
const ARROW_START_OFFSET = 14;

function displaySquares(orientation) {
  const ranks =
    orientation === "white"
      ? [0, 1, 2, 3, 4, 5, 6, 7]
      : [7, 6, 5, 4, 3, 2, 1, 0];
  const files =
    orientation === "white"
      ? [0, 1, 2, 3, 4, 5, 6, 7]
      : [7, 6, 5, 4, 3, 2, 1, 0];
  return ranks.flatMap((rank) =>
    files.map((file) => ({
      rank,
      file,
      square: squareName(file, rank)
    }))
  );
}

function centerForSquare(square, boardSize, orientation) {
  const [rank, file] = squareToIndices(square);
  const displayRank = orientation === "white" ? rank : 7 - rank;
  const displayFile = orientation === "white" ? file : 7 - file;
  const cell = boardSize / 8;
  return {
    x: displayFile * cell + cell / 2,
    y: displayRank * cell + cell / 2
  };
}

function arrowHeadGeometry(start, end) {
  const deltaX = end.x - start.x;
  const deltaY = end.y - start.y;
  const length = Math.hypot(deltaX, deltaY) || 1;
  const unitX = deltaX / length;
  const unitY = deltaY / length;
  const normalX = -unitY;
  const normalY = unitX;
  const tip = end;
  const back = {
    x: tip.x - unitX * ARROW_HEAD_LENGTH,
    y: tip.y - unitY * ARROW_HEAD_LENGTH
  };
  const left = {
    x: back.x + normalX * ARROW_HEAD_HALF_WIDTH,
    y: back.y + normalY * ARROW_HEAD_HALF_WIDTH
  };
  const right = {
    x: back.x - normalX * ARROW_HEAD_HALF_WIDTH,
    y: back.y - normalY * ARROW_HEAD_HALF_WIDTH
  };
  const shaftStart = {
    x: start.x + unitX * ARROW_START_OFFSET,
    y: start.y + unitY * ARROW_START_OFFSET
  };
  const shaftLeftStart = {
    x: shaftStart.x + normalX * ARROW_SHAFT_HALF_WIDTH,
    y: shaftStart.y + normalY * ARROW_SHAFT_HALF_WIDTH
  };
  const shaftRightStart = {
    x: shaftStart.x - normalX * ARROW_SHAFT_HALF_WIDTH,
    y: shaftStart.y - normalY * ARROW_SHAFT_HALF_WIDTH
  };
  const shaftLeftEnd = {
    x: back.x + normalX * ARROW_SHAFT_HALF_WIDTH,
    y: back.y + normalY * ARROW_SHAFT_HALF_WIDTH
  };
  const shaftRightEnd = {
    x: back.x - normalX * ARROW_SHAFT_HALF_WIDTH,
    y: back.y - normalY * ARROW_SHAFT_HALF_WIDTH
  };
  return {
    base: back,
    shaftStart,
    headPoints: `${tip.x},${tip.y} ${left.x},${left.y} ${right.x},${right.y}`,
    shaftPoints: [
      shaftLeftStart,
      shaftLeftEnd,
      shaftRightEnd,
      shaftRightStart
    ]
      .map((point) => `${point.x},${point.y}`)
      .join(" ")
  };
}

function targetIndicatorStyle(isCapture) {
  return [
    styles.targetDot,
    isCapture ? styles.captureDot : styles.legalDot
  ];
}

function PieceIcon({ piece }) {
  return <ChessPieceIcon piece={piece} size={34} />;
}

export function ChessBoard({
  fen,
  orientation = "white",
  selectedSquare = null,
  legalTargets = [],
  arrowMove = null,
  editableSquares = [],
  onSquarePress,
  showCoordinates = true,
  style
}) {
  const grid = parseFenPlacement(fen);
  const squares = displaySquares(orientation);
  const legalTargetSet = new Set(legalTargets);
  const editableSet = new Set(editableSquares);

  return (
    <View style={[styles.board, style]} onLayout={() => {}}>
      <View style={styles.grid}>
        {squares.map(({ rank, file, square }) => {
          const dark = (rank + file) % 2 === 1;
          const piece = grid[rank]?.[file] || null;
          const selected = selectedSquare === square;
          const legal = legalTargetSet.has(square);
          const editable = editableSet.has(square);
          return (
            <Pressable
              accessibilityLabel={`Square ${square}`}
              key={square}
              onPress={() => onSquarePress?.(square)}
              style={[
                styles.square,
                dark ? styles.dark : styles.light,
                selected && styles.selected,
                editable && styles.editable
              ]}
            >
              {showCoordinates &&
              file === (orientation === "white" ? 0 : 7) ? (
                <Text style={styles.rankLabel}>{8 - rank}</Text>
              ) : null}
              {showCoordinates &&
              rank === (orientation === "white" ? 7 : 0) ? (
                <Text style={styles.fileLabel}>{FILES[file]}</Text>
              ) : null}
              {piece ? <PieceIcon piece={piece} /> : null}
              {legal ? (
                <View
                  pointerEvents="none"
                  style={targetIndicatorStyle(Boolean(piece))}
                />
              ) : null}
            </Pressable>
          );
        })}
      </View>
      {arrowMove ? (
        <BoardArrow moveUci={arrowMove} orientation={orientation} />
      ) : null}
    </View>
  );
}

function BoardArrow({ moveUci, orientation }) {
  const from = moveUci.slice(0, 2);
  const to = moveUci.slice(2, 4);
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <Svg
        height="100%"
        viewBox={`0 0 ${BOARD_VIEWBOX_SIZE} ${BOARD_VIEWBOX_SIZE}`}
        width="100%"
      >
        <ArrowGeometry from={from} to={to} orientation={orientation} />
      </Svg>
    </View>
  );
}

function ArrowGeometry({ from, to, orientation }) {
  const start = centerForSquare(from, BOARD_VIEWBOX_SIZE, orientation);
  const end = centerForSquare(to, BOARD_VIEWBOX_SIZE, orientation);
  const head = arrowHeadGeometry(start, end);
  return (
    <>
      <Polygon
        fill={colors.gold}
        fillOpacity={ANALYSIS_ARROW_OPACITY}
        points={head.shaftPoints}
      />
      <Circle
        cx={head.shaftStart.x}
        cy={head.shaftStart.y}
        fill={colors.gold}
        fillOpacity={ANALYSIS_ARROW_OPACITY}
        r={ARROW_SHAFT_HALF_WIDTH}
      />
      <Polygon
        fill={colors.gold}
        fillOpacity={ANALYSIS_ARROW_OPACITY}
        points={head.headPoints}
      />
    </>
  );
}

const styles = StyleSheet.create({
  board: {
    aspectRatio: 1,
    width: "100%",
    maxWidth: 430,
    alignSelf: "center",
    borderRadius: 24,
    overflow: "hidden",
    borderWidth: 4,
    borderColor: colors.ink,
    backgroundColor: colors.ink
  },
  grid: {
    flex: 1,
    flexDirection: "row",
    flexWrap: "wrap"
  },
  square: {
    width: "12.5%",
    height: "12.5%",
    alignItems: "center",
    justifyContent: "center"
  },
  light: {
    backgroundColor: "#ead7b2"
  },
  dark: {
    backgroundColor: "#9c6a3d"
  },
  selected: {
    borderWidth: 3,
    borderColor: colors.green
  },
  editable: {
    shadowColor: colors.gold,
    shadowOpacity: 1,
    shadowRadius: 6
  },
  targetDot: {
    position: "absolute",
    top: "50%",
    left: "50%"
  },
  legalDot: {
    position: "absolute",
    width: 18,
    height: 18,
    top: "50%",
    left: "50%",
    marginLeft: -9,
    marginTop: -9,
    borderRadius: 9,
    backgroundColor: "rgba(70, 70, 70, 0.42)"
  },
  captureDot: {
    position: "absolute",
    width: 36,
    height: 36,
    top: "50%",
    left: "50%",
    marginLeft: -18,
    marginTop: -18,
    borderRadius: 18,
    borderWidth: 5,
    borderColor: "rgba(70, 70, 70, 0.38)"
  },
  rankLabel: {
    position: "absolute",
    top: 2,
    left: 3,
    fontSize: 9,
    color: "rgba(30,30,30,0.62)",
    fontWeight: "900"
  },
  fileLabel: {
    position: "absolute",
    right: 3,
    bottom: 1,
    fontSize: 9,
    color: "rgba(30,30,30,0.62)",
    fontWeight: "900"
  }
});
