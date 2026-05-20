import { useRef, useState } from "react";
import {
  Image,
  Pressable,
  StyleSheet,
  Text,
  View
} from "react-native";
import Svg, { Polygon } from "react-native-svg";

import {
  BOARD_SIZE,
  getBoardDisplayBounds,
  getNearestSquare,
  getSquareCenterPoint,
  getSquareFromBoardPoint,
  getSquareFromBoardLocation,
  getSquarePolygon,
  toDisplayPoint
} from "./BoardGeometry.js";
import { CornerPrecisionOverlay } from "./CornerPrecisionOverlay.js";
import { colors, spacing, shadow } from "../theme.js";
import { squareName } from "../utils/chess.js";

function pointsToSvg(points, imageWidth, imageHeight, layoutWidth, layoutHeight) {
  return points
    .map((point) => {
      const display = toDisplayPoint(
        point,
        imageWidth,
        imageHeight,
        layoutWidth,
        layoutHeight
      );
      return `${display.x},${display.y}`;
    })
    .join(" ");
}

function BoardSurface({
  image,
  boardCorners,
  selectedPoint,
  onSelectSquare,
  onOpenPrecision,
  compact = false
}) {
  const [layout, setLayout] = useState(null);
  const suppressNextPress = useRef(false);

  const imageWidth = image?.width || 0;
  const imageHeight = image?.height || 0;
  const selectedSquare =
    boardCorners && selectedPoint
      ? getNearestSquare(boardCorners, selectedPoint)
      : null;
  const boardBounds =
    layout && boardCorners
      ? getBoardDisplayBounds(
          boardCorners,
          imageWidth,
          imageHeight,
          layout.width,
          layout.height
        )
      : null;
  const selectedDisplayPoint =
    layout && boardCorners && selectedSquare
      ? toDisplayPoint(
          getSquareCenterPoint(boardCorners, selectedSquare),
          imageWidth,
          imageHeight,
          layout.width,
          layout.height
        )
      : selectedPoint && layout
        ? toDisplayPoint(
            selectedPoint,
            imageWidth,
            imageHeight,
            layout.width,
            layout.height
          )
        : null;

  const handleRawPress = (event) => {
    if (!image || !layout || !onSelectSquare) {
      return;
    }
    const { locationX, locationY } = event.nativeEvent;
    onSelectSquare({
      x: (locationX / layout.width) * imageWidth,
      y: (locationY / layout.height) * imageHeight
    });
  };

  const handleBoardLongPress = (event) => {
    if (!boardCorners || !boardBounds || !onOpenPrecision) {
      return;
    }
    const { locationX, locationY } = event.nativeEvent;
    const square = getSquareFromBoardLocation(
      boardBounds,
      locationX,
      locationY
    );
    if (square) {
      suppressNextPress.current = true;
      onOpenPrecision(square);
    }
  };

  return (
    <View
      onLayout={(event) => setLayout(event.nativeEvent.layout)}
      style={[
        styles.surface,
        imageWidth && imageHeight ? { aspectRatio: imageWidth / imageHeight } : null,
        compact && styles.compactSurface
      ]}
    >
      {boardCorners && boardBounds && layout ? (
        <>
          <Image
            source={{ uri: image?.uri }}
            style={styles.image}
            resizeMode="contain"
          />
          <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
            <Svg height={layout.height} width={layout.width}>
              {Array.from({ length: BOARD_SIZE }).flatMap((_, row) =>
                Array.from({ length: BOARD_SIZE }).map((__, file) => {
                  const square = squareName(file, row);
                  const polygon = getSquarePolygon(boardCorners, square);
                  const dark = (row + file) % 2 === 1;
                  const selected = square === selectedSquare;
                  return (
                    <Polygon
                      fill={
                        selected ? colors.boardSelect : dark ? "#a56f40" : "#ead7b2"
                      }
                      fillOpacity={selected ? 0.58 : 0.24}
                      key={square}
                      points={pointsToSvg(
                        polygon,
                        imageWidth,
                        imageHeight,
                        layout.width,
                        layout.height
                      )}
                      stroke="rgba(20, 35, 27, 0.16)"
                      strokeWidth={1}
                    />
                  );
                })
              )}
            </Svg>
            {Array.from({ length: BOARD_SIZE }).flatMap((_, row) =>
              Array.from({ length: BOARD_SIZE }).map((__, file) => {
                const square = squareName(file, row);
                const left =
                  boardBounds.left + (boardBounds.width / BOARD_SIZE) * file;
                const top =
                  boardBounds.top + (boardBounds.height / BOARD_SIZE) * row;
                const width = boardBounds.width / BOARD_SIZE;
                const height = boardBounds.height / BOARD_SIZE;
                return (
                  <Pressable
                    accessibilityLabel={`Select ${square}`}
                    delayLongPress={250}
                    key={square}
                    onLongPress={() => {
                      suppressNextPress.current = true;
                      onOpenPrecision?.(square);
                    }}
                    onPress={() => {
                      if (suppressNextPress.current) {
                        suppressNextPress.current = false;
                        return;
                      }
                      onSelectSquare?.(
                        getSquareCenterPoint(boardCorners, square)
                      );
                    }}
                    style={[
                      styles.cell,
                      {
                        left,
                        top,
                        width,
                        height
                      }
                    ]}
                  />
                );
              })
            )}
            {selectedDisplayPoint ? (
              <View
                pointerEvents="none"
                style={[
                  styles.selectionMarker,
                  {
                    left: selectedDisplayPoint.x - 18,
                    top: selectedDisplayPoint.y - 18
                  }
                ]}
              >
                <Text style={styles.selectionGlyph}>♔</Text>
              </View>
            ) : null}
          </View>
        </>
      ) : (
        <Pressable
          onLongPress={handleBoardLongPress}
          onPress={handleRawPress}
          style={StyleSheet.absoluteFill}
        >
          <Image
            source={{ uri: image?.uri }}
            style={styles.image}
            resizeMode="contain"
          />
          {selectedDisplayPoint ? (
            <View
              pointerEvents="none"
              style={[
                styles.selectionMarker,
                {
                  left: selectedDisplayPoint.x - 18,
                  top: selectedDisplayPoint.y - 18
                }
              ]}
            >
              <Text style={styles.selectionGlyph}>♔</Text>
            </View>
          ) : null}
        </Pressable>
      )}
    </View>
  );
}

/**
 * Shows the detected board surface and lets the user select the White king start
 * square, including a long-press precision picker.
 *
 * @param {{image: object, boardCorners: Array<{x: number, y: number}> | null,
 *   selectedPoint: {x: number, y: number} | null,
 *   onSelectPoint: Function}} props - Component props.
 * @returns {JSX.Element} The board selection surface.
 */
export function BoardSelectionSurface({
  image,
  boardCorners,
  selectedPoint,
  onSelectPoint
}) {
  const [precisionSquare, setPrecisionSquare] = useState(null);
  const imageWidth = image?.width || 0;
  const imageHeight = image?.height || 0;
  const precisionFocusPoint =
    precisionSquare && boardCorners
      ? getSquareCenterPoint(boardCorners, precisionSquare)
      : null;

  const handleSelectSquare = (point) => {
    onSelectPoint(point);
    setPrecisionSquare(null);
  };

  const handlePrecisionSelectPoint = (point) => {
    if (!boardCorners) {
      handleSelectSquare(point);
      return;
    }

    const square =
      getSquareFromBoardPoint(boardCorners, point) ||
      getNearestSquare(boardCorners, point);
    handleSelectSquare(getSquareCenterPoint(boardCorners, square));
  };

  return (
    <>
      <BoardSurface
        boardCorners={boardCorners}
        compact
        image={image}
        onOpenPrecision={setPrecisionSquare}
        onSelectSquare={handleSelectSquare}
        selectedPoint={selectedPoint}
      />
      <CornerPrecisionOverlay
        cornerLabel="White king start square"
        focusPoint={precisionFocusPoint}
        imageHeight={imageHeight}
        imageUri={image?.uri}
        imageWidth={imageWidth}
        onClose={() => setPrecisionSquare(null)}
        onSelectPoint={handlePrecisionSelectPoint}
        visible={precisionSquare !== null}
      />
    </>
  );
}

const styles = StyleSheet.create({
  surface: {
    width: "100%",
    borderRadius: 18,
    overflow: "hidden",
    backgroundColor: colors.line,
    borderWidth: 1,
    borderColor: colors.line
  },
  compactSurface: {
    marginBottom: spacing.sm
  },
  image: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: colors.line
  },
  cell: {
    position: "absolute"
  },
  selectionMarker: {
    position: "absolute",
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.green,
    borderWidth: 3,
    borderColor: "#ffffff",
    ...shadow.card
  },
  selectionGlyph: {
    color: "#ffffff",
    fontSize: 18,
    lineHeight: 20
  }
});
