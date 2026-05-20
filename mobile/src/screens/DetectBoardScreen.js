import { useRef, useState } from "react";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";

import { AppButton } from "../components/AppButton";
import { Card } from "../components/Card";
import { CornerPrecisionOverlay } from "../components/CornerPrecisionOverlay";
import { Screen } from "../components/Screen";
import {
  STAGE_ROUTES,
  STAGES,
  StageProgress
} from "../components/StageProgress";
import { useAppState } from "../state/AppContext";
import { colors, spacing, typography } from "../theme";
import {
  getContainedImageBounds,
  getSourcePointFromContainedPoint,
  nextClockwiseCornerIndex
} from "../utils/cornerPrecision";

const CORNER_CONTROLS = [
  { label: "Top left", cornerIndex: 0 },
  { label: "Top right", cornerIndex: 1 },
  { label: "Bottom left", cornerIndex: 3 },
  { label: "Bottom right", cornerIndex: 2 }
];

/**
 * Lets the user correct the detected board corners on the uploaded photo.
 */
export function DetectBoardScreen({ navigation }) {
  const {
    boardDetection,
    image,
    position,
    session,
    setBoardDetection,
    whiteKingStartClick
  } = useAppState();
  const [selectedCorner, setSelectedCorner] = useState(0);
  const [layout, setLayout] = useState(null);
  const [precisionFocusPoint, setPrecisionFocusPoint] = useState(null);
  const ignoreNextPressRef = useRef(false);
  const currentStage = "Detect";
  const flowState = {
    image,
    boardDetection,
    whiteKingStartClick,
    position,
    session
  };

  const corners = boardDetection?.board_corners || [];
  const imageWidth = image?.width || boardDetection?.image_width || 0;
  const imageHeight = image?.height || boardDetection?.image_height || 0;
  const imageBounds = getContainedImageBounds(layout, imageWidth, imageHeight);

  const updateSelectedCorner = (sourcePoint, { advance = true } = {}) => {
    if (!boardDetection) {
      return;
    }

    const nextCorners = corners.map((corner, index) =>
      index === selectedCorner
        ? { x: sourcePoint.x, y: sourcePoint.y }
        : corner
    );
    setBoardDetection({ ...boardDetection, board_corners: nextCorners });
    if (advance) {
      setSelectedCorner(nextClockwiseCornerIndex(selectedCorner));
    }
  };

  const closePrecisionOverlay = () => {
    ignoreNextPressRef.current = false;
    setPrecisionFocusPoint(null);
  };

  const handleImagePress = (event) => {
    if (ignoreNextPressRef.current) {
      ignoreNextPressRef.current = false;
      return;
    }

    const { locationX, locationY } = event.nativeEvent;
    const sourcePoint = getSourcePointFromContainedPoint(
      locationX,
      locationY,
      imageBounds,
      imageWidth,
      imageHeight
    );

    if (!sourcePoint) {
      return;
    }

    updateSelectedCorner(sourcePoint);
  };

  const handleImageLongPress = (event) => {
    const { locationX, locationY } = event.nativeEvent;
    const sourcePoint = getSourcePointFromContainedPoint(
      locationX,
      locationY,
      imageBounds,
      imageWidth,
      imageHeight
    );

    if (!sourcePoint) {
      return;
    }

    ignoreNextPressRef.current = true;
    setPrecisionFocusPoint(sourcePoint);
  };

  const handleStagePress = (stage) => {
    const currentIndex = STAGES.indexOf(currentStage);
    const targetIndex = STAGES.indexOf(stage);
    if (targetIndex < 0 || targetIndex === currentIndex) {
      return;
    }
    if (targetIndex < currentIndex) {
      navigation.pop(currentIndex - targetIndex);
      return;
    }
    navigation.navigate(STAGE_ROUTES[stage]);
  };

  if (!image || !boardDetection) {
    return (
      <Screen title="Detect Board">
        <Text style={typography.body}>Load an image first.</Text>
        <AppButton
          title="Back to Capture"
          onPress={() => navigation.navigate("Capture")}
        />
      </Screen>
    );
  }

  return (
    <Screen
      title="Adjust Board"
      subtitle="Tap the photo to place corners clockwise. Use the labels to revisit a corner."
    >
      <StageProgress
        current={currentStage}
        flowState={flowState}
        onStagePress={handleStagePress}
      />
      <AppButton
        title="Back to Capture"
        variant="secondary"
        onPress={() => navigation.goBack()}
      />
      <Card>
        <View style={styles.boardContent}>
          <Pressable
            onLayout={(event) => setLayout(event.nativeEvent.layout)}
            onPress={handleImagePress}
            onLongPress={handleImageLongPress}
            delayLongPress={250}
            style={[
              styles.imageFrame,
              imageWidth && imageHeight
                ? { aspectRatio: imageWidth / imageHeight }
                : null
            ]}
          >
            <Image
              source={{ uri: image.uri }}
              style={styles.image}
              resizeMode="contain"
            />
            {layout && imageBounds
              ? corners.map((corner, index) => (
                  <View
                    key={index}
                    style={[
                      styles.corner,
                      {
                        left:
                          imageBounds.offsetX +
                          (corner.x / imageWidth) * imageBounds.displayWidth -
                          10,
                        top:
                          imageBounds.offsetY +
                          (corner.y / imageHeight) * imageBounds.displayHeight -
                          10
                      },
                      index === selectedCorner && styles.activeCorner
                    ]}
                  />
                ))
              : null}
          </Pressable>
        </View>
        <View style={styles.cornerPicker}>
          {CORNER_CONTROLS.map(({ label, cornerIndex }) => (
            <Pressable
              key={label}
              onPress={() => setSelectedCorner(cornerIndex)}
              style={[
                styles.cornerButton,
                cornerIndex === selectedCorner && styles.activeCornerButton
              ]}
            >
              <Text
                style={[
                  styles.cornerText,
                  cornerIndex === selectedCorner && styles.activeCornerText
                ]}
              >
                {label}
              </Text>
            </Pressable>
          ))}
        </View>
        <CornerPrecisionOverlay
          cornerLabel={CORNER_CONTROLS[selectedCorner].label}
          focusPoint={precisionFocusPoint}
          imageHeight={imageHeight}
          imageUri={image.uri}
          imageWidth={imageWidth}
          onClose={closePrecisionOverlay}
          onSelectPoint={updateSelectedCorner}
          visible={precisionFocusPoint !== null}
        />
        {boardDetection.confidence < 0.5 ? (
          <Text style={styles.warning}>
            Detection confidence is low. Correct the corners before continuing.
          </Text>
        ) : null}
      </Card>
      <AppButton
        title="Continue"
        onPress={() => navigation.navigate("SetupPosition")}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  imageFrame: {
    width: "100%",
    borderRadius: 18,
    backgroundColor: colors.line
  },
  image: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 18,
    backgroundColor: colors.line
  },
  corner: {
    position: "absolute",
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 3,
    borderColor: "#ffffff",
    backgroundColor: colors.boardSelect
  },
  activeCorner: {
    backgroundColor: colors.green
  },
  boardContent: {
    gap: spacing.md
  },
  cornerPicker: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.xs
  },
  cornerButton: {
    width: "48%",
    minHeight: 52,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.sm,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 14,
    backgroundColor: colors.greenSoft,
    borderWidth: 1,
    borderColor: "rgba(23, 107, 51, 0.12)"
  },
  activeCornerButton: {
    backgroundColor: colors.green,
    borderColor: colors.greenDark
  },
  cornerText: {
    width: "100%",
    fontSize: 13,
    lineHeight: 18,
    fontWeight: "800",
    color: colors.greenDark,
    textAlign: "center",
    textAlignVertical: "center",
    includeFontPadding: false
  },
  activeCornerText: {
    color: "#ffffff"
  },
  warning: {
    ...typography.body,
    color: colors.warning,
    fontWeight: "800"
  }
});
