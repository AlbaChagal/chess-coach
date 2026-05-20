import { useEffect, useMemo, useState } from "react";
import {
  Image,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
  useWindowDimensions
} from "react-native";

import { colors, radii, spacing, typography } from "../theme";
import {
  getPrecisionOverlayGeometry,
  getSourcePointFromPrecisionTap
} from "../utils/cornerPrecision";
import { AppButton } from "./AppButton";

const ZOOM_SCALE = 3;

export function CornerPrecisionOverlay({
  visible,
  imageUri,
  imageWidth,
  imageHeight,
  focusPoint,
  cornerLabel,
  onClose,
  onSelectPoint
}) {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const viewportWidth = Math.min(windowWidth - spacing.lg * 2, 420);
  const viewportHeight = Math.min(windowHeight * 0.42, 380);
  const [selectedPoint, setSelectedPoint] = useState(focusPoint);

  useEffect(() => {
    setSelectedPoint(focusPoint);
  }, [focusPoint]);

  const geometry = useMemo(
    () =>
      getPrecisionOverlayGeometry({
        focusPoint: selectedPoint || focusPoint,
        imageWidth,
        imageHeight,
        viewportWidth,
        viewportHeight,
        zoomScale: ZOOM_SCALE
      }),
    [
      imageHeight,
      imageWidth,
      selectedPoint,
      focusPoint,
      viewportHeight,
      viewportWidth
    ]
  );

  if (!visible || !geometry) {
    return null;
  }

  const handleTap = (event) => {
    const { locationX, locationY } = event.nativeEvent;
    const nextPoint = getSourcePointFromPrecisionTap(
      locationX,
      locationY,
      geometry,
      imageWidth,
      imageHeight
    );

    if (!nextPoint) {
      return;
    }

    setSelectedPoint(nextPoint);
  };

  const handleConfirm = () => {
    if (!selectedPoint) {
      return;
    }
    onSelectPoint(selectedPoint);
    onClose();
  };

  return (
    <Modal
      animationType="fade"
      onRequestClose={onClose}
      presentationStyle="overFullScreen"
      transparent
      visible={visible}
    >
      <View style={styles.backdrop}>
        <View style={[styles.sheet, { width: viewportWidth + spacing.lg * 2 }]}>
          <View style={styles.header}>
            <View style={styles.titleBlock}>
              <Text style={typography.sectionTitle}>Precision zoom</Text>
              <Text style={styles.help}>
                Tap the zoomed image to refine {cornerLabel}, then confirm it.
              </Text>
            </View>
            <View style={styles.badge}>
              <Text style={styles.badgeText}>{cornerLabel}</Text>
            </View>
          </View>
          <View style={styles.viewportShell}>
            <Pressable
              accessibilityRole="button"
              onPress={handleTap}
              style={[
                styles.viewport,
                { width: viewportWidth, height: viewportHeight }
              ]}
            >
              <Image
                pointerEvents="none"
                source={{ uri: imageUri }}
                style={[
                  styles.zoomedImage,
                  {
                    left: geometry.imageLeft,
                    top: geometry.imageTop,
                    width: geometry.imageWidth,
                    height: geometry.imageHeight
                  }
                ]}
                resizeMode="contain"
              />
              <View
                pointerEvents="none"
                style={[
                  styles.crosshair,
                  {
                    left:
                      geometry.imageLeft +
                      (selectedPoint || focusPoint).x * geometry.zoomScale,
                    top:
                      geometry.imageTop +
                      (selectedPoint || focusPoint).y * geometry.zoomScale
                  }
                ]}
              />
            </Pressable>
          </View>
          <Text style={styles.caption}>
            Long press opens this view near your touch. Tap again to move the
            marker before confirming.
          </Text>
          <View style={styles.actions}>
            <AppButton title="Use Point" onPress={handleConfirm} />
            <AppButton title="Cancel" variant="secondary" onPress={onClose} />
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.md,
    backgroundColor: "rgba(20, 35, 27, 0.62)"
  },
  sheet: {
    gap: spacing.md,
    padding: spacing.lg,
    borderRadius: radii.xl,
    backgroundColor: colors.cream,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.35)"
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: spacing.md
  },
  titleBlock: {
    flex: 1,
    gap: spacing.xs
  },
  help: {
    ...typography.body,
    color: colors.muted
  },
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: colors.greenSoft
  },
  badgeText: {
    ...typography.label,
    color: colors.greenDark
  },
  viewportShell: {
    borderRadius: radii.lg,
    padding: 4,
    backgroundColor: colors.ink,
    shadowColor: "#0d1911",
    shadowOpacity: 0.2,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
    elevation: 6
  },
  viewport: {
    overflow: "hidden",
    borderRadius: radii.md,
    backgroundColor: colors.line
  },
  zoomedImage: {
    position: "absolute"
  },
  crosshair: {
    position: "absolute",
    width: 22,
    height: 22,
    marginLeft: -11,
    marginTop: -11,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: colors.boardSelect,
    backgroundColor: "rgba(240, 201, 74, 0.12)"
  },
  caption: {
    ...typography.caption,
    color: colors.muted,
    textAlign: "center"
  },
  actions: {
    flexDirection: "row",
    justifyContent: "flex-end"
  }
});
