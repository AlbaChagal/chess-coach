import { Pressable, StyleSheet, Text, View } from "react-native";

import { colors } from "../theme";

export const STAGES = ["Capture", "Detect", "Setup", "Ready", "Analyze"];
export const STAGE_ROUTES = {
  Capture: "Capture",
  Detect: "DetectBoard",
  Setup: "SetupPosition",
  Ready: "Ready",
  Analyze: "Analysis"
};

function isStageAvailable(stage, flowState) {
  if (stage === "Capture") {
    return true;
  }
  if (stage === "Detect") {
    return Boolean(flowState?.image && flowState?.boardDetection);
  }
  if (stage === "Setup") {
    return Boolean(
      flowState?.image &&
        flowState?.boardDetection &&
        flowState?.whiteKingStartClick
    );
  }
  if (stage === "Ready") {
    return Boolean(
      flowState?.image &&
        flowState?.boardDetection &&
        flowState?.whiteKingStartClick &&
        flowState?.position
    );
  }
  if (stage === "Analyze") {
    return Boolean(flowState?.session);
  }
  return false;
}

export function StageProgress({ current, flowState = null, onStagePress = null }) {
  const currentIndex = STAGES.indexOf(current);
  return (
    <View style={styles.wrap} accessibilityLabel={`Current step ${current}`}>
      {STAGES.map((stage, index) => {
        const complete = index < currentIndex;
        const active = index === currentIndex;
        const available = active || isStageAvailable(stage, flowState);
        const clickable = Boolean(onStagePress && available && !active);
        return (
          <Pressable
            key={stage}
            accessibilityRole="button"
            accessibilityState={{
              disabled: !clickable,
              selected: active
            }}
            disabled={!clickable}
            onPress={() => onStagePress?.(stage)}
            style={({ pressed }) => [
              styles.item,
              !available && styles.unavailableItem,
              clickable && pressed && styles.pressedItem
            ]}
          >
            <View
              style={[
                styles.dot,
                complete && styles.complete,
                active && styles.active,
                !available && styles.inactiveDot
              ]}
            >
              <Text style={[styles.dotText, (complete || active) && styles.dotTextOn]}>
                {complete ? "✓" : index + 1}
              </Text>
            </View>
            <Text
              style={[
                styles.label,
                active && styles.activeLabel,
                available && !active && styles.availableLabel,
                !available && styles.inactiveLabel
              ]}
            >
              {stage}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    paddingVertical: 4
  },
  item: {
    flex: 1,
    alignItems: "center",
    gap: 4,
    paddingVertical: 2
  },
  pressedItem: {
    opacity: 0.85
  },
  unavailableItem: {
    opacity: 0.45
  },
  dot: {
    width: 25,
    height: 25,
    borderRadius: 13,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: colors.paper
  },
  active: {
    borderColor: colors.green,
    backgroundColor: colors.green
  },
  inactiveDot: {
    borderColor: colors.line,
    backgroundColor: "#faf7f0"
  },
  complete: {
    borderColor: colors.green,
    backgroundColor: colors.green
  },
  dotText: {
    fontSize: 12,
    fontWeight: "800",
    color: colors.muted
  },
  dotTextOn: {
    color: "#ffffff"
  },
  label: {
    fontSize: 10,
    color: colors.muted,
    fontWeight: "700"
  },
  activeLabel: {
    color: colors.greenDark
  },
  availableLabel: {
    color: colors.ink
  },
  inactiveLabel: {
    color: colors.muted
  }
});
