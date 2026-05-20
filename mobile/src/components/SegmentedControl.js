import { Pressable, StyleSheet, Text, View } from "react-native";

import { colors, radii, spacing } from "../theme";

export function SegmentedControl({ options, value, onChange }) {
  return (
    <View style={styles.wrap}>
      {options.map((option) => {
        const active = option.value === value;
        return (
          <Pressable
            accessibilityRole="button"
            key={option.value}
            onPress={() => onChange(option.value)}
            style={[styles.option, active && styles.active]}
          >
            <Text style={[styles.text, active && styles.activeText]}>
              {option.label}
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
    gap: spacing.xs,
    padding: 4,
    borderRadius: radii.lg,
    backgroundColor: colors.paper,
    borderWidth: 1,
    borderColor: colors.line
  },
  option: {
    flex: 1,
    minHeight: 42,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radii.md
  },
  active: {
    backgroundColor: colors.green
  },
  text: {
    color: colors.muted,
    fontWeight: "800"
  },
  activeText: {
    color: "#ffffff"
  }
});
