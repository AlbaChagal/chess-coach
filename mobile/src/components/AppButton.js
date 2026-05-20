import { Pressable, StyleSheet, Text } from "react-native";

import { colors, radii, spacing } from "../theme";

export function AppButton({
  title,
  onPress,
  disabled = false,
  variant = "primary",
  style
}) {
  return (
    <Pressable
      accessibilityRole="button"
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.base,
        variant === "primary" ? styles.primary : styles.secondary,
        disabled && styles.disabled,
        pressed && !disabled && styles.pressed,
        style
      ]}
    >
      <Text
        style={[
          styles.text,
          variant === "primary" ? styles.primaryText : styles.secondaryText,
          disabled && styles.disabledText
        ]}
      >
        {title}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    minHeight: 52,
    borderRadius: radii.md,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.md
  },
  primary: {
    backgroundColor: colors.green
  },
  secondary: {
    backgroundColor: colors.paper,
    borderColor: colors.line,
    borderWidth: 1
  },
  disabled: {
    backgroundColor: "#d8d5cc",
    borderColor: "#d8d5cc"
  },
  pressed: {
    transform: [{ scale: 0.99 }],
    opacity: 0.9
  },
  text: {
    fontSize: 15,
    fontWeight: "800"
  },
  primaryText: {
    color: "#ffffff"
  },
  secondaryText: {
    color: colors.greenDark
  },
  disabledText: {
    color: "#807d74"
  }
});
