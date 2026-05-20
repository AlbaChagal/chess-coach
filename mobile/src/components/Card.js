import { StyleSheet, View } from "react-native";

import { colors, radii, shadow, spacing } from "../theme";

export function Card({ children, style }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

const styles = StyleSheet.create({
  card: {
    padding: spacing.md,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: colors.paper,
    ...shadow.card
  }
});
