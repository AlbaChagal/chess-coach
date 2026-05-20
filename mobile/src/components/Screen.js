import { ScrollView, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { colors, spacing, typography } from "../theme";

export function Screen({ title, subtitle, children, footer, scroll = true }) {
  const content = (
    <>
      <View style={styles.header}>
        <Text style={typography.title}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>
      {children}
    </>
  );

  return (
    <SafeAreaView style={styles.safe}>
      {scroll ? (
        <ScrollView
          contentContainerStyle={[styles.content, footer && styles.withFooter]}
          keyboardShouldPersistTaps="handled"
        >
          {content}
        </ScrollView>
      ) : (
        <View style={[styles.content, footer && styles.withFooter]}>{content}</View>
      )}
      {footer ? <View style={styles.footer}>{footer}</View> : null}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.cream
  },
  content: {
    gap: spacing.md,
    padding: spacing.md,
    paddingBottom: spacing.lg
  },
  withFooter: {
    paddingBottom: 112
  },
  header: {
    gap: spacing.xs,
    paddingTop: spacing.sm
  },
  subtitle: {
    ...typography.body,
    color: colors.muted
  },
  footer: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    padding: spacing.md,
    borderTopWidth: 1,
    borderColor: colors.line,
    backgroundColor: "rgba(247, 243, 234, 0.96)"
  }
});
