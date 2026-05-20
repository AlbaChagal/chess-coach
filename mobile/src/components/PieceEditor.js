import { Modal, Pressable, StyleSheet, Text, View } from "react-native";

import { ChessPieceIcon } from "./ChessPieceIcon";
import { colors, radii, spacing, typography } from "../theme";
import { AppButton } from "./AppButton";

const PIECES = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"];

export function PieceEditor({ square, visible, onClose, onSetPiece, onClear }) {
  return (
    <Modal animationType="slide" transparent visible={visible}>
      <View style={styles.backdrop}>
        <View style={styles.sheet}>
          <Text style={typography.sectionTitle}>
            {square ? `Edit ${square}` : "Edit square"}
          </Text>
          <Text style={styles.help}>Choose a piece or clear the square.</Text>
          <View style={styles.grid}>
            {PIECES.map((piece) => (
              <Pressable
                accessibilityRole="button"
                key={piece}
                onPress={() => onSetPiece(piece)}
                style={styles.pieceButton}
              >
                <ChessPieceIcon piece={piece} size={28} />
              </Pressable>
            ))}
          </View>
          <View style={styles.actions}>
            <AppButton title="Clear Square" variant="secondary" onPress={onClear} />
            <AppButton title="Done" onPress={onClose} />
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "rgba(20, 28, 24, 0.32)"
  },
  sheet: {
    gap: spacing.md,
    padding: spacing.lg,
    borderTopLeftRadius: radii.xl,
    borderTopRightRadius: radii.xl,
    backgroundColor: colors.cream
  },
  help: {
    ...typography.body,
    color: colors.muted
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.sm
  },
  pieceButton: {
    width: "14.8%",
    aspectRatio: 1,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radii.md,
    backgroundColor: colors.paper,
    borderWidth: 1,
    borderColor: colors.line
  },
  actions: {
    flexDirection: "row",
    gap: spacing.sm
  }
});
