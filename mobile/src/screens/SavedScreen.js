import { useCallback, useState } from "react";
import { Alert, Pressable, StyleSheet, Text, View } from "react-native";
import { useFocusEffect } from "@react-navigation/native";

import { AppButton } from "../components/AppButton";
import { Card } from "../components/Card";
import { Screen } from "../components/Screen";
import { useAppState } from "../state/AppContext";
import { colors, spacing, typography } from "../theme";

export function SavedScreen() {
  const { api } = useAppState();
  const [snapshots, setSnapshots] = useState([]);
  const [busy, setBusy] = useState(false);

  const loadSaved = useCallback(async () => {
    setBusy(true);
    try {
      const payload = await api.listSaved();
      setSnapshots(payload.snapshots || []);
    } catch (error) {
      Alert.alert("Saved positions failed", error.message);
    } finally {
      setBusy(false);
    }
  }, [api]);

  useFocusEffect(
    useCallback(() => {
      loadSaved();
    }, [loadSaved])
  );

  const handleDelete = async (id) => {
    try {
      await api.deleteSaved(id);
      await loadSaved();
    } catch (error) {
      Alert.alert("Delete failed", error.message);
    }
  };

  return (
    <Screen title="Saved" subtitle="Local backend snapshots tied to your account.">
      {busy ? <Text style={typography.body}>Loading...</Text> : null}
      {snapshots.length === 0 && !busy ? (
        <Card>
          <Text style={typography.sectionTitle}>No saved positions yet</Text>
          <Text style={typography.body}>Save an analysis from the Analyze tab.</Text>
        </Card>
      ) : null}
      {snapshots.map((snapshot) => (
        <Card key={snapshot.id}>
          <Text style={styles.title}>
            {snapshot.summary?.best_move_san || "Saved position"}
          </Text>
          <Text style={styles.meta}>
            {snapshot.summary?.side_to_move === "w" ? "White" : "Black"} to move
          </Text>
          <Text style={styles.meta}>{snapshot.summary?.fen}</Text>
          <View style={styles.actions}>
            <Pressable onPress={() => handleDelete(snapshot.id)}>
              <Text style={styles.deleteText}>Delete</Text>
            </Pressable>
          </View>
        </Card>
      ))}
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: {
    ...typography.sectionTitle,
    color: colors.ink
  },
  meta: {
    ...typography.caption
  },
  actions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: spacing.sm
  },
  deleteText: {
    color: colors.danger,
    fontWeight: "900"
  }
});
