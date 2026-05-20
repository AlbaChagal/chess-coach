import { useState } from "react";
import { Alert, StyleSheet, Text, View } from "react-native";

import { colors, spacing, typography } from "../theme";
import { AppButton } from "./AppButton";
import { Card } from "./Card";
import { Field } from "./Field";

export function BackendConnectionCard({ api }) {
  const [apiUrl, setApiUrl] = useState(api.baseUrl);
  const [busy, setBusy] = useState(false);

  const handleSave = async () => {
    await api.setBaseUrl(apiUrl.trim());
    Alert.alert("Backend URL saved", `Using ${api.baseUrl}`);
  };

  const handleTest = async () => {
    setBusy(true);
    try {
      await api.setBaseUrl(apiUrl.trim());
      const payload = await api.health();
      Alert.alert("Backend reachable", payload.status || "ok");
    } catch (error) {
      Alert.alert("Backend not reachable", error.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <Text style={typography.sectionTitle}>Backend connection</Text>
      <Text style={styles.help}>
        On iPhone this must be your Mac LAN IP, not 127.0.0.1.
      </Text>
      <Field
        autoCapitalize="none"
        keyboardType="url"
        label="API URL"
        onChangeText={setApiUrl}
        placeholder="http://192.168.178.47:8000"
        value={apiUrl}
      />
      <View style={styles.actions}>
        <AppButton title="Save URL" variant="secondary" onPress={handleSave} />
        <AppButton
          disabled={busy}
          title={busy ? "Testing..." : "Test Connection"}
          onPress={handleTest}
        />
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  help: {
    ...typography.caption,
    color: colors.muted
  },
  actions: {
    gap: spacing.sm
  }
});
