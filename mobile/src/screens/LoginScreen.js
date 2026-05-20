import { useState } from "react";
import { Alert, StyleSheet, Text, View } from "react-native";

import { AppButton } from "../components/AppButton";
import { BackendConnectionCard } from "../components/BackendConnectionCard";
import { Card } from "../components/Card";
import { Field } from "../components/Field";
import { Screen } from "../components/Screen";
import { useAppState } from "../state/AppContext";
import { colors, spacing, typography } from "../theme";

export function LoginScreen({ navigation }) {
  const { api, setSettings, setUser } = useAppState();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  const handleLogin = async () => {
    setBusy(true);
    try {
      const payload = await api.login(email.trim(), password);
      setUser(payload.user);
      const settingsPayload = await api.loadSettings();
      setSettings(settingsPayload.settings);
    } catch (error) {
      Alert.alert("Login failed", error.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen
      title="Chess Insight"
      subtitle="Load a real board, correct it, and get engine-backed coaching."
    >
      <Card style={styles.hero}>
        <Text style={styles.mark}>♞</Text>
        <Text style={styles.heroTitle}>Improve your chess with AI analysis</Text>
      </Card>
      <Card>
        <Field label="Email" value={email} onChangeText={setEmail} />
        <Field
          label="Password"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />
        <AppButton
          disabled={!email || !password || busy}
          title={busy ? "Logging in..." : "Log In"}
          onPress={handleLogin}
        />
        <View style={styles.switchRow}>
          <Text style={styles.muted}>No account yet?</Text>
          <AppButton
            title="Sign Up"
            variant="secondary"
            onPress={() => navigation.navigate("Signup")}
          />
        </View>
      </Card>
      <BackendConnectionCard api={api} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  hero: {
    alignItems: "center",
    backgroundColor: colors.ink
  },
  mark: {
    color: colors.gold,
    fontSize: 52
  },
  heroTitle: {
    ...typography.sectionTitle,
    color: "#ffffff",
    textAlign: "center"
  },
  switchRow: {
    gap: spacing.sm
  },
  muted: {
    color: colors.muted,
    textAlign: "center",
    fontWeight: "700"
  }
});
