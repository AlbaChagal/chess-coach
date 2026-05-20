import { useState } from "react";
import { Alert } from "react-native";

import { AppButton } from "../components/AppButton";
import { BackendConnectionCard } from "../components/BackendConnectionCard";
import { Card } from "../components/Card";
import { Field } from "../components/Field";
import { Screen } from "../components/Screen";
import { useAppState } from "../state/AppContext";

export function SignupScreen({ navigation }) {
  const { api, setSettings, setUser } = useAppState();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState(false);

  const handleSignup = async () => {
    if (password !== confirmPassword) {
      Alert.alert("Passwords do not match", "Please confirm the same password.");
      return;
    }
    setBusy(true);
    try {
      const payload = await api.signup(email.trim(), password);
      try {
        const settingsPayload = await api.loadSettings();
        setSettings(settingsPayload.settings);
      } catch {
        setSettings({ show_coordinates: true });
      }
      setUser(payload.user);
    } catch (error) {
      Alert.alert("Sign up failed", error.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen title="Create Your Account" subtitle="Email and password for now.">
      <Card>
        <Field label="Email" value={email} onChangeText={setEmail} />
        <Field
          label="Password"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />
        <Field
          label="Confirm Password"
          secureTextEntry
          value={confirmPassword}
          onChangeText={setConfirmPassword}
        />
        <AppButton
          disabled={!email || !password || !confirmPassword || busy}
          title={busy ? "Creating..." : "Sign Up"}
          onPress={handleSignup}
        />
        <AppButton
          title="Log In Instead"
          variant="secondary"
          onPress={() => navigation.navigate("Login")}
        />
      </Card>
      <BackendConnectionCard api={api} />
    </Screen>
  );
}
