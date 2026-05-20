import { Alert, StyleSheet, Text } from "react-native";

import { AppButton } from "../components/AppButton";
import { Card } from "../components/Card";
import { Screen } from "../components/Screen";
import { useAppState } from "../state/AppContext";
import { typography } from "../theme";

export function ProfileScreen() {
  const { api, resetAnalyzeFlow, setUser, user } = useAppState();

  const handleLogout = async () => {
    try {
      await api.logout();
      resetAnalyzeFlow();
      setUser(null);
    } catch (error) {
      Alert.alert("Logout failed", error.message);
    }
  };

  return (
    <Screen title="Profile" subtitle="Account and session controls.">
      <Card>
        <Text style={typography.sectionTitle}>Signed in</Text>
        <Text style={styles.email}>{user?.email}</Text>
      </Card>
      <AppButton title="Log Out" variant="secondary" onPress={handleLogout} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  email: {
    ...typography.body,
    fontWeight: "800"
  }
});
