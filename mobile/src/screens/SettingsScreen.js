import { Alert, Switch, StyleSheet, Text, View } from "react-native";

import { Card } from "../components/Card";
import { Screen } from "../components/Screen";
import { useAppState } from "../state/AppContext";
import { colors, typography } from "../theme";

export function SettingsScreen() {
  const { api, settings, setSettings } = useAppState();

  const handleToggleCoordinates = async (value) => {
    const previousSettings = settings;
    setSettings({ ...settings, show_coordinates: value });
    try {
      const payload = await api.saveSettings({ show_coordinates: value });
      setSettings(payload.settings);
    } catch (error) {
      setSettings(previousSettings);
      Alert.alert("Settings failed", error.message);
    }
  };

  return (
    <Screen title="Settings" subtitle="Minimal display settings for this phase.">
      <Card>
        <View style={styles.row}>
          <View style={styles.textWrap}>
            <Text style={typography.sectionTitle}>Board coordinates</Text>
            <Text style={typography.caption}>Show rank and file labels.</Text>
          </View>
          <Switch
            onValueChange={handleToggleCoordinates}
            thumbColor="#ffffff"
            trackColor={{ false: colors.line, true: colors.green }}
            value={settings.show_coordinates}
          />
        </View>
      </Card>
    </Screen>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between"
  },
  textWrap: {
    flex: 1
  }
});
