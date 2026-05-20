import "react-native-gesture-handler";

import { useEffect, useState } from "react";
import { ActivityIndicator, Text, View } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { StatusBar } from "expo-status-bar";

import { AppProvider, useAppState } from "./src/state/AppContext";
import { AnalysisScreen } from "./src/screens/AnalysisScreen";
import { CaptureScreen } from "./src/screens/CaptureScreen";
import { DetectBoardScreen } from "./src/screens/DetectBoardScreen";
import { ExplanationScreen } from "./src/screens/ExplanationScreen";
import { LoginScreen } from "./src/screens/LoginScreen";
import { ProfileScreen } from "./src/screens/ProfileScreen";
import { ReadyScreen } from "./src/screens/ReadyScreen";
import { SavedScreen } from "./src/screens/SavedScreen";
import { SettingsScreen } from "./src/screens/SettingsScreen";
import { SetupPositionScreen } from "./src/screens/SetupPositionScreen";
import { SignupScreen } from "./src/screens/SignupScreen";
import { colors } from "./src/theme";

const AuthStack = createNativeStackNavigator();
const AnalyzeStack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function AnalyzeNavigator() {
  return (
    <AnalyzeStack.Navigator
      initialRouteName="Capture"
      screenOptions={{ headerShown: false }}
    >
      <AnalyzeStack.Screen name="Capture" component={CaptureScreen} />
      <AnalyzeStack.Screen name="DetectBoard" component={DetectBoardScreen} />
      <AnalyzeStack.Screen name="SetupPosition" component={SetupPositionScreen} />
      <AnalyzeStack.Screen name="Ready" component={ReadyScreen} />
      <AnalyzeStack.Screen name="Analysis" component={AnalysisScreen} />
      <AnalyzeStack.Screen name="Explanation" component={ExplanationScreen} />
    </AnalyzeStack.Navigator>
  );
}

function AppTabs() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: colors.green,
        tabBarInactiveTintColor: colors.muted,
        tabBarStyle: {
          backgroundColor: colors.paper,
          borderTopColor: colors.line
        }
      }}
    >
      <Tab.Screen
        name="AnalyzeTab"
        component={AnalyzeNavigator}
        options={{ title: "Analyze", tabBarIcon: TabIcon("♞") }}
      />
      <Tab.Screen
        name="Saved"
        component={SavedScreen}
        options={{ tabBarIcon: TabIcon("♜") }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{ tabBarIcon: TabIcon("♚") }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ tabBarIcon: TabIcon("⚙") }}
      />
    </Tab.Navigator>
  );
}

function TabIcon(label) {
  return ({ color, size }) => (
    <Text style={{ color, fontSize: Math.max(size - 2, 18), fontWeight: "900" }}>
      {label}
    </Text>
  );
}

function AuthNavigator() {
  return (
    <AuthStack.Navigator screenOptions={{ headerShown: false }}>
      <AuthStack.Screen name="Login" component={LoginScreen} />
      <AuthStack.Screen name="Signup" component={SignupScreen} />
    </AuthStack.Navigator>
  );
}

function Root() {
  const { api, setSettings, setUser, user } = useAppState();
  const [booting, setBooting] = useState(true);

  useEffect(() => {
    let active = true;
    async function bootstrap() {
      try {
        await api.restoreConfig();
        const payload = await api.me();
        if (active) {
          setUser(payload.user);
          setSettings(payload.settings);
        }
      } catch {
        await api.clearSession();
      } finally {
        if (active) {
          setBooting(false);
        }
      }
    }
    bootstrap();
    return () => {
      active = false;
    };
  }, [api, setSettings, setUser]);

  if (booting) {
    return (
      <View
        style={{
          flex: 1,
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: colors.cream
        }}
      >
        <ActivityIndicator color={colors.green} />
      </View>
    );
  }

  return (
    <NavigationContainer>{user ? <AppTabs /> : <AuthNavigator />}</NavigationContainer>
  );
}

export default function App() {
  return (
    <AppProvider>
      <StatusBar style="dark" />
      <Root />
    </AppProvider>
  );
}
