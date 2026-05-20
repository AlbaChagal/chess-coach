const APP_ENV = process.env.APP_ENV || "development";
const API_URL =
  process.env.EXPO_PUBLIC_CHESSCOACH_API_URL || "http://127.0.0.1:8000";
const IOS_BUNDLE_ID =
  process.env.EXPO_PUBLIC_CHESSCOACH_IOS_BUNDLE_ID ||
  "com.shaharheyman.chesscoach";
const ANDROID_PACKAGE =
  process.env.EXPO_PUBLIC_CHESSCOACH_ANDROID_PACKAGE ||
  "com.shaharheyman.chesscoach";

export default {
  expo: {
    name: "ChessCoach",
    slug: "chesscoach-mobile",
    scheme: "chesscoach",
    version: "0.1.0",
    orientation: "portrait",
    userInterfaceStyle: "light",
    ios: {
      bundleIdentifier: IOS_BUNDLE_ID,
      buildNumber: "1",
      supportsTablet: true,
      infoPlist: {
        NSAppTransportSecurity: {
          NSAllowsArbitraryLoads: true
        },
        NSCameraUsageDescription:
          "ChessCoach uses the camera to capture chess board positions.",
        NSPhotoLibraryUsageDescription:
          "ChessCoach uses your photos to analyze chess board positions."
      }
    },
    android: {
      package: ANDROID_PACKAGE,
      versionCode: 1,
      adaptiveIcon: {
        backgroundColor: "#f7f3ea"
      }
    },
    extra: {
      appEnv: APP_ENV,
      apiUrl: API_URL
    },
    plugins: ["expo-asset", "expo-secure-store"]
  }
};
