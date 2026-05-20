import { useState } from "react";
import { Alert, Image, StyleSheet, Text, View } from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as ImageManipulator from "expo-image-manipulator";

import { AppButton } from "../components/AppButton";
import { Card } from "../components/Card";
import { Screen } from "../components/Screen";
import {
  STAGE_ROUTES,
  STAGES,
  StageProgress
} from "../components/StageProgress";
import { useAppState } from "../state/AppContext";
import { colors, spacing, typography } from "../theme";

const MAX_UPLOAD_WIDTH = 1600;

async function normalizedImageFromAsset(asset) {
  const sourceWidth = asset.width || MAX_UPLOAD_WIDTH;
  const resizeWidth = Math.min(sourceWidth, MAX_UPLOAD_WIDTH);
  const actions = [{ resize: { width: resizeWidth } }];
  const normalized = await ImageManipulator.manipulateAsync(asset.uri, actions, {
    base64: true,
    compress: 0.9,
    format: ImageManipulator.SaveFormat.JPEG
  });
  if (!normalized.base64) {
    throw new Error("Could not prepare the image for upload.");
  }
  const cleanBase64 = normalized.base64.replace(/\s/g, "");
  return {
    uri: normalized.uri,
    dataUrl: `data:image/jpeg;base64,${cleanBase64}`,
    width: normalized.width,
    height: normalized.height
  };
}

export function CaptureScreen({ navigation }) {
  const {
    api,
    boardDetection,
    image,
    position,
    resetAnalyzeFlow,
    session,
    setBoardDetection,
    setImage,
    whiteKingStartClick
  } = useAppState();
  const [busy, setBusy] = useState(false);
  const currentStage = "Capture";
  const flowState = {
    image,
    boardDetection,
    whiteKingStartClick,
    position,
    session
  };

  const handleStagePress = (stage) => {
    const currentIndex = STAGES.indexOf(currentStage);
    const targetIndex = STAGES.indexOf(stage);
    if (targetIndex < 0 || targetIndex === currentIndex) {
      return;
    }
    if (targetIndex < currentIndex) {
      navigation.pop(currentIndex - targetIndex);
      return;
    }
    navigation.navigate(STAGE_ROUTES[stage]);
  };

  const handlePick = async (useCamera) => {
    const permission = useCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission needed", "Allow photo access to load a board.");
      return;
    }

    try {
      const result = useCamera
        ? await ImagePicker.launchCameraAsync({ quality: 0.85 })
        : await ImagePicker.launchImageLibraryAsync({ quality: 0.85 });
      if (result.canceled || !result.assets?.[0]?.uri) {
        return;
      }

      const normalizedImage = await normalizedImageFromAsset(result.assets[0]);
      resetAnalyzeFlow();
      setImage(normalizedImage);
    } catch (error) {
      Alert.alert("Image upload failed", error.message);
    }
  };

  const handleDetect = async () => {
    if (!image) {
      return;
    }
    setBusy(true);
    try {
      const payload = await api.detectBoard(image.dataUrl);
      setBoardDetection(payload.detection);
      navigation.navigate("DetectBoard");
    } catch (error) {
      Alert.alert("Board detection failed", error.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen title="Analyze a Board" subtitle="Start with a clear photo.">
      <StageProgress
        current={currentStage}
        flowState={flowState}
        onStagePress={handleStagePress}
      />
      <Card>
        {image ? (
          <Image source={{ uri: image.uri }} style={styles.preview} />
        ) : (
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>♘</Text>
            <Text style={styles.emptyText}>Upload or take a board photo</Text>
          </View>
        )}
        <View style={styles.actions}>
          <AppButton title="Upload Image" onPress={() => handlePick(false)} />
          <AppButton
            title="Take Photo"
            variant="secondary"
            onPress={() => handlePick(true)}
          />
        </View>
      </Card>
      <AppButton
        disabled={!image || busy}
        title={busy ? "Detecting..." : "Detect Board"}
        onPress={handleDetect}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  preview: {
    width: "100%",
    aspectRatio: 1,
    borderRadius: 18,
    backgroundColor: colors.line
  },
  empty: {
    aspectRatio: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
    borderWidth: 1,
    borderStyle: "dashed",
    borderColor: colors.line,
    borderRadius: 18,
    backgroundColor: "#fbf8ef"
  },
  emptyIcon: {
    fontSize: 58,
    color: colors.green
  },
  emptyText: {
    ...typography.body,
    color: colors.muted,
    fontWeight: "800"
  },
  actions: {
    flexDirection: "row",
    gap: spacing.sm
  }
});
