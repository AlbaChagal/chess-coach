import { useState } from "react";
import { Alert, StyleSheet, Text } from "react-native";

import { AppButton } from "../components/AppButton";
import { BoardSelectionSurface } from "../components/BoardSelectionSurface";
import { Card } from "../components/Card";
import { Screen } from "../components/Screen";
import { SegmentedControl } from "../components/SegmentedControl";
import {
  STAGE_ROUTES,
  STAGES,
  StageProgress
} from "../components/StageProgress";
import { useAppState } from "../state/AppContext";
import { typography } from "../theme";

/**
 * Lets the user confirm the White king start square on the detected board.
 *
 * @param {{navigation: object}} props - Navigation props.
 * @returns {JSX.Element} The setup position screen.
 */
export function SetupPositionScreen({ navigation }) {
  const {
    api,
    boardDetection,
    image,
    position,
    session,
    setPosition,
    setSideToMove,
    setVision,
    setWhiteKingStartClick,
    sideToMove,
    whiteKingStartClick
  } = useAppState();
  const [busy, setBusy] = useState(false);
  const currentStage = "Setup";
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

  const handleComplete = async () => {
    if (!image || !whiteKingStartClick) {
      return;
    }
    setBusy(true);
    try {
      const visionPayload = await api.runVision({
        imageBase64: image.dataUrl,
        whiteKingStartClick,
        boardCorners: boardDetection?.board_corners || null
      });
      if (!visionPayload.vision?.fen_placement) {
        Alert.alert(
          "Board could not be read",
          "Please upload a clearer image or adjust the detected board."
        );
        return;
      }
      setVision(visionPayload.vision);
      const completionPayload = await api.completePosition({
        fenPlacement: visionPayload.vision.fen_placement,
        sideToMove,
        whiteKingStartClick,
        castlingRights: null,
        enPassant: "-"
      });
      if (completionPayload.status !== "success") {
        Alert.alert("Position is invalid", "Correct the setup and try again.");
        return;
      }
      setPosition(completionPayload.position);
      navigation.navigate("Ready");
    } catch (error) {
      Alert.alert("Setup failed", error.message);
    } finally {
      setBusy(false);
    }
  };

  if (!image) {
    return (
      <Screen title="Setup Position" subtitle="Load a board photo first.">
        <Text style={typography.body}>No image is available.</Text>
      </Screen>
    );
  }

  return (
    <Screen
      title="Setup Position"
      subtitle={
        "Tap the White king start square on the detected board. " +
        "Long press for precision."
      }
    >
      <StageProgress
        current={currentStage}
        flowState={flowState}
        onStagePress={handleStagePress}
      />
      <AppButton
        title="Back to Detect"
        variant="secondary"
        onPress={() => navigation.goBack()}
      />
      <Card>
        <BoardSelectionSurface
          boardCorners={boardDetection?.board_corners || null}
          image={image}
          onSelectPoint={setWhiteKingStartClick}
          selectedPoint={whiteKingStartClick}
        />
        <Text style={styles.help}>
          The marker locks to the detected board, then translates back to the raw
          image for vision.
        </Text>
        <SegmentedControl
          onChange={setSideToMove}
          options={[
            { label: "White to move", value: "w" },
            { label: "Black to move", value: "b" }
          ]}
          value={sideToMove}
        />
      </Card>
      <AppButton
        disabled={!whiteKingStartClick || busy}
        title={busy ? "Reading Position..." : "Complete Position"}
        onPress={handleComplete}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  help: {
    ...typography.caption,
    fontWeight: "700"
  }
});
