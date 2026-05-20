import { useState } from "react";
import { Alert, StyleSheet, Text, View } from "react-native";

import { AppButton } from "../components/AppButton";
import { Card } from "../components/Card";
import { ChessBoard } from "../components/ChessBoard";
import { Screen } from "../components/Screen";
import {
  STAGE_ROUTES,
  STAGES,
  StageProgress
} from "../components/StageProgress";
import { useAppState } from "../state/AppContext";
import { typography } from "../theme";

function explanationText(explanation) {
  if (!explanation) {
    return "No explanation loaded yet.";
  }
  if (typeof explanation.explanation_text === "string") {
    return explanation.explanation_text;
  }
  if (typeof explanation.summary === "string") {
    return explanation.summary;
  }
  const structured = explanation.structured_explanation;
  if (structured?.summary) {
    return structured.summary;
  }
  return "The explanation service returned structured data without a summary.";
}

export function ExplanationScreen({ navigation, route }) {
  const {
    api,
    boardDetection,
    image,
    orientation,
    position,
    settings,
    session,
    whiteKingStartClick
  } = useAppState();
  const { fen, moveSan, moveUci } = route.params;
  const [busy, setBusy] = useState(false);
  const [payload, setPayload] = useState(null);
  const currentStage = "Analyze";
  const flowOffset = 1;
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
      navigation.pop(currentIndex - targetIndex + flowOffset);
      return;
    }
    navigation.navigate(STAGE_ROUTES[stage]);
  };

  const handleExplain = async () => {
    setBusy(true);
    try {
      const nextPayload = await api.explain({
        fen,
        playedMoveUci: moveUci,
        topN: 3
      });
      setPayload(nextPayload);
    } catch (error) {
      Alert.alert("Explanation failed", error.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen
      title={`Explain ${moveSan}`}
      subtitle="Generated only when you request it to control LLM cost."
    >
      <StageProgress
        current={currentStage}
        flowState={flowState}
        onStagePress={handleStagePress}
      />
      <AppButton
        title="Back to Analysis"
        variant="secondary"
        onPress={() => navigation.goBack()}
      />
      <Card>
        <Text style={typography.sectionTitle}>Explained position</Text>
        <View pointerEvents="none" style={styles.boardShell}>
          <ChessBoard
            arrowMove={moveUci}
            fen={fen}
            orientation={orientation}
            showCoordinates={settings.show_coordinates}
          />
        </View>
      </Card>
      <Card>
        <Text style={typography.sectionTitle}>Coach explanation</Text>
        <Text style={styles.body}>{explanationText(payload?.explanation)}</Text>
        <AppButton
          disabled={busy}
          title={busy ? "Generating..." : "Generate Explanation"}
          onPress={handleExplain}
        />
      </Card>
      {payload?.warnings?.length ? (
        <Card>
          <Text style={typography.sectionTitle}>Warnings</Text>
          {payload.warnings.map((warning) => (
            <Text key={warning.code} style={styles.body}>
              {warning.message}
            </Text>
          ))}
        </Card>
      ) : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  boardShell: {
    marginTop: 14
  },
  body: {
    ...typography.body
  }
});
