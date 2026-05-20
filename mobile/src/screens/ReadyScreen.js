import { useEffect, useState } from "react";
import { Alert, Image, StyleSheet, Text, View } from "react-native";

import { AppButton } from "../components/AppButton";
import { Card } from "../components/Card";
import { ChessBoard } from "../components/ChessBoard";
import { PieceEditor } from "../components/PieceEditor";
import { Screen } from "../components/Screen";
import {
  STAGE_ROUTES,
  STAGES,
  StageProgress
} from "../components/StageProgress";
import { useAppState } from "../state/AppContext";
import { colors, spacing, typography } from "../theme";
import {
  fullFenFromParts,
  setPieceAtSquare,
  validatePlacement
} from "../utils/chess";

export function ReadyScreen({ navigation }) {
  const {
    api,
    boardDetection,
    image,
    orientation,
    position,
    session,
    setOrientation,
    setPosition,
    sideToMove,
    startAnalysisSession,
    whiteKingStartClick
  } = useAppState();
  const [draftPlacement, setDraftPlacement] = useState(position?.fen_placement || "");
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [busy, setBusy] = useState(false);
  const currentStage = "Ready";
  const flowState = {
    image,
    boardDetection,
    whiteKingStartClick,
    position,
    session
  };

  useEffect(() => {
    if (position?.fen_placement) {
      setDraftPlacement(position.fen_placement);
    }
  }, [position?.fen_placement]);

  const draftFen = fullFenFromParts({
    placement: draftPlacement || position?.fen_placement || "8/8/8/8/8/8/8/8",
    sideToMove: position?.side_to_move || sideToMove,
    castlingRights: position?.castling_rights || "-",
    enPassant: position?.en_passant || "-"
  });

  const handleSetPiece = (piece) => {
    if (!selectedSquare) {
      return;
    }
    setDraftPlacement(setPieceAtSquare(draftPlacement, selectedSquare, piece));
  };

  const handleClear = () => {
    if (!selectedSquare) {
      return;
    }
    setDraftPlacement(setPieceAtSquare(draftPlacement, selectedSquare, null));
  };

  const handleApply = async () => {
    const validationMessage = validatePlacement(draftPlacement);
    if (validationMessage) {
      Alert.alert("Position needs attention", validationMessage);
      return null;
    }
    const payload = await api.completePosition({
      fenPlacement: draftPlacement,
      sideToMove,
      whiteKingStartClick,
      castlingRights: null,
      enPassant: "-"
    });
    if (payload.status !== "success") {
      Alert.alert("Position is invalid", "Check the board and try again.");
      return null;
    }
    setPosition(payload.position);
    return payload.position;
  };

  const handleAnalyze = async () => {
    setBusy(true);
    try {
      const nextPosition = (await handleApply()) || position;
      if (!nextPosition) {
        return;
      }
      const payload = await api.analyze(nextPosition.fen, 3);
      startAnalysisSession(nextPosition, payload.analysis);
      navigation.navigate("Analysis");
    } catch (error) {
      Alert.alert("Analysis failed", error.message);
    } finally {
      setBusy(false);
    }
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

  if (!position) {
    return (
      <Screen title="Ready">
        <Text style={typography.body}>Complete a position first.</Text>
        <StageProgress
          current={currentStage}
          flowState={flowState}
          onStagePress={handleStagePress}
        />
        <AppButton
          title="Back to Setup"
          variant="secondary"
          onPress={() => navigation.goBack()}
        />
      </Screen>
    );
  }

  return (
    <Screen
      title="Ready to Analyze"
      subtitle="Correct any model mistakes before running the engine."
    >
      <StageProgress
        current={currentStage}
        flowState={flowState}
        onStagePress={handleStagePress}
      />
      <AppButton
        title="Back to Setup"
        variant="secondary"
        onPress={() => navigation.goBack()}
      />
      {image ? (
        <Card>
          <Text style={typography.sectionTitle}>Uploaded photo</Text>
          <Image source={{ uri: image.uri }} style={styles.image} />
        </Card>
      ) : null}
      <Card>
        <View style={styles.boardHeader}>
          <Text style={typography.sectionTitle}>Detected board</Text>
          <AppButton
            title="Flip Board"
            variant="secondary"
            onPress={() =>
              setOrientation(orientation === "white" ? "black" : "white")
            }
          />
        </View>
        <ChessBoard
          fen={draftFen}
          orientation={orientation}
          selectedSquare={selectedSquare}
          onSquarePress={setSelectedSquare}
        />
        <Text style={styles.help}>
          Tap a square to edit it. Changes are applied before analysis.
        </Text>
      </Card>
      <View style={styles.actions}>
        <AppButton title="Apply Corrections" variant="secondary" onPress={handleApply} />
        <AppButton
          disabled={busy}
          title={busy ? "Analyzing..." : "Analyze Position"}
          onPress={handleAnalyze}
        />
      </View>
      <PieceEditor
        square={selectedSquare}
        visible={Boolean(selectedSquare)}
        onClose={() => setSelectedSquare(null)}
        onSetPiece={handleSetPiece}
        onClear={handleClear}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  image: {
    width: "100%",
    aspectRatio: 1,
    borderRadius: 18,
    backgroundColor: colors.line
  },
  boardHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.sm
  },
  help: {
    ...typography.caption,
    fontWeight: "700"
  },
  actions: {
    gap: spacing.sm
  }
});
