import { useEffect, useState } from "react";
import { Alert, Image, Pressable, StyleSheet, Text, View } from "react-native";

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
import {
  branchWithAnalysis,
  currentLineMoves,
  playSuggestedMove,
  previousStep,
  suggestedMoveIndex,
  selectLine,
  resetSession
} from "../state/analysisSession";
import { colors, spacing, typography } from "../theme";

function topMoves(analysis) {
  return analysis?.top_moves || [];
}

function moveTitle(move, index) {
  return `${index + 1}. ${move.move_san || move.move_uci}`;
}

function lineSummary(move) {
  const continuation = move.continuation?.length
    ? move.continuation
    : move.continuation_uci || [];
  return [move.move_san || move.move_uci, ...continuation].join(" ");
}

export function AnalysisScreen({ navigation }) {
  const {
    api,
    boardDetection,
    image,
    orientation,
    position,
    session,
    setAnalysis,
    setOrientation,
    setSession,
    whiteKingStartClick,
    settings
  } = useAppState();
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [legalMoves, setLegalMoves] = useState([]);
  const [busy, setBusy] = useState(false);
  const currentStage = "Analyze";
  const flowState = {
    image,
    boardDetection,
    whiteKingStartClick,
    position,
    session
  };

  const analysis = session?.analysis;
  const boardFen = session?.currentFen || session?.rootFen;
  const currentMoves = session ? currentLineMoves(session) : [];
  const arrowMove = currentMoves[session?.stepIndex || 0] || null;
  const legalTargets = legalMoves
    .filter((move) => move.from === selectedSquare)
    .map((move) => move.to);

  useEffect(() => {
    let active = true;

    async function loadLegalMoves() {
      if (!session || !boardFen) {
        setLegalMoves([]);
        return;
      }
      try {
        const payload = await api.legalMoves(boardFen);
        if (active) {
          setLegalMoves(payload.legal_moves || []);
        }
      } catch (error) {
        if (active) {
          setLegalMoves([]);
          Alert.alert("Move validation failed", error.message);
        }
      }
    }

    setSelectedSquare(null);
    void loadLegalMoves();

    return () => {
      active = false;
    };
  }, [api, boardFen]);

  const handleSquarePress = async (square) => {
    if (!session || !boardFen) {
      return;
    }
    try {
      if (!selectedSquare) {
        if (!legalMoves.some((move) => move.from === square)) {
          return;
        }
        setSelectedSquare(square);
        return;
      }
      if (selectedSquare === square) {
        setSelectedSquare(null);
        return;
      }
      const move = legalMoves.find(
        (candidate) => candidate.from === selectedSquare && candidate.to === square
      );
      if (!move) {
        if (legalMoves.some((candidate) => candidate.from === square)) {
          setSelectedSquare(square);
        } else {
          setSelectedSquare(null);
        }
        return;
      }
      await handlePlayMove(move.uci);
      setSelectedSquare(null);
    } catch (error) {
      Alert.alert("Move failed", error.message);
    }
  };

  const handlePlayMove = async (moveUci) => {
    if (!session) {
      return;
    }
    setBusy(true);
    try {
      const played = await api.playMove(session.currentFen, moveUci);
      const suggestedIndex = suggestedMoveIndex(session, moveUci);
      if (suggestedIndex >= 0) {
        setSession(playSuggestedMove(session, moveUci, played.position.fen));
        return;
      }
      const nextAnalysis = await api.analyze(played.position.fen, 3);
      setAnalysis(nextAnalysis.analysis);
      setSession(
        branchWithAnalysis(session, {
          moveUci,
          nextFen: played.position.fen,
          analysis: nextAnalysis.analysis
        })
      );
    } finally {
      setBusy(false);
    }
  };

  const handleNext = async () => {
    if (!arrowMove) {
      return;
    }
    await handlePlayMove(arrowMove);
  };

  const handleSave = async () => {
    if (!position || !session?.analysis) {
      return;
    }
    try {
      await api.saveSnapshot({
        position: {
          ...position,
          fen: session.currentFen
        },
        analysis: session.analysis,
        explanation: null
      });
      Alert.alert("Saved", "This analysis snapshot is now in Saved.");
    } catch (error) {
      Alert.alert("Save failed", error.message);
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

  if (!session || !boardFen) {
    return (
      <Screen title="Analysis">
        <Text style={typography.body}>Analyze a position first.</Text>
        <StageProgress
          current={currentStage}
          flowState={flowState}
          onStagePress={handleStagePress}
        />
        <AppButton
          title="Back to Ready"
          variant="secondary"
          onPress={() => navigation.goBack()}
        />
        <AppButton
          title="Start New Upload"
          onPress={() => navigation.navigate("Capture")}
        />
      </Screen>
    );
  }

  return (
    <Screen title="Engine Analysis" subtitle="Tap legal moves to branch.">
      <StageProgress
        current={currentStage}
        flowState={flowState}
        onStagePress={handleStagePress}
      />
      <AppButton
        title="Back to Ready"
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
          <Text style={typography.sectionTitle}>Analysis board</Text>
          <AppButton
            title="Flip Board"
            variant="secondary"
            onPress={() =>
              setOrientation(orientation === "white" ? "black" : "white")
            }
          />
        </View>
        <ChessBoard
          arrowMove={arrowMove}
          fen={boardFen}
          legalTargets={legalTargets}
          onSquarePress={handleSquarePress}
          orientation={orientation}
          selectedSquare={selectedSquare}
          showCoordinates={settings.show_coordinates}
        />
        <View style={styles.playback}>
          <AppButton
            title="Previous"
            variant="secondary"
            onPress={() => setSession(previousStep(session))}
          />
          <AppButton
            title="Reset"
            variant="secondary"
            onPress={() => setSession(resetSession(session))}
          />
          <AppButton disabled={!arrowMove || busy} title="Next" onPress={handleNext} />
        </View>
        <AppButton title="Save Analysis" variant="secondary" onPress={handleSave} />
      </Card>
      <Card>
        <View style={styles.linesHeader}>
          <Text style={typography.sectionTitle}>Top lines</Text>
          <AppButton
            title="Explain Position"
            variant="secondary"
            onPress={() =>
              navigation.navigate("Explanation", {
                fen: session.currentFen,
                moveUci: null,
                moveSan: "Position"
              })
            }
          />
        </View>
        {topMoves(analysis).map((move, index) => (
          <Pressable
            key={move.move_uci}
            onPress={() => setSession(selectLine(session, index))}
            style={[
              styles.line,
              index === session.selectedLineIndex && styles.activeLine
            ]}
          >
            <View style={styles.lineHeader}>
              <Text style={styles.lineTitle}>{moveTitle(move, index)}</Text>
              <View style={styles.scoreBadge}>
                <Text style={styles.scoreText}>{move.score_display || "—"}</Text>
              </View>
            </View>
            <Text style={styles.continuation}>
              {lineSummary(move)}
            </Text>
          </Pressable>
        ))}
      </Card>
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
  linesHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.sm
  },
  playback: {
    flexDirection: "row",
    gap: spacing.sm
  },
  line: {
    gap: spacing.xs,
    padding: spacing.sm,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: colors.paper
  },
  activeLine: {
    borderColor: colors.green,
    backgroundColor: colors.greenSoft
  },
  lineHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.sm
  },
  lineTitle: {
    flex: 1,
    fontWeight: "900",
    color: colors.ink
  },
  continuation: {
    ...typography.caption
  },
  scoreBadge: {
    minWidth: 64,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.line,
    backgroundColor: colors.paper
  },
  scoreText: {
    fontSize: 12,
    fontWeight: "900",
    color: colors.ink
  }
});
