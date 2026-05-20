import { createContext, useContext, useMemo, useState } from "react";

import { api } from "../api/client";
import { createAnalysisSession } from "./analysisSession";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [user, setUser] = useState(null);
  const [settings, setSettings] = useState({ show_coordinates: true });
  const [image, setImage] = useState(null);
  const [boardDetection, setBoardDetection] = useState(null);
  const [whiteKingStartClick, setWhiteKingStartClick] = useState(null);
  const [sideToMove, setSideToMove] = useState("w");
  const [vision, setVision] = useState(null);
  const [position, setPosition] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [session, setSession] = useState(null);
  const [orientation, setOrientation] = useState("white");
  const [statusMessage, setStatusMessage] = useState(null);

  const resetAnalyzeFlow = () => {
    setImage(null);
    setBoardDetection(null);
    setWhiteKingStartClick(null);
    setSideToMove("w");
    setVision(null);
    setPosition(null);
    setAnalysis(null);
    setSession(null);
    setOrientation("white");
    setStatusMessage(null);
  };

  const startAnalysisSession = (rootPosition, nextAnalysis) => {
    setPosition(rootPosition);
    setAnalysis(nextAnalysis);
    setSession(
      createAnalysisSession({
        rootFen: rootPosition.fen,
        analysis: nextAnalysis
      })
    );
  };

  const value = useMemo(
    () => ({
      api,
      user,
      setUser,
      settings,
      setSettings,
      image,
      setImage,
      boardDetection,
      setBoardDetection,
      whiteKingStartClick,
      setWhiteKingStartClick,
      sideToMove,
      setSideToMove,
      vision,
      setVision,
      position,
      setPosition,
      analysis,
      setAnalysis,
      session,
      setSession,
      orientation,
      setOrientation,
      statusMessage,
      setStatusMessage,
      resetAnalyzeFlow,
      startAnalysisSession
    }),
    [
      user,
      settings,
      image,
      boardDetection,
      whiteKingStartClick,
      sideToMove,
      vision,
      position,
      analysis,
      session,
      orientation,
      statusMessage
    ]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppState() {
  const value = useContext(AppContext);
  if (!value) {
    throw new Error("useAppState must be used inside AppProvider.");
  }
  return value;
}
