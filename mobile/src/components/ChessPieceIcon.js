import { Image, StyleSheet, View } from "react-native";

const PIECE_IMAGES = {
  K: require("../../assets/pieces/cburnett/wK.png"),
  Q: require("../../assets/pieces/cburnett/wQ.png"),
  R: require("../../assets/pieces/cburnett/wR.png"),
  B: require("../../assets/pieces/cburnett/wB.png"),
  N: require("../../assets/pieces/cburnett/wN.png"),
  P: require("../../assets/pieces/cburnett/wP.png"),
  k: require("../../assets/pieces/cburnett/bK.png"),
  q: require("../../assets/pieces/cburnett/bQ.png"),
  r: require("../../assets/pieces/cburnett/bR.png"),
  b: require("../../assets/pieces/cburnett/bB.png"),
  n: require("../../assets/pieces/cburnett/bN.png"),
  p: require("../../assets/pieces/cburnett/bP.png")
};

export function ChessPieceIcon({ piece, size = 34 }) {
  const source = PIECE_IMAGES[piece];
  if (!source) {
    return null;
  }
  return (
    <View pointerEvents="none" style={[styles.frame, { width: size, height: size }]}>
      <Image
        resizeMode="contain"
        source={source}
        style={{ width: size, height: size }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  frame: {
    alignItems: "center",
    justifyContent: "center"
  }
});
