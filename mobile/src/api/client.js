import * as SecureStore from "expo-secure-store";
import Constants from "expo-constants";

import {
  deriveApiUrlFromHost,
  isLoopbackUrl,
  normalizeBaseUrl
} from "./config";

const SESSION_COOKIE_KEY = "chesscoach.sessionCookie";
const API_URL_KEY = "chesscoach.apiUrl";

function configuredApiUrl() {
  const envConfigured = process.env.EXPO_PUBLIC_CHESSCOACH_API_URL;
  if (envConfigured) {
    return normalizeBaseUrl(envConfigured);
  }
  const extraConfigured = Constants.expoConfig?.extra?.apiUrl;
  if (extraConfigured) {
    return normalizeBaseUrl(extraConfigured);
  }
  return null;
}

function deriveDefaultApiUrl() {
  const hostUri =
    Constants.expoConfig?.hostUri ||
    Constants.manifest2?.extra?.expoClient?.hostUri ||
    Constants.manifest?.debuggerHost;
  return deriveApiUrlFromHost(hostUri);
}

function networkErrorMessage(baseUrl) {
  return (
    `Could not reach ChessCoach backend at ${baseUrl}. ` +
    "Make sure the Mac backend is running with --host 0.0.0.0, " +
    "the iPhone is on the same Wi-Fi, and this URL uses your Mac LAN IP."
  );
}

export class ChessCoachApi {
  constructor({ baseUrl = configuredApiUrl() || deriveDefaultApiUrl() } = {}) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
    this.sessionCookie = null;
  }

  async restoreConfig() {
    const configured = configuredApiUrl();
    if (configured) {
      this.baseUrl = configured;
      return this.restoreSession();
    }
    const storedBaseUrl = await SecureStore.getItemAsync(API_URL_KEY);
    if (storedBaseUrl && !isLoopbackUrl(storedBaseUrl)) {
      this.baseUrl = normalizeBaseUrl(storedBaseUrl);
    } else {
      this.baseUrl = deriveDefaultApiUrl();
    }
    return this.restoreSession();
  }

  async setBaseUrl(baseUrl) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
    await SecureStore.setItemAsync(API_URL_KEY, this.baseUrl);
  }

  async restoreSession() {
    this.sessionCookie = await SecureStore.getItemAsync(SESSION_COOKIE_KEY);
    return this.sessionCookie;
  }

  async clearSession() {
    this.sessionCookie = null;
    await SecureStore.deleteItemAsync(SESSION_COOKIE_KEY);
  }

  async login(email, password) {
    return this.postAuth("/auth/login", { email, password });
  }

  async signup(email, password) {
    return this.postAuth("/auth/signup", { email, password });
  }

  async logout() {
    await this.request("/auth/logout", { method: "POST" });
    await this.clearSession();
  }

  async me() {
    return this.request("/auth/me");
  }

  async health() {
    return this.request("/health");
  }

  async detectBoard(imageBase64) {
    return this.request("/detect-board", {
      method: "POST",
      body: { image_base64: imageBase64 }
    });
  }

  async runVision({ imageBase64, whiteKingStartClick, boardCorners }) {
    return this.request("/vision", {
      method: "POST",
      body: {
        image_base64: imageBase64,
        white_king_start_click: whiteKingStartClick,
        board_corners: boardCorners
      }
    });
  }

  async completePosition({
    fenPlacement,
    sideToMove,
    whiteKingStartClick,
    castlingRights,
    enPassant
  }) {
    return this.request("/complete-position", {
      method: "POST",
      body: {
        fen_placement: fenPlacement,
        side_to_move: sideToMove,
        white_king_start_click: whiteKingStartClick,
        castling_rights: castlingRights,
        en_passant: enPassant
      }
    });
  }

  async analyze(fen, topN = 3) {
    return this.request("/analyze", {
      method: "POST",
      body: { fen, top_n: topN }
    });
  }

  async legalMoves(fen) {
    return this.request("/legal-moves", {
      method: "POST",
      body: { fen }
    });
  }

  async playMove(fen, moveUci) {
    return this.request("/play-move", {
      method: "POST",
      body: { fen, move_uci: moveUci }
    });
  }

  async explain({ fen, playedMoveUci = null, topN = 3 }) {
    return this.request("/explain", {
      method: "POST",
      body: {
        fen,
        played_move_uci: playedMoveUci,
        top_n: topN
      }
    });
  }

  async loadSettings() {
    return this.request("/api/settings");
  }

  async saveSettings(settings) {
    return this.request("/api/settings", {
      method: "POST",
      body: settings
    });
  }

  async listSaved() {
    return this.request("/api/saved");
  }

  async getSaved(id) {
    return this.request(`/api/saved/${id}`);
  }

  async saveSnapshot(snapshot) {
    return this.request("/api/saved", {
      method: "POST",
      body: { snapshot }
    });
  }

  async deleteSaved(id) {
    return this.request(`/api/saved/${id}`, {
      method: "DELETE"
    });
  }

  async postAuth(path, body) {
    const payload = await this.request(path, {
      method: "POST",
      body,
      captureCookie: true
    });
    if (payload.native_session_cookie) {
      this.sessionCookie = payload.native_session_cookie;
      await SecureStore.setItemAsync(SESSION_COOKIE_KEY, this.sessionCookie);
    }
    return payload;
  }

  async request(path, options = {}) {
    const headers = {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(this.sessionCookie ? { Cookie: this.sessionCookie } : {})
    };
    let response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method: options.method || "GET",
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined
      });
    } catch (error) {
      throw new Error(networkErrorMessage(this.baseUrl), { cause: error });
    }

    if (options.captureCookie) {
      const cookie = response.headers.get("set-cookie");
      if (cookie) {
        this.sessionCookie = cookie.split(";")[0];
        await SecureStore.setItemAsync(SESSION_COOKIE_KEY, this.sessionCookie);
      }
    }

    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }
    return payload;
  }
}

export const api = new ChessCoachApi();
