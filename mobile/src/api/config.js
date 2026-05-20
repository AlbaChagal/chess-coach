export function normalizeBaseUrl(baseUrl) {
  return baseUrl.replace(/\/$/, "");
}

export function deriveApiUrlFromHost(hostUri) {
  const host = hostUri?.split(":")[0];
  if (host && host !== "localhost" && host !== "127.0.0.1") {
    return `http://${host}:8000`;
  }
  return "http://127.0.0.1:8000";
}

export function isLoopbackUrl(baseUrl) {
  return (
    baseUrl.includes("://127.0.0.1") ||
    baseUrl.includes("://localhost") ||
    baseUrl.includes("://0.0.0.0")
  );
}
