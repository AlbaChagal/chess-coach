import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { deriveApiUrlFromHost, isLoopbackUrl } from "./config.js";

describe("api client configuration", () => {
  it("derives the backend URL from the Expo host", () => {
    assert.equal(
      deriveApiUrlFromHost("192.168.178.47:8081"),
      "http://192.168.178.47:8000"
    );
  });

  it("detects loopback URLs that cannot work from a phone", () => {
    assert.equal(isLoopbackUrl("http://127.0.0.1:8000"), true);
    assert.equal(isLoopbackUrl("http://localhost:8000"), true);
    assert.equal(isLoopbackUrl("http://192.168.178.47:8000"), false);
  });
});
