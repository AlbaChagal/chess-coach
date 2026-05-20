# ChessCoach Mobile

This directory contains the Expo React Native mobile client for ChessCoach.

It connects to the FastAPI backend in the repository root and currently
implements the authenticated mobile analysis flow end to end.

## Implemented Features

- email/password auth
- persistent session restore
- bottom-tab app shell:
  - Analyze
  - Saved
  - Profile
  - Settings
- staged analyze flow:
  - Capture
  - Detect
  - Setup
  - Ready
  - Analyze
- image capture and upload
- board corner correction
- precision zoom for corner placement
- White king start square selection
- side-to-move selection
- ready-stage piece correction
- interactive analysis board
- suggested-line playback
- branch analysis from legal user moves
- per-line explanation requests
- synced settings
- saved snapshots
- standard readable chess piece assets bundled locally

## Local Development

Start the backend from the repository root:

```bash
cd /Users/shaharheyman/PycharmProjects/chesscoach
uv run python -m chesscoach.cli serve --host 0.0.0.0 --port 8000
```

Install and start the mobile app from this directory:

```bash
npm install
EXPO_PUBLIC_CHESSCOACH_API_URL=http://YOUR_MAC_LAN_IP:8000 npm run start
```

For simulator-only or local loopback workflows, you can omit the env var and
let the app derive the base URL when possible, but for a physical phone you
should set it explicitly.

## Real iPhone Testing

Requirements:

- Expo Go installed on the phone
- phone and Mac on the same Wi‑Fi
- backend running with `--host 0.0.0.0`

Then:

1. Start Expo with `npm run start`.
2. Scan the QR code in Expo Go.
3. Log in or sign up.
4. Run the staged analysis flow.

Do not use `127.0.0.1` as the backend host from a physical phone.

## Tests

Run the mobile JS tests:

```bash
npm test
```

## Build Profiles

This app now includes EAS build profiles in `eas.json`:

- `development`
- `preview`
- `production`

Preview build commands:

```bash
npm run build:ios:preview
npm run build:android:preview
```

Production build commands:

```bash
npm run build:ios:production
npm run build:android:production
```

Before running remote builds, set the expected environment values locally or in
your EAS environment:

```bash
cp .env.example .env.local
```

## Notes

- This app currently depends on the repository backend for auth, analysis,
  vision, and saved snapshots.
- It is a real mobile client, but it is not yet packaged as a production iOS or
  Android app.
- The next product phase is to harden and ship this as a real native-distributed
  application.
