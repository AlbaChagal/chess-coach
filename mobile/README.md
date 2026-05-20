# ChessCoach Mobile

Expo React Native implementation of the ChessCoach phone app. This project is
additive and does not replace the existing FastAPI/browser app.

## Run Locally

Start the backend from the repository root:

```bash
uv run python -m chesscoach.cli serve --host 0.0.0.0 --port 8000
```

Install mobile dependencies and run Expo from this directory:

```bash
npm install
npm run start
```

For testing on a real phone, set the API URL to your Mac LAN address:

```bash
EXPO_PUBLIC_CHESSCOACH_API_URL=http://192.168.1.20:8000 npm run start
```

Do not use `127.0.0.1` from a physical phone; that points to the phone itself.

## Scope

The app implements the native phone direction for:

- email/password auth,
- four tabs: Analyze, Saved, Profile, Settings,
- staged analysis flow: Capture, Detect, Setup, Ready, Analyze,
- explanation detail screens,
- saved snapshots,
- local backend integration.

The browser UI remains in `chesscoach/templates` and `chesscoach/static`.
