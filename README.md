# ChessCoach

ChessCoach analyzes a real chess position from a photo, reconstructs the
board, runs engine analysis, and can generate coaching explanations for the
best line or for a move the user played.

This repository currently contains:

- a Python backend and browser app powered by FastAPI,
- an Expo React Native mobile app for iPhone/Android development,
- the vision, analysis, and explanation pipeline,
- local auth, synced settings, and saved analysis snapshots,
- training, evaluation, and benchmarking scripts for the vision stack.

## Current Status

Implemented today:

- browser app with login, signup, analyze, saved, and profile flows,
- Expo mobile app with the same account-backed product flow,
- staged analysis flow:
  - Capture
  - Detect
  - Setup
  - Ready
  - Analyze
- board detection with manual corner correction,
- precision zoom for corner placement,
- orientation selection from the White king start square,
- side-to-move selection,
- conservative FEN completion,
- ready-stage piece correction on a digital board,
- interactive analysis board,
- suggested-line playback,
- branching analysis for non-suggested legal moves,
- `Previous`, `Reset`, and line selection,
- per-line and played-move explanation requests,
- synced settings,
- saved analysis snapshots,
- local SQLite auth/session storage,
- CLI entry points for FEN analysis, image analysis, and serving the app,
- test coverage across backend, pipeline, browser UI, and mobile JS helpers.

Current known limitations:

- the mobile app is still an Expo app, not yet packaged and shipped as a store
  app,
- saved snapshots do not yet persist the uploaded source image,
- castling rights are inferred conservatively from visible start-square pieces,
- en passant is still assumed to be `-`,
- real-world vision robustness still needs continued hardening,
- explanation quality still needs broader product evaluation and benchmarking.

## Repository Layout

- `chesscoach/vision/`: board localization, detection, assignment, FEN
  placement prediction
- `chesscoach/analysis/`: Stockfish integration and top-line analysis
- `chesscoach/explanation/`: structured explanation and optional LLM narration
- `chesscoach/pipeline.py`: typed orchestration across vision, analysis, and
  explanation
- `chesscoach/server.py`: FastAPI backend plus browser routes
- `chesscoach/auth.py`: auth, settings, and saved snapshots
- `chesscoach/static/` and `chesscoach/templates/`: browser UI
- `mobile/`: Expo React Native app
- `scripts/`: training, evaluation, debugging, and benchmark helpers
- `tests/`: backend, pipeline, UI, and vision test suites
- `models/`: local trained checkpoints used by the vision stack

## Prerequisites

Python side:

- Python `>= 3.11`
- `uv`
- Stockfish available on `PATH` as `stockfish`

Mobile side:

- Node.js and npm
- Expo Go for testing on a real phone

## Backend Setup

Install Python dependencies from the repository root:

```bash
uv sync
```

Run the backend locally:

```bash
uv run python -m chesscoach.cli serve --host 127.0.0.1 --port 8000
```

Open the browser app:

```text
http://127.0.0.1:8000
```

Run the backend so a phone on your local network can reach it:

```bash
uv run python -m chesscoach.cli serve --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Mobile App Setup

Install mobile dependencies from the `mobile/` directory:

```bash
cd mobile
npm install
```

Start Expo for local development:

```bash
npm run start
```

Run against a backend on your Mac from a real phone:

```bash
EXPO_PUBLIC_CHESSCOACH_API_URL=http://YOUR_MAC_LAN_IP:8000 npm run start
```

Important:

- do not use `127.0.0.1` from a physical phone,
- the phone and Mac must be on the same Wi‑Fi,
- the backend must be running with `--host 0.0.0.0`.

## How To Use The Product

### Browser App

1. Start the backend with `serve`.
2. Open the app in your browser.
3. Create an account or log in.
4. Upload or capture a board image.
5. Adjust board corners if needed.
6. Mark the White king start square.
7. Select side to move.
8. Review and correct detected pieces.
9. Run analysis.
10. Step through suggested lines, branch with legal moves, request
    explanations, and save snapshots.

### Mobile App

1. Start the backend.
2. Start Expo from `mobile/`.
3. Open the app in Expo Go.
4. Log in or sign up.
5. Use the staged flow:
   - Capture
   - Detect
   - Setup
   - Ready
   - Analyze
6. Save positions or revisit them from the Saved tab.

## CLI Usage

Analyze a FEN:

```bash
uv run python -m chesscoach.cli fen \
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

Run the end-to-end image pipeline:

```bash
uv run python -m chesscoach.cli image path/to/board.jpg \
  --side-to-move w \
  --white-king-start-click-x 123 \
  --white-king-start-click-y 456
```

Emit machine-readable JSON:

```bash
uv run python -m chesscoach.cli image path/to/board.jpg \
  --side-to-move w \
  --white-king-start-click-x 123 \
  --white-king-start-click-y 456 \
  --json
```

Serve the backend:

```bash
uv run python -m chesscoach.cli serve --host 127.0.0.1 --port 8000
```

Run vision-only FEN prediction:

```bash
uv run python -m chesscoach.vision_cli path/to/board.jpg
```

## HTTP Surface

Core staged endpoints:

- `POST /detect-board`
- `POST /vision`
- `POST /complete-position`
- `POST /analyze`
- `POST /legal-moves`
- `POST /play-move`
- `POST /explain`
- `POST /coach`

Auth and account endpoints:

- `POST /auth/signup`
- `POST /auth/login`
- `GET /auth/me`
- `POST /auth/logout`
- `GET /api/settings`
- `POST /api/settings`
- `GET /api/saved`
- `POST /api/saved`
- `GET /api/saved/{snapshot_id}`
- `DELETE /api/saved/{snapshot_id}`

Browser routes:

- `/login`
- `/signup`
- `/app/analyze`
- `/app/saved`
- `/app/profile`

## Environment Variables

- `CHESSCOACH_AUTH_DB`: override the SQLite auth/settings/snapshot database
  path
- `CHESSCOACH_SESSION_SECRET`: override the session signing secret
- `ANTHROPIC_API_KEY`: enable Anthropic explanation narration
- `OPENAI_API_KEY`: enable OpenAI explanation narration
- `EXPO_PUBLIC_CHESSCOACH_API_URL`: point the Expo app at a reachable backend

If no explanation provider is configured, analysis remains usable and the
backend can still return structured explanation payloads where supported.

## Vision Models

Default checkpoints:

- `models/piece_detector.pt`
- `models/board_localizer.pt`

Additional checkpoints and evaluation artifacts also live under `models/`,
`results/`, and `debug/`.

## Testing

Run the backend and Python test suite:

```bash
uv run pytest
```

Current verified baseline:

```text
397 passed
```

Run the mobile JS test suite:

```bash
cd mobile
npm test
```

The repository currently includes:

- unit tests for analysis, explanation, pipeline, and vision,
- FastAPI route tests,
- Playwright browser flow tests,
- mobile JS helper and state tests.

## Development Notes

- use `uv` for Python package management and tooling,
- use `npm` only inside `mobile/`,
- keep the product conservative about game state the image cannot prove,
- prefer regression tests for user-facing bugs,
- keep analysis usable even when narration is unavailable.

## Planning Docs

- [PLAN.md](/Users/shaharheyman/PycharmProjects/chesscoach/PLAN.md):
  native app and product roadmap
- [UI_PLAN.md](/Users/shaharheyman/PycharmProjects/chesscoach/UI_PLAN.md):
  next-phase native app UI plan
- [UX_PLAN.md](/Users/shaharheyman/PycharmProjects/chesscoach/UX_PLAN.md):
  next-phase native product UX rules
