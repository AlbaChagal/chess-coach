# ChessCoach

ChessCoach is a mobile-first chess analysis prototype. It takes a photo of a
real chessboard, reconstructs the position, runs engine analysis, and can
generate coaching explanations for the engine's recommended lines.

The current product is a browser-based prototype backed by FastAPI. It is
designed as the proving ground for a later native iPhone/Android app.

## Current Product State

Implemented:

- image upload and camera capture in the browser UI,
- board detection with user-adjustable board corners,
- orientation by asking where the white king started the game,
- side-to-move selection,
- FEN completion from detected piece placement plus user inputs,
- ready-stage correction of detected pieces directly on the digital board,
- Stockfish-backed top-3 engine analysis,
- interactive analysis board with arrows and legal-move interaction,
- suggested move clicks behaving like line playback,
- non-suggested legal moves branching into a fresh analysis,
- `Previous` and `Reset` across analysis branches back to the original
  detected or corrected position,
- per-line on-demand explanations,
- email/password authentication,
- synced coordinate-display setting,
- saved analysis snapshots scoped to the user account,
- saved-position and profile pages.

Known limitations:

- this is not a native mobile app yet,
- saved snapshots do not yet persist the uploaded source image,
- castling rights are inferred from visible start-square pieces,
- en passant is currently assumed to be `-`,
- real-world vision robustness still needs product hardening,
- explanation quality should continue to be evaluated against curated positions.

## Architecture

Main runtime layers:

- `chesscoach/vision/`: board localization, board post-processing, piece
  detection, square assignment, and FEN placement prediction.
- `chesscoach/analysis/`: Stockfish integration and candidate-line formatting.
- `chesscoach/explanation/`: move classification, tactic detection,
  position-level synthesis, prompt building, and optional LLM narration.
- `chesscoach/pipeline.py`: typed orchestration from image to position,
  analysis, and optional explanation.
- `chesscoach/server.py`: FastAPI backend and browser UI routes.
- `chesscoach/auth.py`: local email/password auth, synced settings, and saved
  snapshots.
- `chesscoach/static/app.js`: browser app state machine and UI behavior.
- `chesscoach/templates/app_shell.html`: authenticated app shell.
- `chesscoach/static/app.css`: mobile-first UI styling.

Primary planning docs:

- `PLAN.md`: full product integration status and roadmap.
- `UI_PLAN.md`: UI/product plan for the browser and future mobile experience.
- `chesscoach/explanation/PLAN.md`: explanation-specific architecture and
  roadmap.

## Running Locally

Use `uv` for all Python tooling.

Run the browser app:

```bash
uv run python -m chesscoach.cli serve --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Create an account through the signup page. Local auth data defaults to:

```text
data/chesscoach_auth.db
```

Useful environment variables:

- `CHESSCOACH_AUTH_DB`: override the auth/settings/saved-snapshot database path.
- `CHESSCOACH_SESSION_SECRET`: override the development session secret.
- `ANTHROPIC_API_KEY`: enable Anthropic explanation narration.
- `OPENAI_API_KEY`: enable OpenAI explanation narration.

If no explanation provider is clearly configured, analysis still works and the
UI can warn that narration was skipped.

## CLI Usage

Analyze a FEN:

```bash
uv run python -m chesscoach.cli fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

Run the image pipeline:

```bash
uv run python -m chesscoach.cli image path/to/board.jpg \
  --side-to-move w \
  --white-king-start-click-x 123 \
  --white-king-start-click-y 456
```

Emit JSON for backend/mobile integration testing:

```bash
uv run python -m chesscoach.cli image path/to/board.jpg \
  --side-to-move w \
  --white-king-start-click-x 123 \
  --white-king-start-click-y 456 \
  --json
```

## HTTP/API Surface

The browser app uses staged endpoints that map to the mobile flow:

- `POST /detect-board`: detect board corners and confidence.
- `POST /vision`: predict piece placement from image, corners, and orientation
  click.
- `POST /complete-position`: build full FEN metadata from placement and user
  inputs.
- `POST /analyze`: run engine analysis.
- `POST /legal-moves`: return legal moves for the current FEN.
- `POST /play-move`: validate and apply a user move.
- `POST /explain`: request explanation for a selected line or played move.
- `POST /coach`: one-shot image-to-coaching API.
- `GET/POST /api/settings`: synced user settings.
- `GET/POST/DELETE /api/saved`: saved analysis snapshots.

Authenticated browser routes:

- `/app/analyze`
- `/app/saved`
- `/app/profile`

## Vision Models

The default model paths are:

- `models/piece_detector.pt`
- `models/board_localizer.pt`

The vision pipeline can fall back where supported, but real product testing
should use the trained checkpoints.

Current best documented vision baseline is tracked in `PLAN.md`.

## Testing

Run the full test suite:

```bash
uv run pytest
```

Current verified baseline:

```text
378 passed
```

The suite includes:

- unit tests for analysis, pipeline, explanation, and vision,
- FastAPI route tests,
- browser UI asset tests,
- Playwright end-to-end tests for the mobile-first UI flow.

For UI regressions, add Playwright tests in `tests/test_ui_e2e.py` that exercise
the actual browser behavior. Do not rely only on static asset assertions for
interactive bugs.

## Development Notes

- Package management is `uv` only.
- Use `uv run pytest` for tests.
- Keep the app useful even when explanation narration is unavailable.
- Do not let vision pretend to infer impossible state from a single image:
  side to move, castling, and en passant must stay explicit or conservative.
- User-reported UI bugs should usually be fixed with browser regression tests.

## Next Priorities

Near-term work should focus on:

- improving ready-stage correction UX on real phones,
- improving board detection confidence and failure guidance,
- persisting source images with saved snapshots,
- making the staged API cleaner for future native mobile clients,
- expanding explanation benchmarks with curated real positions,
- continuing vision robustness work after the product loop remains stable.
