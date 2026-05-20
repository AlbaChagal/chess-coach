"""Authentication helpers for the browser-based ChessCoach UI."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from itsdangerous import BadSignature, URLSafeSerializer

DEFAULT_AUTH_DB = Path("data/chesscoach_auth.db")
DEFAULT_SESSION_COOKIE = "chesscoach_session"
PBKDF2_ITERATIONS = 100_000


@dataclass(frozen=True)
class UserRecord:
    """Stored user record returned by the auth store."""

    id: int
    email: str
    created_at: str


@dataclass(frozen=True)
class UserSettingsRecord:
    """Stored user settings returned by the auth store."""

    user_id: int
    show_coordinates: bool


@dataclass(frozen=True)
class SavedSnapshotSummary:
    """Saved snapshot metadata for list rendering."""

    id: int
    user_id: int
    created_at: str
    updated_at: str
    fen: str
    side_to_move: str
    best_move_san: str | None
    best_move_score_display: str | None
    has_explanation: bool
    has_coaching: bool


@dataclass(frozen=True)
class SavedSnapshotRecord:
    """Full saved snapshot record scoped to a user."""

    id: int
    user_id: int
    created_at: str
    updated_at: str
    snapshot: dict[str, object]
    summary: SavedSnapshotSummary


class AuthError(ValueError):
    """Raised when auth-related user input is invalid."""


class AuthStore:
    """SQLite-backed user storage for email/password authentication."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def initialize(self) -> None:
        """Create the auth and account-data tables if they do not exist yet."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    show_coordinates INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    fen TEXT NOT NULL,
                    side_to_move TEXT NOT NULL,
                    best_move_san TEXT,
                    best_move_score_display TEXT,
                    has_explanation INTEGER NOT NULL,
                    has_coaching INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )

    def create_user(self, email: str, password: str) -> UserRecord:
        """Create a new user after validating the provided credentials."""
        normalized_email = _normalize_email(email)
        _validate_password(password)
        password_hash = hash_password(password)
        created_at = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, password_hash, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (normalized_email, password_hash, created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise AuthError("An account with this email already exists.") from exc
        lastrowid = cursor.lastrowid
        if lastrowid is None:
            raise RuntimeError("User insert did not return a row id.")
        user = UserRecord(
            id=int(lastrowid),
            email=normalized_email,
            created_at=created_at,
        )
        self.update_user_settings(user.id, show_coordinates=True)
        return user

    def authenticate_user(self, email: str, password: str) -> UserRecord:
        """Return the authenticated user or raise an auth error."""
        normalized_email = _normalize_email(email)
        row = self._fetch_user_row_by_email(normalized_email)
        if row is None or not verify_password(password, row["password_hash"]):
            raise AuthError("Invalid email or password.")
        return _user_record_from_row(row)


    def get_user_by_id(self, user_id: int) -> UserRecord | None:
        """Look up a user by primary key."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, email, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return _user_record_from_row(row)

    def get_user_settings(self, user_id: int) -> UserSettingsRecord:
        """Return synced settings for the given user, creating defaults if needed."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, show_coordinates
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return self.update_user_settings(user_id, show_coordinates=True)
        return _settings_record_from_row(row)

    def update_user_settings(
        self,
        user_id: int,
        *,
        show_coordinates: bool,
    ) -> UserSettingsRecord:
        """Persist synced settings for the given user."""
        updated_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_settings (user_id, show_coordinates, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    show_coordinates = excluded.show_coordinates,
                    updated_at = excluded.updated_at
                """,
                (user_id, int(show_coordinates), updated_at),
            )
        return UserSettingsRecord(
            user_id=user_id,
            show_coordinates=show_coordinates,
        )

    def create_saved_snapshot(
        self,
        user_id: int,
        *,
        snapshot: dict[str, object],
    ) -> SavedSnapshotRecord:
        """Create a saved snapshot for the given user."""
        summary = _saved_snapshot_summary_from_payload(
            user_id=user_id,
            snapshot_id=0,
            created_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
            snapshot=snapshot,
        )
        snapshot_json = json.dumps(snapshot)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO saved_snapshots (
                    user_id,
                    fen,
                    side_to_move,
                    best_move_san,
                    best_move_score_display,
                    has_explanation,
                    has_coaching,
                    snapshot_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    summary.fen,
                    summary.side_to_move,
                    summary.best_move_san,
                    summary.best_move_score_display,
                    int(summary.has_explanation),
                    int(summary.has_coaching),
                    snapshot_json,
                    summary.created_at,
                    summary.updated_at,
                ),
            )
        snapshot_id = cursor.lastrowid
        if snapshot_id is None:
            raise RuntimeError("Saved snapshot insert did not return a row id.")
        return SavedSnapshotRecord(
            id=int(snapshot_id),
            user_id=user_id,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            snapshot=snapshot,
            summary=SavedSnapshotSummary(
                id=int(snapshot_id),
                user_id=user_id,
                created_at=summary.created_at,
                updated_at=summary.updated_at,
                fen=summary.fen,
                side_to_move=summary.side_to_move,
                best_move_san=summary.best_move_san,
                best_move_score_display=summary.best_move_score_display,
                has_explanation=summary.has_explanation,
                has_coaching=summary.has_coaching,
            ),
        )

    def list_saved_snapshots(self, user_id: int) -> list[SavedSnapshotSummary]:
        """Return saved snapshot summaries for the given user."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    created_at,
                    updated_at,
                    fen,
                    side_to_move,
                    best_move_san,
                    best_move_score_display,
                    has_explanation,
                    has_coaching
                FROM saved_snapshots
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return [_saved_snapshot_summary_from_row(row) for row in rows]

    def get_saved_snapshot(
        self,
        user_id: int,
        snapshot_id: int,
    ) -> SavedSnapshotRecord | None:
        """Return a saved snapshot if it belongs to the given user."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    created_at,
                    updated_at,
                    fen,
                    side_to_move,
                    best_move_san,
                    best_move_score_display,
                    has_explanation,
                    has_coaching,
                    snapshot_json
                FROM saved_snapshots
                WHERE user_id = ? AND id = ?
                """,
                (user_id, snapshot_id),
            ).fetchone()
        if row is None:
            return None
        snapshot = json.loads(str(row["snapshot_json"]))
        return SavedSnapshotRecord(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            snapshot=snapshot,
            summary=_saved_snapshot_summary_from_row(row),
        )

    def delete_saved_snapshot(self, user_id: int, snapshot_id: int) -> bool:
        """Delete a saved snapshot if it belongs to the given user."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM saved_snapshots
                WHERE user_id = ? AND id = ?
                """,
                (user_id, snapshot_id),
            )
        return cursor.rowcount > 0

    def _fetch_user_row_by_email(self, email: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
                (email,),
            ).fetchone()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection


def create_auth_store() -> AuthStore:
    """Create the auth store from environment or default config."""
    configured_path = os.getenv("CHESSCOACH_AUTH_DB")
    db_path = Path(configured_path) if configured_path else DEFAULT_AUTH_DB
    return AuthStore(db_path=db_path)


def create_session_serializer() -> URLSafeSerializer:
    """Create the cookie serializer used for browser auth sessions."""
    secret = os.getenv("CHESSCOACH_SESSION_SECRET", "dev-session-secret")
    return URLSafeSerializer(secret_key=secret, salt="chesscoach-session")


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    ).hex()
    return f"{PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a stored PBKDF2 hash."""
    try:
        iterations_text, salt, expected_digest = password_hash.split("$", maxsplit=2)
        iterations = int(iterations_text)
    except ValueError:
        return False
    actual_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(actual_digest, expected_digest)


def encode_session_cookie(user: UserRecord) -> str:
    """Encode the authenticated user into a signed cookie payload."""
    serializer = create_session_serializer()
    return serializer.dumps({"user_id": user.id})


def decode_session_cookie(cookie_value: str) -> int | None:
    """Decode a signed session cookie into a user id."""
    serializer = create_session_serializer()
    try:
        payload = serializer.loads(cookie_value)
    except BadSignature:
        return None
    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        return None
    return user_id


def _normalize_email(email: str) -> str:
    value = email.strip().lower()
    if not value or "@" not in value or "." not in value.split("@")[-1]:
        raise AuthError("Enter a valid email address.")
    return value


def _validate_password(password: str) -> None:
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters long.")


def _user_record_from_row(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        id=int(row["id"]),
        email=str(row["email"]),
        created_at=str(row["created_at"]),
    )


def _settings_record_from_row(row: sqlite3.Row) -> UserSettingsRecord:
    return UserSettingsRecord(
        user_id=int(row["user_id"]),
        show_coordinates=bool(row["show_coordinates"]),
    )


def _saved_snapshot_summary_from_row(row: sqlite3.Row) -> SavedSnapshotSummary:
    return SavedSnapshotSummary(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        fen=str(row["fen"]),
        side_to_move=str(row["side_to_move"]),
        best_move_san=(
            None if row["best_move_san"] is None else str(row["best_move_san"])
        ),
        best_move_score_display=(
            None
            if row["best_move_score_display"] is None
            else str(row["best_move_score_display"])
        ),
        has_explanation=bool(row["has_explanation"]),
        has_coaching=bool(row["has_coaching"]),
    )


def _saved_snapshot_summary_from_payload(
    *,
    user_id: int,
    snapshot_id: int,
    created_at: str,
    updated_at: str,
    snapshot: dict[str, object],
) -> SavedSnapshotSummary:
    position = snapshot.get("position")
    analysis = snapshot.get("analysis")
    explanation = snapshot.get("explanation")
    if not isinstance(position, dict) or not isinstance(analysis, dict):
        raise AuthError("Saved snapshot is missing required analysis data.")
    fen = position.get("fen")
    side_to_move = position.get("side_to_move")
    if not isinstance(fen, str) or not isinstance(side_to_move, str):
        raise AuthError("Saved snapshot is missing required position fields.")

    top_moves = analysis.get("top_moves")
    best_move_san: str | None = None
    best_move_score_display: str | None = None
    if isinstance(top_moves, list) and top_moves:
        first_move = top_moves[0]
        if isinstance(first_move, dict):
            san_value = first_move.get("move_san")
            score_value = first_move.get("score_display")
            if isinstance(san_value, str):
                best_move_san = san_value
            if isinstance(score_value, str):
                best_move_score_display = score_value

    has_explanation = False
    has_coaching = False
    if isinstance(explanation, dict):
        has_explanation = explanation.get("structured_explanation") is not None
        has_coaching = explanation.get("played_move_result") is not None

    return SavedSnapshotSummary(
        id=snapshot_id,
        user_id=user_id,
        created_at=created_at,
        updated_at=updated_at,
        fen=fen,
        side_to_move=side_to_move,
        best_move_san=best_move_san,
        best_move_score_display=best_move_score_display,
        has_explanation=has_explanation,
        has_coaching=has_coaching,
    )
