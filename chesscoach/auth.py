"""Authentication helpers for the browser-based ChessCoach UI."""

from __future__ import annotations

import hashlib
import hmac
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


class AuthError(ValueError):
    """Raised when auth-related user input is invalid."""


class AuthStore:
    """SQLite-backed user storage for email/password authentication."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def initialize(self) -> None:
        """Create the users table if it does not exist yet."""
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
        return UserRecord(
            id=int(lastrowid),
            email=normalized_email,
            created_at=created_at,
        )

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
