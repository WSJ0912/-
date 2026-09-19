from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    DOCTOR = "doctor"


PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.ADMIN: frozenset({"manage_users", "manage_models", "manage_assistant", "import", "experiment", "assistant"}),
    Role.DOCTOR: frozenset({"import", "infer", "review", "report", "confirm_report", "experiment", "assistant"}),
}


def require_permission(role: str | Role, permission: str) -> None:
    try:
        normalized = Role(role)
    except ValueError as exc:
        raise PermissionError("unknown role") from exc
    if permission not in PERMISSIONS[normalized]:
        raise PermissionError(f"role {normalized.value!r} cannot perform {permission!r}")


def hash_password(password: str) -> str:
    """Argon2id only; there is deliberately no plaintext or weak fallback."""

    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    try:
        from argon2 import PasswordHasher
        from argon2.low_level import Type
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("argon2-cffi is required for account creation") from exc
    return PasswordHasher(type=Type.ID).hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("argon2-cffi is required for account verification") from exc
    try:
        return bool(PasswordHasher().verify(password_hash, password))
    except VerifyMismatchError:
        return False


@dataclass
class LocalTokenStore:
    """In-memory bearer tokens; a new service process gets a new token."""

    token: str = ""

    def __post_init__(self) -> None:
        self.rotate()

    def rotate(self) -> str:
        self.token = secrets.token_urlsafe(32)
        return self.token

    def validate(self, candidate: str | None) -> bool:
        return bool(candidate) and secrets.compare_digest(candidate, self.token)


@dataclass(frozen=True)
class UserSession:
    user_id: str
    role: Role


class SessionStore:
    """Short-lived process-local user sessions, invalidated on service restart."""

    def __init__(self) -> None:
        self._sessions: dict[str, UserSession] = {}

    def create(self, user_id: str, role: str | Role) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = UserSession(user_id=user_id, role=Role(role))
        return token

    def get(self, token: str | None) -> UserSession | None:
        if not token:
            return None
        return self._sessions.get(token)

    def revoke(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)
