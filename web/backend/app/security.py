"""Password hashing + JWT issuing/decoding."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd.verify(plain, hashed)
    except Exception:
        return False


def issue_token(subject: str) -> str:
    s = get_settings()
    exp = datetime.now(timezone.utc) + timedelta(hours=s.jwt_ttl_hours)
    payload = {"sub": subject, "exp": exp}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_alg)


def decode_token(token: str) -> str | None:
    s = get_settings()
    try:
        data = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_alg])
        return data.get("sub")
    except JWTError:
        return None
