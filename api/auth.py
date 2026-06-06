"""JWT creation/verification, password hashing, and UTC time helper."""

import os
import secrets as _secrets
from datetime import datetime, timezone, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# ---------------------------------------------------------------------------
# Secret key — ephemeral random if env var not set (dev only).
# In production, JWT_SECRET_KEY *must* be set; tokens become invalid on restart
# if the key is ephemeral.
# ---------------------------------------------------------------------------
_env_key = os.getenv("JWT_SECRET_KEY", "")
if _env_key:
    SECRET_KEY = _env_key
    KEY_IS_EPHEMERAL = False
else:
    SECRET_KEY = _secrets.token_hex(32)
    KEY_IS_EPHEMERAL = True   # main.py prints a warning at startup

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def utcnow() -> datetime:
    """Naive UTC datetime, compatible with SQLite DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    """Embed a unique jti (JWT ID) for token revocation support."""
    payload = data.copy()
    payload["jti"] = _secrets.token_hex(16)
    payload["exp"] = utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
