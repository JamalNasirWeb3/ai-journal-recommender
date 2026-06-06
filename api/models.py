"""SQLAlchemy ORM models."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base


def _utcnow() -> datetime:
    """Naive UTC datetime (SQLite-compatible)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role            = Column(String, default="client")   # "client" | "internal"
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=_utcnow)

    searches = relationship("Search", back_populates="user", cascade="all, delete-orphan")
    usage    = relationship("Usage",  back_populates="user", cascade="all, delete-orphan")


class Search(Base):
    __tablename__ = "searches"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    title        = Column(String, nullable=False)
    area         = Column(String)
    abstract     = Column(Text)
    results_json = Column(Text)
    created_at   = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="searches")


class Usage(Base):
    __tablename__ = "usage"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    month       = Column(String, nullable=False)   # "YYYY-MM"
    query_count = Column(Integer, default=0)

    user = relationship("User", back_populates="usage")


class TokenBlacklist(Base):
    """Revoked JWT IDs. Checked on every authenticated request."""
    __tablename__ = "token_blacklist"

    id         = Column(Integer, primary_key=True, index=True)
    jti        = Column(String, unique=True, nullable=False, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    revoked_at = Column(DateTime, default=_utcnow)
    expires_at = Column(DateTime, nullable=False)   # for periodic cleanup


class LoginAttempt(Base):
    """Per-email failed login tracking for brute-force protection."""
    __tablename__ = "login_attempts"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, nullable=False, index=True)
    attempt_count = Column(Integer, default=0)
    locked_until  = Column(DateTime, nullable=True)
    last_attempt  = Column(DateTime, default=_utcnow)
