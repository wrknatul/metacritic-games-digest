from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(500), index=True)
    cover_url: Mapped[str | None] = mapped_column(String(1000))
    developer: Mapped[str | None] = mapped_column(String(500))
    publisher: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    release_date: Mapped[str | None] = mapped_column(String(20))
    genres: Mapped[list] = mapped_column(JSON, default=list)
    franchise: Mapped[str | None] = mapped_column(String(500))
    video_url: Mapped[str | None] = mapped_column(String(1000))
    video_title: Mapped[str | None] = mapped_column(String(500))

    # Best score over platforms, used for sorting the list
    metascore: Mapped[int | None] = mapped_column(Integer, index=True)
    userscore: Mapped[float | None] = mapped_column(Float, index=True)
    critic_review_count: Mapped[int] = mapped_column(Integer, default=0)
    user_review_count: Mapped[int] = mapped_column(Integer, default=0)

    critic_summary: Mapped[dict | None] = mapped_column(JSON)
    user_summary: Mapped[dict | None] = mapped_column(JSON)
    # Fingerprints of the review samples the summaries were built from:
    # summaries are regenerated only when the sample changes
    critic_reviews_fp: Mapped[str | None] = mapped_column(String(64))
    user_reviews_fp: Mapped[str | None] = mapped_column(String(64))
    summaries_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    letsplay: Mapped[dict | None] = mapped_column(JSON)
    # pending | done | not_found | failed | blocked
    letsplay_status: Mapped[str | None] = mapped_column(String(20))
    letsplay_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    platforms: Mapped[list[GamePlatform]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
        order_by="GamePlatform.position",
        lazy="selectin",
    )


class GamePlatform(Base):
    __tablename__ = "game_platforms"
    __table_args__ = (UniqueConstraint("game_id", "slug"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    slug: Mapped[str] = mapped_column(String(100))
    position: Mapped[int] = mapped_column(Integer, default=0)
    is_lead: Mapped[bool] = mapped_column(Boolean, default=False)
    metascore: Mapped[int | None] = mapped_column(Integer)
    critic_reviews: Mapped[int] = mapped_column(Integer, default=0)
    userscore: Mapped[float | None] = mapped_column(Float)
    user_reviews: Mapped[int] = mapped_column(Integer, default=0)

    game: Mapped[Game] = relationship(back_populates="platforms")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trigger: Mapped[str] = mapped_column(String(20))  # schedule | manual
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | done | failed
    day: Mapped[str] = mapped_column(String(10), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sources: Mapped[list] = mapped_column(JSON, default=list)
    games_total: Mapped[int] = mapped_column(Integer, default=0)
    games_ok: Mapped[int] = mapped_column(Integer, default=0)
    games_failed: Mapped[int] = mapped_column(Integer, default=0)
    games_new: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


class DailyProcessed(Base):
    """Which games were already taken today: a game is processed at most once per day."""

    __tablename__ = "daily_processed"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)


class DailyCursor(Base):
    """Where today's walk stopped: New Releases first, then SEE ALL pages one by one."""

    __tablename__ = "daily_cursor"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    new_releases_done: Mapped[bool] = mapped_column(Boolean, default=False)
    next_page: Mapped[int] = mapped_column(Integer, default=1)


class LlmUsage(Base):
    __tablename__ = "llm_usage"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


def make_engine(url: str):
    if url.startswith("sqlite:///"):
        path = url.removeprefix("sqlite:///")
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

    return engine


def make_session_factory(url: str):
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
