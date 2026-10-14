import pytest
from sqlalchemy import select

from app import letsplay as lp
from app.config import Settings
from app.db import Game, make_session_factory
from app.monitor import Monitor
from app.pipeline import Pipeline

from .fakes import FakeLLM, FakeMetacritic

TEXT = "so this is the game and honestly the combat feels great but the camera is annoying " * 5


@pytest.fixture
def pipeline(tmp_path):
    sf = make_session_factory(f"sqlite:///{tmp_path}/t.db")
    settings = Settings(_env_file=None, games_per_run=3, detail_workers=2, letsplay_enabled=True, scheduler_enabled=False, letsplay_strategies="captions,audio,gemini_video")
    return Pipeline(sf, FakeMetacritic(["aa-game", "bb-game", "cc-game"], {}), FakeLLM(), Monitor(), settings)


async def run_and_collect(pipeline):
    if not hasattr(pipeline, "_fixed_day"):
        pipeline._fixed_day = True
        pipeline.today = lambda: "2026-10-14"
    pipeline.start("manual")
    await pipeline.wait()
    with pipeline.sf() as s:
        return {g.slug: g for g in s.scalars(select(Game))}


async def test_new_games_are_queued_and_processed(pipeline, monkeypatch):
    async def search(title):
        if title.startswith("Cc"):
            return []
        return [lp.VideoCandidate("v" + title[:2], f"{title} Let's Play Part 1", "Chan", 1000, 1800)]

    async def captions(video, minutes):
        return TEXT if video.id == "vAa" else None

    async def no_audio(video, minutes):
        raise lp.YouTubeBlocked("HTTP Error 403")

    monkeypatch.setattr(lp, "search", search)
    monkeypatch.setattr(lp, "fetch_captions", captions)
    monkeypatch.setattr(lp, "fetch_audio", no_audio)

    async def no_video(url, minutes):
        from app.llm import LLMError

        raise LLMError("video input failed")

    pipeline.llm.transcribe_youtube = no_video

    games = await run_and_collect(pipeline)
    assert {g.letsplay_status for g in games.values()} == {"pending"}

    # popular games go first; every pending game is claimed once
    claimed = []
    while (gid := pipeline.claim_letsplay()) is not None:
        claimed.append(gid)
        await pipeline.run_letsplay(gid)
        pipeline._lp_inflight.discard(gid)
    assert len(claimed) == 3

    with pipeline.sf() as s:
        by_slug = {g.slug: g for g in s.scalars(select(Game))}
    aa, bb, cc = by_slug["aa-game"], by_slug["bb-game"], by_slug["cc-game"]
    assert aa.letsplay_status == "done"
    assert aa.letsplay["transcript_source"] == "captions" and aa.letsplay["conclusion"] == "ok"
    assert bb.letsplay_status == "failed"  # audio blocked, captions empty, Gemini error
    assert bb.letsplay.get("video_id") == "vBb", bb.letsplay
    assert len(bb.letsplay["errors"]) == 3
    assert cc.letsplay_status == "not_found"

    # a re-visit next day: done stays done, not_found waits for the retry period
    pipeline.today = lambda: "2026-10-16"
    games = await run_and_collect(pipeline)
    assert games["aa-game"].letsplay_status == "done"
    assert games["cc-game"].letsplay_status == "not_found"
    assert games["bb-game"].letsplay_status == "failed"


async def test_blocked_when_every_strategy_is_refused(pipeline, monkeypatch):
    async def search(title):
        return [lp.VideoCandidate("v1", f"{title} gameplay walkthrough", "Chan", 5, 900)]

    async def blocked(video, minutes):
        raise lp.YouTubeBlocked("HTTP Error 429")

    monkeypatch.setattr(lp, "search", search)
    monkeypatch.setattr(lp, "fetch_captions", blocked)
    monkeypatch.setattr(lp, "fetch_audio", blocked)

    async def video_blocked(url, minutes):
        raise lp.YouTubeBlocked("refused")

    pipeline.llm.transcribe_youtube = video_blocked
    await run_and_collect(pipeline)
    gid = pipeline.claim_letsplay()
    assert await pipeline.run_letsplay(gid) == "blocked"


async def test_llm_out_of_balance_keeps_the_game_pending(pipeline, monkeypatch):
    from app.llm import LLMUnavailable

    async def search(title):
        return [lp.VideoCandidate("v1", f"{title} let's play part 1", "Chan", 5, 1800)]

    async def captions(video, minutes):
        return TEXT

    async def no_balance(*a):
        raise LLMUnavailable("LLM HTTP 402: balance has run out")

    monkeypatch.setattr(lp, "search", search)
    monkeypatch.setattr(lp, "fetch_captions", captions)
    pipeline.llm.letsplay_conclusion = no_balance
    await run_and_collect(pipeline)
    gid = pipeline.claim_letsplay()
    assert await pipeline.run_letsplay(gid) == "llm_unavailable"
    with pipeline.sf() as s:
        assert s.get(Game, gid).letsplay_status == "pending"


async def test_network_error_on_search_keeps_the_game_pending(pipeline, monkeypatch):
    async def offline(title):
        raise lp.YouTubeUnreachable("Unable to download API page: [Errno 61] Connection refused")

    monkeypatch.setattr(lp, "search", offline)
    await run_and_collect(pipeline)
    gid = pipeline.claim_letsplay()
    assert await pipeline.run_letsplay(gid) == "youtube_unreachable"
    with pipeline.sf() as s:
        assert s.get(Game, gid).letsplay_status == "pending"
