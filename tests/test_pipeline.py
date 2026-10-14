import pytest
from sqlalchemy import select

from app.config import Settings
from app.db import DailyCursor, Game, Run, make_session_factory
from app.monitor import Monitor
from app.pipeline import AlreadyRunning, Pipeline

from .fakes import FakeLLM, FakeMetacritic

NEW = [f"new-{i:02d}" for i in range(20)]
PAGE1 = ["new-00", "new-01"] + [f"p1-{i:02d}" for i in range(22)]  # New Releases also show up in SEE ALL
PAGE2 = [f"p2-{i:02d}" for i in range(24)]
PAGE3 = [f"p3-{i:02d}" for i in range(24)]


@pytest.fixture
def pipeline(tmp_path):
    sf = make_session_factory(f"sqlite:///{tmp_path}/t.db")
    settings = Settings(_env_file=None, games_per_run=20, detail_workers=3, letsplay_enabled=False, scheduler_enabled=False)
    p = Pipeline(sf, FakeMetacritic(NEW, {1: PAGE1, 2: PAGE2, 3: PAGE3}), FakeLLM(), Monitor(), settings)
    return p


async def pick(p, day, run_id):
    return await p.select_games(day, run_id)


async def test_daily_walk_new_releases_then_see_all_pages(pipeline):
    day = "2026-10-14"
    picked, sources = await pick(pipeline, day, 1)
    assert picked == NEW and sources == ["New Releases: 20"]

    picked, _ = await pick(pipeline, day, 2)
    assert picked == [f"p1-{i:02d}" for i in range(20)]  # new-00/01 are already done today

    picked, sources = await pick(pipeline, day, 3)
    assert picked == ["p1-20", "p1-21"] + PAGE2[:18]
    assert sources == ["SEE ALL стр. 1: 2", "SEE ALL стр. 2: 18"]
    with pipeline.sf() as s:
        assert s.get(DailyCursor, day).next_page == 2

    picked, _ = await pick(pipeline, day, 4)
    assert picked == PAGE2[18:] + PAGE3[:14]


async def test_new_day_starts_over_from_new_releases(pipeline):
    await pick(pipeline, "2026-10-14", 1)
    await pick(pipeline, "2026-10-14", 2)
    picked, sources = await pick(pipeline, "2026-10-15", 3)
    assert picked == NEW and sources == ["New Releases: 20"]


async def test_full_run_upserts_and_reuses_summaries(pipeline):
    pipeline.today = lambda: "2026-10-14"
    run_id = pipeline.start("manual")
    with pytest.raises(AlreadyRunning):
        pipeline.start("manual")
    await pipeline.wait()

    with pipeline.sf() as s:
        run = s.get(Run, run_id)
        assert (run.status, run.games_total, run.games_ok, run.games_new) == ("done", 20, 20, 20)
        games = list(s.scalars(select(Game)))
        assert len(games) == 20
        g = games[0]
        assert g.critic_summary["verdict"].startswith("critic")
        assert g.user_summary["likes"] == ["gameplay"]
        assert [p.name for p in g.platforms] == ["PC", "PlayStation 5"]
        assert g.metascore is not None and g.userscore == 7.5
    assert pipeline.llm.summary_calls == 40  # critics + users for 20 games

    # the same games again (next day): same reviews, so no new LLM calls, games are updated not duplicated
    pipeline.today = lambda: "2026-10-15"
    pipeline.start("schedule")
    await pipeline.wait()
    assert pipeline.llm.summary_calls == 40
    with pipeline.sf() as s:
        assert len(list(s.scalars(select(Game)))) == 20
        assert s.get(Run, run_id + 1).games_new == 0


async def test_failed_game_does_not_stop_the_run(pipeline):
    pipeline.today = lambda: "2026-10-14"
    original = pipeline.mc.game_details

    async def flaky(slug):
        if slug == "new-03":
            raise RuntimeError("boom")
        return await original(slug)

    pipeline.mc.game_details = flaky
    run_id = pipeline.start("manual")
    await pipeline.wait()
    with pipeline.sf() as s:
        run = s.get(Run, run_id)
        assert (run.status, run.games_ok, run.games_failed) == ("done", 19, 1)
    assert any("new-03" in e["msg"] for e in pipeline.monitor.log)
