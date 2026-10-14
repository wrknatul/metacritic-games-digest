import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.db import Game, GamePlatform


def add(s, slug, title, meta, user, platforms, genres, desc="", franchise=None):
    g = Game(slug=slug, title=title, metascore=meta, userscore=user, genres=genres, description=desc, franchise=franchise, release_date="2026-10-0" + str(len(slug) % 9 + 1))
    g.platforms = [GamePlatform(name=p, slug=p.lower(), position=i, metascore=meta) for i, p in enumerate(platforms)]
    g.critic_summary = {"verdict": "Критики довольны", "likes": ["сюжет"], "dislikes": ["цена"], "sample_size": 3, "total": 3, "counts": {"positive": 3, "neutral": 0, "negative": 0}, "model": "m", "generated_at": "2026-10-14T10:00:00"}
    s.add(g)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setattr(main.settings, "scheduler_enabled", False)
    with TestClient(main.app) as c:
        with c.app.state.sf.begin() as s:
            add(s, "gears-e-day", "Gears of War: E-Day", 88, 8.7, ["Xbox Series X", "PC"], ["Third Person Shooter"], "Cover shooter against the horde", "Gears of War")
            add(s, "gears-5", "Gears 5", 84, 6.1, ["Xbox Series X", "PC"], ["Third Person Shooter"], "Cover shooter campaign", "Gears of War")
            add(s, "planet-zoo-2", "Planet Zoo 2", 85, None, ["PC"], ["Tycoon"], "Build a zoo")
            add(s, "boltgun-2", "Warhammer 40,000: Boltgun 2", 81, 9.0, ["PC", "PlayStation 5"], ["FPS"], "Retro shooter")
            add(s, "no-score", "Unscored Indie", None, None, ["Nintendo Switch 2"], ["Puzzle"])
        c.app.state.pipeline.data_version += 1
        yield c


def slugs(resp):
    return [g["slug"] for g in resp.json()["items"]]


def test_sort_by_metascore_puts_unscored_last(client):
    assert slugs(client.get("/api/games")) == ["gears-e-day", "planet-zoo-2", "gears-5", "boltgun-2", "no-score"]


def test_sort_by_userscore(client):
    assert slugs(client.get("/api/games?sort=userscore"))[:3] == ["boltgun-2", "gears-e-day", "gears-5"]


def test_filter_by_platform_and_search(client):
    assert slugs(client.get("/api/games", params={"platform": "PlayStation 5"})) == ["boltgun-2"]
    assert set(slugs(client.get("/api/games", params={"q": "gears"}))) == {"gears-e-day", "gears-5"}
    assert slugs(client.get("/api/games", params={"q": "100%_"})) == []  # LIKE wildcards are escaped


def test_bad_sort_is_a_structured_error(client):
    r = client.get("/api/games?sort=drop")
    assert r.status_code == 422
    body = r.json()
    assert body["success"] is False and body["error_code"] == "BAD_SORT" and body["trace_id"]


def test_game_card_shows_similar_games(client):
    item = client.get("/api/games/gears-e-day").json()["item"]
    assert item["similar"][0]["slug"] == "gears-5"
    page = client.get("/games/gears-e-day")
    assert page.status_code == 200
    assert 'href="/games/gears-5"' in page.text
    assert "Критики довольны" in page.text


def test_pages_render(client):
    r = client.get("/", params={"platform": "PC", "sort": "userscore", "q": "o"})
    assert r.status_code == 200 and "Planet Zoo 2" in r.text
    assert client.get("/monitor").status_code == 200
    assert client.get("/games/missing").status_code == 404
    assert client.get("/api/games/missing").json()["error_code"] == "HTTP_ERROR"


def test_manual_run_conflict(client, monkeypatch):
    calls = []

    def fake_start(trigger):
        calls.append(trigger)
        if len(calls) > 1:
            from app.pipeline import AlreadyRunning

            raise AlreadyRunning(7)
        return 7

    monkeypatch.setattr(client.app.state.pipeline, "start", fake_start)
    assert client.post("/api/runs").json() == {"success": True, "run_id": 7}
    r = client.post("/api/runs")
    assert r.status_code == 409 and r.json()["error_code"] == "RUN_IN_PROGRESS"
