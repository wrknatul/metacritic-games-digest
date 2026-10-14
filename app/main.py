from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select, update
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .db import Game, GamePlatform, Run, make_session_factory
from .llm import LLMClient
from .metacritic import MetacriticClient
from .monitor import Monitor
from .pipeline import AlreadyRunning, Pipeline
from .similar import GameFeatures, SimilarityCache

log = logging.getLogger("app")
BASE_DIR = Path(__file__).parent
PER_PAGE = 30
SORTS = {
    "metascore": "Metascore",
    "userscore": "Userscore",
    "new": "Дата выхода",
    "title": "Название",
}

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _score_class(score, scale: int = 100) -> str:
    if score is None:
        return "tbd"
    value = score if scale == 100 else score * 10
    return "good" if value >= 75 else "mixed" if value >= 50 else "bad"


templates.env.globals["score_class"] = _score_class
templates.env.filters["views"] = lambda n: f"{int(n or 0):,}".replace(",", " ")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        self.status, self.code, self.message, self.details = status, code, message, details or {}


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error_code": code, "message": message, "trace_id": str(uuid.uuid4()), "details": details or {}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sf = make_session_factory(settings.database_url)
    with sf.begin() as s:
        s.execute(
            update(Run)
            .where(Run.status == "running")
            .values(status="failed", error="прерван перезапуском сервиса", finished_at=datetime.now(timezone.utc))
        )
    monitor = Monitor()
    mc = MetacriticClient()
    llm = LLMClient(settings.llm_api_key, settings.llm_model, base_url=settings.llm_base_url)
    pipeline = Pipeline(sf, mc, llm, monitor, settings)
    llm.set_hooks(pipeline.check_budget, pipeline.record_usage)

    scheduler = AsyncIOScheduler(timezone=pipeline.tz)

    async def scheduled_run():
        try:
            pipeline.start("schedule")
        except AlreadyRunning as exc:
            monitor.event("warn", f"Плановый запуск пропущен: идёт запуск #{exc.run_id}")

    if settings.scheduler_enabled:
        with sf() as s:
            last_start = s.scalar(select(func.max(Run.started_at)))
        interval = timedelta(minutes=settings.run_interval_minutes)
        now = datetime.now(timezone.utc)
        if last_start is not None and last_start.tzinfo is None:
            last_start = last_start.replace(tzinfo=timezone.utc)
        first = now + timedelta(seconds=20) if last_start is None or last_start + interval <= now else last_start + interval
        scheduler.add_job(
            scheduled_run,
            IntervalTrigger(minutes=settings.run_interval_minutes),
            next_run_time=first,
            id="hourly",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
        )
        scheduler.start()
        monitor.event("info", f"Сервис запущен, первый плановый запуск в {first.astimezone(pipeline.tz):%H:%M:%S}")
    else:
        monitor.event("info", "Сервис запущен, планировщик выключен")

    app.state.sf = sf
    app.state.monitor = monitor
    app.state.pipeline = pipeline
    app.state.scheduler = scheduler
    app.state.similar = SimilarityCache()
    pipeline.start_letsplay_workers()
    yield
    await pipeline.stop_letsplay_workers()
    if scheduler.running:
        scheduler.shutdown(wait=False)
    if pipeline._task and not pipeline._task.done():
        pipeline._task.cancel()
    await mc.aclose()
    await llm.aclose()


app = FastAPI(title="Metacritic Games Digest", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError):
    return _error(exc.status, exc.code, exc.message, exc.details)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/api/"):
        return _error(exc.status_code, "HTTP_ERROR", str(exc.detail))
    return templates.TemplateResponse(request, "error.html", {"status": exc.status_code, "message": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    return _error(422, "VALIDATION_ERROR", "Некорректные параметры запроса", {"errors": exc.errors()})


# ---- data access ------------------------------------------------------------------


def _list_games(request: Request, q: str, platform: str, sort: str, page: int) -> tuple[list[Game], int]:
    stmt = select(Game)
    count = select(func.count()).select_from(Game)
    if q:
        pattern = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        stmt = stmt.where(Game.title.ilike(pattern, escape="\\"))
        count = count.where(Game.title.ilike(pattern, escape="\\"))
    if platform:
        cond = Game.platforms.any(GamePlatform.name == platform)
        stmt = stmt.where(cond)
        count = count.where(cond)
    if sort == "userscore":
        order = (Game.userscore.is_(None), Game.userscore.desc(), Game.metascore.desc())
    elif sort == "new":
        order = (Game.release_date.is_(None), Game.release_date.desc(), Game.title)
    elif sort == "title":
        order = (Game.title,)
    else:
        order = (Game.metascore.is_(None), Game.metascore.desc(), Game.userscore.desc())
    stmt = stmt.order_by(*order, Game.id).offset((page - 1) * PER_PAGE).limit(PER_PAGE)
    with request.app.state.sf() as s:
        return list(s.scalars(stmt)), s.scalar(count) or 0


def _platforms(request: Request) -> list[tuple[str, int]]:
    with request.app.state.sf() as s:
        rows = s.execute(
            select(GamePlatform.name, func.count(GamePlatform.id)).group_by(GamePlatform.name).order_by(func.count(GamePlatform.id).desc())
        )
        return [(name, n) for name, n in rows]


def _get_game(request: Request, slug: str) -> Game:
    with request.app.state.sf() as s:
        game = s.scalar(select(Game).where(Game.slug == slug))
    if game is None:
        raise HTTPException(404, "Игра не найдена")
    return game


def _similar(request: Request, game: Game) -> list[dict]:
    def load() -> list[GameFeatures]:
        with request.app.state.sf() as s:
            rows = s.execute(select(Game.id, Game.slug, Game.title, Game.description, Game.genres, Game.franchise, Game.developer))
            return [GameFeatures(r.id, r.slug, r.title, r.description or "", r.genres or [], r.franchise, r.developer) for r in rows]

    index = request.app.state.similar.get(request.app.state.pipeline.data_version, load)
    return [s.__dict__ for s in index.similar(game.id)]


def _game_json(g: Game, full: bool = False) -> dict:
    data = {
        "slug": g.slug,
        "title": g.title,
        "cover_url": g.cover_url,
        "release_date": g.release_date,
        "developer": g.developer,
        "metascore": g.metascore,
        "userscore": g.userscore,
        "platforms": [
            {"name": p.name, "metascore": p.metascore, "critic_reviews": p.critic_reviews, "userscore": p.userscore, "user_reviews": p.user_reviews}
            for p in g.platforms
        ],
        "url": f"/games/{g.slug}",
    }
    if full:
        data.update(
            publisher=g.publisher,
            description=g.description,
            genres=g.genres,
            franchise=g.franchise,
            video_url=g.video_url,
            video_title=g.video_title,
            critic_summary=g.critic_summary,
            user_summary=g.user_summary,
            letsplay_status=g.letsplay_status,
            letsplay=g.letsplay,
            metacritic_url=f"https://www.metacritic.com/game/{g.slug}/",
            updated_at=g.updated_at.isoformat() if g.updated_at else None,
        )
    return data


def _monitor_payload(request: Request) -> dict:
    pipeline: Pipeline = request.app.state.pipeline
    job = request.app.state.scheduler.get_job("hourly") if request.app.state.scheduler.running else None
    with request.app.state.sf() as s:
        runs = list(s.scalars(select(Run).order_by(Run.id.desc()).limit(12)))
    return {
        **request.app.state.monitor.snapshot(),
        "stats": pipeline.stats(),
        "running": pipeline.running,
        "next_run_at": job.next_run_time.isoformat() if job and job.next_run_time else None,
        "interval_minutes": settings.run_interval_minutes,
        "token_required": bool(settings.admin_token),
        "runs": [
            {
                "id": r.id,
                "trigger": r.trigger,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "sources": r.sources,
                "games_total": r.games_total,
                "games_ok": r.games_ok,
                "games_failed": r.games_failed,
                "games_new": r.games_new,
                "error": r.error,
            }
            for r in runs
        ],
    }


# ---- pages ------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    q: str = Query("", max_length=200),
    platform: str = Query("", max_length=100),
    sort: str = Query("metascore"),
    page: int = Query(1, ge=1, le=10000),
):
    sort = sort if sort in SORTS else "metascore"
    games, total = _list_games(request, q.strip(), platform, sort, page)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "games": games,
            "total": total,
            "q": q,
            "platform": platform,
            "sort": sort,
            "sorts": SORTS,
            "page": page,
            "pages": max(1, -(-total // PER_PAGE)),
            "platforms": _platforms(request),
        },
    )


@app.get("/games/{slug}", response_class=HTMLResponse)
def game_page(request: Request, slug: str):
    game = _get_game(request, slug)
    return templates.TemplateResponse(request, "game.html", {"g": game, "similar": _similar(request, game)})


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page(request: Request):
    return templates.TemplateResponse(request, "monitor.html", {"initial": json.dumps(_monitor_payload(request), ensure_ascii=False).replace("</", "<\\/")})


# ---- JSON API -----------------------------------------------------------------------


@app.get("/api/health")
def health(request: Request):
    with request.app.state.sf() as s:
        s.execute(select(1))
    return {"success": True, "status": "ok", "running": request.app.state.pipeline.running}


@app.get("/api/games")
def api_games(
    request: Request,
    q: str = Query("", max_length=200),
    platform: str = Query("", max_length=100),
    sort: str = Query("metascore"),
    page: int = Query(1, ge=1, le=10000),
):
    if sort not in SORTS:
        raise ApiError(422, "BAD_SORT", f"sort должен быть одним из: {', '.join(SORTS)}")
    games, total = _list_games(request, q.strip(), platform, sort, page)
    return {"success": True, "total": total, "page": page, "per_page": PER_PAGE, "items": [_game_json(g) for g in games]}


@app.get("/api/games/{slug}")
def api_game(request: Request, slug: str):
    game = _get_game(request, slug)
    return {"success": True, "item": {**_game_json(game, full=True), "similar": _similar(request, game)}}


@app.get("/api/platforms")
def api_platforms(request: Request):
    return {"success": True, "items": [{"name": n, "games": c} for n, c in _platforms(request)]}


@app.get("/api/monitor")
def api_monitor(request: Request):
    return {"success": True, **_monitor_payload(request)}


@app.get("/api/monitor/stream")
async def api_monitor_stream(request: Request):
    async def events():
        last_version, last_sent = -1, 0.0
        while not await request.is_disconnected():
            version = request.app.state.monitor.version
            if version != last_version or time.time() - last_sent > 5:
                payload = await asyncio.to_thread(_monitor_payload, request)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_version, last_sent = version, time.time()
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/runs", status_code=202)
async def api_start_run(request: Request):
    if settings.admin_token and request.headers.get("X-Admin-Token") != settings.admin_token:
        raise ApiError(403, "BAD_ADMIN_TOKEN", "Нужен корректный X-Admin-Token")
    try:
        run_id = request.app.state.pipeline.start("manual")
    except AlreadyRunning as exc:
        raise ApiError(409, "RUN_IN_PROGRESS", "Обработка уже идёт", {"run_id": exc.run_id})
    return {"success": True, "run_id": run_id}


@app.get("/api/runs")
def api_runs(request: Request):
    return {"success": True, "items": _monitor_payload(request)["runs"]}
