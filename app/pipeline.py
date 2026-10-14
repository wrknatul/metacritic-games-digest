"""One processing run: pick today's next games, fetch them, summarize reviews, find let's plays."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from . import letsplay as lp
from .config import Settings
from .db import DailyCursor, DailyProcessed, Game, GamePlatform, LlmUsage, Run, utcnow
from .llm import BudgetExceeded, LLMClient, LLMError, LLMUnavailable
from .metacritic import GameDetails, MetacriticClient, MetacriticError, ReviewSample
from .monitor import Monitor, RunProgress, WorkerState

log = logging.getLogger(__name__)

# SEE ALL pages looked at in one run before giving up (all of them may be done already)
MAX_PAGES_PER_RUN = 6
GAME_TIMEOUT_S = 180
LETSPLAY_TIMEOUT_S = 600
LETSPLAY_BLOCKED_STREAK = 3
LETSPLAY_BLOCKED_PAUSE_S = 15 * 60
LLM_UNAVAILABLE_PAUSE_S = 10 * 60


class AlreadyRunning(Exception):
    def __init__(self, run_id: int | None):
        super().__init__(f"run {run_id} is in progress")
        self.run_id = run_id


class Pipeline:
    def __init__(
        self,
        session_factory: sessionmaker,
        metacritic: MetacriticClient,
        llm: LLMClient,
        monitor: Monitor,
        settings: Settings,
    ):
        self.sf = session_factory
        self.mc = metacritic
        self.llm = llm
        self.monitor = monitor
        self.settings = settings
        self.tz = ZoneInfo(settings.timezone)
        self.active_run_id: int | None = None
        self.data_version = 0
        self._task: asyncio.Task | None = None
        self._run_day: str | None = None
        self._lp_tasks: list[asyncio.Task] = []
        self._lp_inflight: set[int] = set()
        self._lp_wakeup: asyncio.Event | None = None

    # ---- day and budget helpers -------------------------------------------------

    def today(self) -> str:
        return datetime.now(self.tz).date().isoformat()

    def check_budget(self) -> None:
        with self.sf() as s:
            usage = s.get(LlmUsage, self.today())
            if usage and usage.calls >= self.settings.llm_daily_call_limit:
                raise BudgetExceeded(f"daily LLM call limit {self.settings.llm_daily_call_limit} reached")

    def record_usage(self, usage: dict) -> None:
        with self.sf.begin() as s:
            row = s.get(LlmUsage, self.today())
            if row is None:
                row = LlmUsage(day=self.today(), calls=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0)
                s.add(row)
            row.calls += 1
            row.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            row.completion_tokens += int(usage.get("completion_tokens") or 0)
            row.cost_usd += float(usage.get("cost") or 0.0)

    # ---- run lifecycle ----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.active_run_id is not None

    def start(self, trigger: str) -> int:
        """Create a run and process it in the background. Raises AlreadyRunning."""
        if self.active_run_id is not None:
            raise AlreadyRunning(self.active_run_id)
        with self.sf.begin() as s:
            run = Run(trigger=trigger, status="running", day=self.today(), started_at=utcnow())
            s.add(run)
            s.flush()
            run_id = run.id
        self.active_run_id = run_id
        self._task = asyncio.create_task(self._execute(run_id, trigger))
        return run_id

    async def wait(self) -> None:
        if self._task:
            await self._task

    async def _execute(self, run_id: int, trigger: str) -> None:
        progress = self.monitor.start_run(run_id, trigger)
        self.monitor.event("info", f"Запуск #{run_id} ({'по расписанию' if trigger == 'schedule' else 'вручную'})")
        status, error = "done", None
        try:
            day = self._run_day = self.today()
            slugs, sources = await self.select_games(day, run_id)
            progress.sources, progress.total = sources, len(slugs)
            self.monitor.event("info", f"Выбрано игр: {len(slugs)} ({'; '.join(sources) or 'нет новых'})")
            if slugs:
                await self._process(slugs, run_id, progress)
        except Exception as exc:  # the run must always be closed in the DB
            log.exception("run %s failed", run_id)
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            self.monitor.event("error", f"Запуск #{run_id} упал: {error}")
        finally:
            with self.sf.begin() as s:
                run = s.get(Run, run_id)
                run.status = status
                run.error = error
                run.finished_at = utcnow()
                run.sources = progress.sources
                run.games_total = progress.total
                run.games_ok = progress.done
                run.games_failed = progress.failed
                run.games_new = progress.new
            self.monitor.event(
                "info" if status == "done" else "error",
                f"Запуск #{run_id} завершён: {progress.done} ок, {progress.failed} ошибок, новых {progress.new}",
            )
            self.monitor.finish_run()
            self.active_run_id = None

    # ---- selection --------------------------------------------------------------

    async def select_games(self, day: str, run_id: int) -> tuple[list[str], list[str]]:
        """Today's next games: New Releases first, then SEE ALL pages from where we stopped."""
        n = self.settings.games_per_run
        with self.sf() as s:
            cursor = s.get(DailyCursor, day)
            new_releases_done = cursor.new_releases_done if cursor else False
            page = cursor.next_page if cursor else 1
            done = set(s.scalars(select(DailyProcessed.slug).where(DailyProcessed.day == day)))

        picked: list[str] = []
        sources: list[str] = []

        def take(slugs: list[str]) -> int:
            taken = 0
            for slug in slugs:
                if len(picked) >= n:
                    break
                if slug not in done and slug not in picked:
                    picked.append(slug)
                    taken += 1
            return taken

        if not new_releases_done:
            try:
                taken = take(await self.mc.new_releases())
                sources.append(f"New Releases: {taken}")
                new_releases_done = True
            except MetacriticError as exc:
                self.monitor.event("warn", f"New Releases не получены, беру SEE ALL: {exc}")

        for _ in range(MAX_PAGES_PER_RUN):
            if len(picked) >= n:
                break
            try:
                slugs = await self.mc.browse_page(page)
            except MetacriticError as exc:
                if not picked:
                    raise
                self.monitor.event("warn", f"SEE ALL стр. {page} не получена, обрабатываю уже выбранные: {exc}")
                break
            if not slugs:
                break
            taken = take(slugs)
            if taken:
                sources.append(f"SEE ALL стр. {page}: {taken}")
            if all(s in done or s in picked for s in slugs):
                page += 1  # the page is used up, the next run starts from the next one
            else:
                break

        with self.sf.begin() as s:
            cursor = s.get(DailyCursor, day)
            if cursor is None:
                cursor = DailyCursor(day=day)
                s.add(cursor)
            cursor.new_releases_done = new_releases_done
            cursor.next_page = page
            # claim the games right away so a crash never makes the walk repeat itself
            for slug in picked:
                s.merge(DailyProcessed(day=day, slug=slug, run_id=run_id, ok=False))
        return picked, sources

    # ---- processing -------------------------------------------------------------

    async def _process(self, slugs: list[str], run_id: int, progress: RunProgress) -> None:
        detail_q: asyncio.Queue[str] = asyncio.Queue()
        for slug in slugs:
            detail_q.put_nowait(slug)

        progress.phase = "details"
        detail_workers = [
            asyncio.create_task(self._detail_worker(self.monitor.register(f"detail-{i + 1}", "detail"), detail_q, progress))
            for i in range(self.settings.detail_workers)
        ]
        try:
            await detail_q.join()
        finally:
            for t in detail_workers:
                t.cancel()
            await asyncio.gather(*detail_workers, return_exceptions=True)
            self.monitor.unregister_all("detail")

    async def _detail_worker(self, w: WorkerState, q: asyncio.Queue, progress: RunProgress) -> None:
        while True:
            slug = await q.get()
            try:
                self.monitor.busy(w, slug, "Metacritic: карточка")
                game_id, is_new, needs_lp = await asyncio.wait_for(self.process_game(slug, w), GAME_TIMEOUT_S)
                progress.done += 1
                progress.new += int(is_new)
                self._mark(slug, ok=True)
                if needs_lp:
                    self.wake_letsplay()
                self.monitor.idle(w, ok=True)
            except Exception as exc:
                progress.failed += 1
                err = f"{type(exc).__name__}: {exc}"
                log.warning("game %s failed: %s", slug, err)
                self.monitor.idle(w, ok=False, error=err)
                self.monitor.event("error", f"{slug}: {err}")
            finally:
                q.task_done()

    def _mark(self, slug: str, ok: bool) -> None:
        with self.sf.begin() as s:
            row = s.get(DailyProcessed, (self._run_day or self.today(), slug))
            if row:
                row.ok = ok

    async def process_game(self, slug: str, w: WorkerState | None = None) -> tuple[int, bool, bool]:
        details = await self.mc.game_details(slug)
        if w:
            self.monitor.stage(w, "Metacritic: отзывы")
        # games without any reviews (most of the SEE ALL tail) need no review calls
        if sum(p.critic_reviews for p in details.platforms) or details.critic_total:
            critic = await self.mc.critic_reviews(slug, details.critic_counts, details.critic_total)
        else:
            critic = ReviewSample("critic", 0, details.critic_counts, [])
        if sum(p.user_reviews for p in details.platforms):
            users = await self.mc.user_reviews(slug)
        else:
            users = ReviewSample("user", 0, {"positive": 0, "neutral": 0, "negative": 0}, [])

        with self.sf() as s:
            existing = s.scalar(select(Game).where(Game.slug == slug))
            old = {
                "critic_fp": existing.critic_reviews_fp if existing else None,
                "user_fp": existing.user_reviews_fp if existing else None,
                "critic_summary": existing.critic_summary if existing else None,
                "user_summary": existing.user_summary if existing else None,
            }

        if w:
            self.monitor.stage(w, "LLM: резюме отзывов")
        (critic_summary, critic_fp), (user_summary, user_fp) = await asyncio.gather(
            self._summary(details.title, critic, old["critic_fp"], old["critic_summary"]),
            self._summary(details.title, users, old["user_fp"], old["user_summary"]),
        )

        return self._upsert(details, critic, users, critic_summary, critic_fp, user_summary, user_fp)

    async def _summary(self, title: str, sample: ReviewSample, old_fp: str | None, old_summary: dict | None) -> tuple[dict | None, str | None]:
        fp = sample.fingerprint()
        if fp is None:
            return None, None
        if fp == old_fp and old_summary and not old_summary.get("error"):
            return old_summary, fp  # same reviews as last time, no LLM call
        if not self.llm.enabled:
            return {"error": "LLM не настроен (нет LLM_API_KEY)"}, None
        try:
            result = await self.llm.summarize_reviews(
                title,
                sample.kind,
                sample.total,
                sample.counts,
                [{"text": r.text, "score": r.score} for r in sample.reviews],
            )
        except (LLMError, BudgetExceeded) as exc:
            self.monitor.event("warn", f"{title}: резюме {sample.kind} не построено: {exc}")
            # keep the previous summary if there is one, retry next time
            return (old_summary or {"error": str(exc)[:200]}), None
        return {
            **result.model_dump(),
            "sample_size": len(sample.reviews),
            "total": sample.total,
            "counts": sample.counts,
            "model": self.llm.model,
            "generated_at": utcnow().isoformat(timespec="seconds"),
        }, fp

    def _upsert(
        self,
        d: GameDetails,
        critic: ReviewSample,
        users: ReviewSample,
        critic_summary: dict | None,
        critic_fp: str | None,
        user_summary: dict | None,
        user_fp: str | None,
    ) -> tuple[int, bool, bool]:
        now = utcnow()
        with self.sf.begin() as s:
            g = s.scalar(select(Game).where(Game.slug == d.slug))
            is_new = g is None
            if is_new:
                g = Game(slug=d.slug, first_seen_at=now)
                s.add(g)
            g.title = d.title
            g.cover_url = d.cover_url
            g.developer = d.developer
            g.publisher = d.publisher
            g.description = d.description
            g.release_date = d.release_date
            g.genres = d.genres
            g.franchise = d.franchise
            g.video_url = d.video_url
            g.video_title = d.video_title

            lead = next((p for p in d.platforms if p.is_lead), d.platforms[0] if d.platforms else None)
            metascores = [p.metascore for p in d.platforms if p.metascore is not None]
            userscores = [p.userscore for p in d.platforms if p.userscore is not None]
            g.metascore = lead.metascore if lead and lead.metascore is not None else max(metascores, default=None)
            g.userscore = lead.userscore if lead and lead.userscore is not None else max(userscores, default=None)
            g.critic_review_count = critic.total or sum(p.critic_reviews for p in d.platforms)
            g.user_review_count = sum(p.user_reviews for p in d.platforms) or users.total

            if critic_summary is not None or critic.fingerprint() is None:
                g.critic_summary = critic_summary
                g.critic_reviews_fp = critic_fp
            if user_summary is not None or users.fingerprint() is None:
                g.user_summary = user_summary
                g.user_reviews_fp = user_fp
            g.summaries_updated_at = now

            g.platforms.clear()
            s.flush()
            for pos, p in enumerate(d.platforms):
                g.platforms.append(
                    GamePlatform(
                        name=p.name,
                        slug=p.slug,
                        position=pos,
                        is_lead=p.is_lead,
                        metascore=p.metascore,
                        critic_reviews=p.critic_reviews,
                        userscore=p.userscore,
                        user_reviews=p.user_reviews,
                    )
                )
            g.updated_at = now
            needs_lp = self._needs_letsplay(g, now)
            if needs_lp:
                g.letsplay_status = "pending"
            s.flush()
            game_id = g.id
        self.data_version += 1
        return game_id, is_new, needs_lp

    def _needs_letsplay(self, g: Game, now: datetime) -> bool:
        if g.letsplay_status in (None, "pending"):
            return True
        if g.letsplay_status == "failed" and not (g.letsplay or {}).get("video_id"):
            return True  # failed before a video was found (search error): retry on the next visit
        if g.letsplay_status == "done":
            return False
        checked = g.letsplay_checked_at
        if checked is not None and checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
        return checked is None or now - checked > timedelta(days=self.settings.letsplay_retry_days)

    # ---- let's plays ------------------------------------------------------------

    def start_letsplay_workers(self) -> None:
        """Long-lived workers over the DB queue (games with letsplay_status = pending).

        They run apart from the hourly runs: a let's play takes minutes, and 20 of them
        must not push the next hourly run off schedule.
        """
        if not self.settings.letsplay_enabled or self._lp_tasks:
            return
        self._lp_wakeup = asyncio.Event()
        for i in range(self.settings.letsplay_workers):
            w = self.monitor.register(f"letsplay-{i + 1}", "letsplay")
            self._lp_tasks.append(asyncio.create_task(self._letsplay_loop(w)))

    async def stop_letsplay_workers(self) -> None:
        for t in self._lp_tasks:
            t.cancel()
        await asyncio.gather(*self._lp_tasks, return_exceptions=True)
        self._lp_tasks.clear()

    def wake_letsplay(self) -> None:
        if self._lp_wakeup is not None:
            self._lp_wakeup.set()

    def claim_letsplay(self) -> int | None:
        """Next pending game, popular ones first: they are the ones that have let's plays."""
        with self.sf() as s:
            q = select(Game.id).where(Game.letsplay_status == "pending")
            if self._lp_inflight:
                q = q.where(Game.id.not_in(self._lp_inflight))
            q = q.order_by(Game.user_review_count.desc(), Game.critic_review_count.desc(), Game.id).limit(1)
            game_id = s.scalar(q)
        if game_id is not None:
            self._lp_inflight.add(game_id)
        return game_id

    async def _letsplay_loop(self, w: WorkerState) -> None:
        blocked_streak = 0
        while True:
            if not self.llm.enabled:
                self.monitor.stage(w, "ждёт ключ LLM")
                await self._lp_idle(300)
                continue
            game_id = self.claim_letsplay()
            if game_id is None:
                self.monitor.stage(w, None)
                await self._lp_idle(120)
                continue
            try:
                status = await self.run_letsplay(game_id, w)
            finally:
                self._lp_inflight.discard(game_id)
            if status in ("llm_unavailable", "youtube_unreachable"):
                reason = "LLM недоступен (ключ или баланс)" if status == "llm_unavailable" else "нет связи с YouTube"
                self.monitor.stage(w, f"пауза 10 мин: {reason}")
                await asyncio.sleep(LLM_UNAVAILABLE_PAUSE_S)
                continue
            blocked_streak = blocked_streak + 1 if status == "blocked" else 0
            if blocked_streak >= LETSPLAY_BLOCKED_STREAK:
                self.monitor.event("warn", f"YouTube отказал {blocked_streak} раза подряд, пауза {LETSPLAY_BLOCKED_PAUSE_S // 60} мин")
                self.monitor.stage(w, "пауза: YouTube ограничивает запросы")
                await asyncio.sleep(LETSPLAY_BLOCKED_PAUSE_S)
                blocked_streak = 0

    async def run_letsplay(self, game_id: int, w: WorkerState | None = None) -> str:
        """process_letsplay with a timeout; any error is stored on the game, never raised."""
        try:
            status = await asyncio.wait_for(self.process_letsplay(game_id, w), LETSPLAY_TIMEOUT_S)
            if w:
                self.monitor.idle(w, ok=True)
            return status
        except asyncio.CancelledError:
            raise
        except lp.YouTubeUnreachable as exc:
            self.monitor.event("warn", f"летсплей игры #{game_id} отложен: нет связи с YouTube ({str(exc)[-120:]})")
            if w:
                self.monitor.idle(w, ok=False, error=str(exc)[-300:])
            return "youtube_unreachable"
        except (LLMUnavailable, BudgetExceeded) as exc:
            # the game stays pending: nothing is wrong with it, the LLM is out for now
            self.monitor.event("warn", f"летсплей игры #{game_id} отложен: {str(exc)[:160]}")
            if w:
                self.monitor.idle(w, ok=False, error=str(exc)[:300])
            return "llm_unavailable"
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self._save_letsplay(game_id, "failed", {"error": err[:500]})
            if w:
                self.monitor.idle(w, ok=False, error=err)
            self.monitor.event("warn", f"летсплей игры #{game_id}: {err}")
            return "failed"

    async def _lp_idle(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._lp_wakeup.wait(), seconds)
        except asyncio.TimeoutError:
            pass
        self._lp_wakeup.clear()

    async def process_letsplay(self, game_id: int, w: WorkerState | None = None) -> str:
        with self.sf() as s:
            g = s.get(Game, game_id)
            title = g.title
        if w:
            self.monitor.busy(w, title, "YouTube: поиск летсплея")
        video = lp.pick_letsplay(title, await lp.search(title))
        if video is None:
            self._save_letsplay(game_id, "not_found", None)
            return "not_found"

        minutes = self.settings.letsplay_audio_minutes
        transcript, source, errors, blocked = None, None, [], 0
        strategies = [x.strip() for x in self.settings.letsplay_strategies.split(",") if x.strip()]
        for strategy in strategies:
            if strategy != "captions" and not self.llm.enabled:
                continue
            if w:
                self.monitor.stage(w, f"речь в текст: {strategy}")
            try:
                if strategy == "captions":
                    text = await lp.fetch_captions(video, minutes)
                elif strategy == "audio":
                    text = await self.llm.transcribe(await lp.fetch_audio(video, minutes))
                else:
                    text = await self.llm.transcribe_youtube(video.url, minutes)
            except lp.YouTubeBlocked as exc:
                blocked += 1
                errors.append(f"{strategy}: YouTube отказал ({str(exc)[-120:]})")
                continue
            except (lp.LetsPlayError, LLMError) as exc:
                errors.append(f"{strategy}: {str(exc)[-200:]}")
                continue
            if text and len(text) >= 200:
                transcript, source = text, strategy
                break
            errors.append(f"{strategy}: пустой текст")

        info = {
            "video_id": video.id,
            "url": video.url,
            "title": video.title,
            "channel": video.channel,
            "views": video.views,
            "duration": video.duration,
        }
        if transcript is None:
            status = "blocked" if blocked and blocked == len(errors) else "failed"
            self._save_letsplay(game_id, status, {**info, "errors": errors})
            return status

        if w:
            self.monitor.stage(w, "LLM: заключение по летсплею")
        conclusion = await self.llm.letsplay_conclusion(title, video.title, video.channel, transcript)
        self._save_letsplay(
            game_id,
            "done",
            {
                **info,
                **conclusion.model_dump(),
                "transcript_source": source,
                "transcript_minutes": minutes,
                "transcript": transcript[:12000],
                "errors": errors,
                "model": self.llm.model,
            },
        )
        self.monitor.event("info", f"{title}: летсплей разобран ({video.channel}, {video.views:,} просмотров, {source})")
        return "done"

    def _save_letsplay(self, game_id: int, status: str, data: dict | None) -> None:
        with self.sf.begin() as s:
            g = s.get(Game, game_id)
            if g is None:
                return
            g.letsplay_status = status
            g.letsplay_checked_at = utcnow()
            if data is not None or status == "not_found":
                g.letsplay = data
        self.data_version += 1

    # ---- stats for the monitor ----------------------------------------------------

    def stats(self) -> dict:
        day = self.today()
        with self.sf() as s:
            usage = s.get(LlmUsage, day)
            return {
                "day": day,
                "games_total": s.scalar(select(func.count()).select_from(Game)) or 0,
                "processed_today": s.scalar(select(func.count()).select_from(DailyProcessed).where(DailyProcessed.day == day)) or 0,
                "runs_today": s.scalar(select(func.count()).select_from(Run).where(Run.day == day)) or 0,
                "letsplay_done": s.scalar(select(func.count()).select_from(Game).where(Game.letsplay_status == "done")) or 0,
                "letsplay_pending": s.scalar(select(func.count()).select_from(Game).where(Game.letsplay_status == "pending")) or 0,
                "letsplay_enabled": self.settings.letsplay_enabled,
                "with_summaries": s.scalar(
                    select(func.count())
                    .select_from(Game)
                    .where(
                        func.json_extract(Game.critic_summary, "$.verdict").is_not(None)
                        | func.json_extract(Game.user_summary, "$.verdict").is_not(None)
                    )
                )
                or 0,
                "next_page": (s.get(DailyCursor, day).next_page if s.get(DailyCursor, day) else 1),
                "llm_calls_today": usage.calls if usage else 0,
                "llm_cost_today": round(usage.cost_usd, 4) if usage else 0.0,
                "llm_limit": self.settings.llm_daily_call_limit,
                "llm_model": self.llm.model,
                "llm_enabled": self.llm.enabled,
            }
