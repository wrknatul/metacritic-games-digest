"""In-process live state of the service: workers, current run, event log.

Workers and the pipeline write here; the /monitor page reads snapshots over SSE.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class WorkerState:
    name: str
    kind: str  # detail | letsplay
    status: str = "idle"  # idle | busy | error
    game: str | None = None
    stage: str | None = None
    since: float = field(default_factory=time.time)
    done: int = 0
    failed: int = 0
    last_error: str | None = None


@dataclass
class RunProgress:
    id: int
    trigger: str
    started_at: float
    phase: str = "discovery"  # discovery | details | finished
    sources: list[str] = field(default_factory=list)
    total: int = 0
    done: int = 0
    failed: int = 0
    new: int = 0


class Monitor:
    def __init__(self, log_size: int = 300):
        self.workers: dict[str, WorkerState] = {}
        self.run: RunProgress | None = None
        self.last_run: RunProgress | None = None
        self.log: deque[dict] = deque(maxlen=log_size)
        self.version = 0
        self.started_at = time.time()

    def _touch(self) -> None:
        self.version += 1

    def register(self, name: str, kind: str) -> WorkerState:
        w = WorkerState(name=name, kind=kind)
        self.workers[name] = w
        self._touch()
        return w

    def unregister_all(self, kind: str) -> None:
        for name in [n for n, w in self.workers.items() if w.kind == kind]:
            del self.workers[name]
        self._touch()

    def busy(self, w: WorkerState, game: str, stage: str) -> None:
        if w.game != game:
            w.since = time.time()
        w.status, w.game, w.stage = "busy", game, stage
        self._touch()

    def stage(self, w: WorkerState, stage: str | None) -> None:
        w.stage = stage
        self._touch()

    def idle(self, w: WorkerState, ok: bool, error: str | None = None) -> None:
        if ok:
            w.done += 1
        else:
            w.failed += 1
            w.last_error = (error or "")[:300]
        w.status = "idle" if ok else "error"
        w.game, w.stage, w.since = None, None, time.time()
        self._touch()

    def event(self, level: str, msg: str) -> None:
        self.log.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "level": level, "msg": msg})
        self._touch()

    def start_run(self, run_id: int, trigger: str) -> RunProgress:
        self.run = RunProgress(id=run_id, trigger=trigger, started_at=time.time())
        self._touch()
        return self.run

    def finish_run(self) -> None:
        if self.run:
            self.run.phase = "finished"
            self.last_run = self.run
        self.run = None
        self._touch()

    def bump(self) -> None:
        self._touch()

    def snapshot(self) -> dict:
        return {
            "version": self.version,
            "uptime_s": int(time.time() - self.started_at),
            "run": asdict(self.run) if self.run else None,
            "last_run": asdict(self.last_run) if self.last_run else None,
            "workers": [asdict(w) for w in self.workers.values()],
            "log": list(self.log)[-80:],
        }
