"""Maintenance commands that run next to the service on the same database.

    python -m app.cli resummarize     # rebuild review summaries that failed or were skipped (no LLM key at the time)
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import select

from .config import settings
from .db import Game, make_session_factory
from .llm import is_russian_text, make_llm
from .metacritic import MetacriticClient
from .monitor import Monitor
from .pipeline import Pipeline


def _needs_summary(summary: dict | None, review_count: int) -> bool:
    if review_count <= 0:
        return False
    if summary is None or summary.get("error"):
        return True
    return not is_russian_text([summary.get("verdict", "")] + summary.get("likes", []) + summary.get("dislikes", []))


async def resummarize() -> None:
    sf = make_session_factory(settings.database_url)
    mc = MetacriticClient()
    llm = make_llm(settings)
    pipeline = Pipeline(sf, mc, llm, Monitor(), settings)
    llm.set_hooks(pipeline.check_budget, pipeline.record_usage)
    with sf() as s:
        slugs = [
            g.slug
            for g in s.scalars(select(Game))
            if _needs_summary(g.critic_summary, g.critic_review_count) or _needs_summary(g.user_summary, g.user_review_count)
        ]
    print(f"games to resummarize: {len(slugs)}")
    with sf.begin() as s:
        for g in s.scalars(select(Game).where(Game.slug.in_(slugs))):
            g.critic_reviews_fp = None
            g.user_reviews_fp = None
    try:
        for slug in slugs:
            try:
                await pipeline.process_game(slug)
                with sf() as s:
                    g = s.scalar(select(Game).where(Game.slug == slug))
                    ok = [k for k, v in (("critics", g.critic_summary), ("users", g.user_summary)) if v and v.get("verdict")]
                print(f"  {slug}: {', '.join(ok) or 'no summary'}")
            except Exception as exc:
                print(f"  {slug}: {type(exc).__name__}: {exc}")
    finally:
        await mc.aclose()
        await llm.aclose()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["resummarize"])
    args = parser.parse_args()
    if args.command == "resummarize":
        asyncio.run(resummarize())


if __name__ == "__main__":
    main()
