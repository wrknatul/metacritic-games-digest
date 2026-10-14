"""Metacritic client.

Game lists are taken from the HTML pages named in the task (Games -> New Releases,
SEE ALL sorted by New). Game details and reviews come from backend.metacritic.com,
the JSON API the metacritic.com frontend itself calls: it gives per-platform scores
and reviews without parsing fragile markup.
"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import logging
import random
import re
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

SITE = "https://www.metacritic.com"
BACKEND = "https://backend.metacritic.com"
# Public key that metacritic.com embeds in its own pages for backend calls
API_KEY = "1MOZgmNFxvmljaQR1X9KAij9Mo4xAY3u"
IMAGE_BASE = "https://www.metacritic.com/a/img/catalog"

NEW_RELEASES_URL = f"{SITE}/game/"
BROWSE_URL = f"{SITE}/browse/game/all/all/all-time/new/"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
)

_GAME_LINK_RE = re.compile(r'href="/game/([a-z0-9][a-z0-9-]*)/"')

# How many reviews of each sentiment go into an LLM summary
CRITIC_SAMPLE = {"positive": 12, "neutral": 8, "negative": 8}
USER_SAMPLE = {"positive": 20, "neutral": 12, "negative": 20}
QUOTE_MAX_CHARS = 700


class MetacriticError(Exception):
    pass


class ParseError(MetacriticError):
    pass


@dataclass
class PlatformInfo:
    name: str
    slug: str
    is_lead: bool
    metascore: int | None
    critic_reviews: int
    userscore: float | None = None
    user_reviews: int = 0


@dataclass
class GameDetails:
    slug: str
    title: str
    cover_url: str | None
    developer: str | None
    publisher: str | None
    description: str | None
    release_date: str | None
    genres: list[str]
    franchise: str | None
    video_url: str | None
    video_title: str | None
    platforms: list[PlatformInfo]
    critic_total: int = 0
    critic_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Review:
    text: str
    score: float | None
    author: str | None
    platform: str | None
    key: str  # stable identity for fingerprints


@dataclass
class ReviewSample:
    kind: str  # critic | user
    total: int
    counts: dict[str, int]  # positive / neutral / negative over all reviews
    reviews: list[Review]

    def fingerprint(self) -> str | None:
        if not self.reviews:
            return None
        h = hashlib.sha1()
        h.update(f"{self.kind}:{self.total}".encode())
        for key in sorted(r.key for r in self.reviews):
            h.update(key.encode())
        return h.hexdigest()


def _uniq(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def parse_new_releases(page_html: str) -> list[str]:
    """Slugs of the New Releases carousel on https://www.metacritic.com/game/."""
    start = page_html.find('global-header-title__text">New Releases')
    if start < 0:
        raise ParseError("New Releases section not found on /game/")
    end = page_html.find("global-header-title__text", start + 40)
    section = page_html[start : end if end > 0 else None]
    slugs = _uniq(_GAME_LINK_RE.findall(section))
    if not slugs:
        raise ParseError("New Releases section has no game links")
    return slugs


def parse_browse_page(page_html: str) -> list[str]:
    """Slugs of the result grid on a SEE ALL page (sorted by New)."""
    start = page_html.find('data-testid="filter-results"')
    if start < 0:
        raise ParseError("filter-results grid not found on browse page")
    end = page_html.find('data-testid="navigation-pagination"', start)
    section = page_html[start : end if end > 0 else None]
    return _uniq(_GAME_LINK_RE.findall(section))


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    text = html_lib.unescape(value).replace("\r\n", "\n").replace("​", "").strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text or None


def _https(url: str | None) -> str | None:
    # the value ends up in an iframe src: only plain https links are kept
    return url if isinstance(url, str) and url.startswith("https://") else None


def _image_url(images: list[dict], preferred: tuple[str, ...]) -> str | None:
    by_type = {img.get("typeName"): img for img in images or [] if img.get("bucketPath")}
    for type_name in preferred:
        img = by_type.get(type_name)
        if img:
            return IMAGE_BASE + img["bucketPath"]
    return None


def parse_product(composer: dict) -> GameDetails:
    """Turn the composer page payload of a game into GameDetails (user scores filled later)."""
    components = {
        c.get("meta", {}).get("componentName"): c for c in composer.get("components", [])
    }
    product = (components.get("product") or {}).get("data", {}).get("item")
    if not product:
        raise ParseError("composer payload has no product item")

    companies = (product.get("production") or {}).get("companies") or []
    developers = [c["name"] for c in companies if c.get("typeName") == "Developer" and c.get("name")]
    publishers = [c["name"] for c in companies if c.get("typeName") == "Publisher" and c.get("name")]

    platforms = []
    for pos, p in enumerate(product.get("platforms") or []):
        css = p.get("criticScoreSummary") or {}
        platforms.append(
            PlatformInfo(
                name=p.get("name") or p.get("slug") or "?",
                slug=p.get("slug") or "",
                is_lead=bool(p.get("isLeadPlatform")),
                metascore=css.get("score"),
                critic_reviews=css.get("reviewCount") or 0,
            )
        )

    video = product.get("video") or {}
    franchises = (product.get("gameTaxonomy") or {}).get("franchises") or []
    css_all = product.get("criticScoreSummary") or {}

    return GameDetails(
        slug=product["slug"],
        title=_clean_text(product.get("title")) or product["slug"],
        cover_url=_image_url(product.get("images"), ("cardImage", "mainImage")),
        developer=", ".join(developers) or None,
        publisher=", ".join(publishers) or None,
        description=_clean_text(product.get("description")),
        release_date=product.get("releaseDate"),
        genres=[g["name"] for g in product.get("genres") or [] if g.get("name")],
        franchise=franchises[0]["name"] if franchises and franchises[0].get("name") else None,
        video_url=_https(video.get("embedUrl") or video.get("url")),
        video_title=_clean_text(video.get("title")),
        platforms=platforms,
        critic_total=css_all.get("reviewCount") or 0,
        critic_counts={
            "positive": css_all.get("positiveCount") or 0,
            "neutral": css_all.get("neutralCount") or 0,
            "negative": css_all.get("negativeCount") or 0,
        },
    )


def parse_reviews(payload: dict, kind: str) -> list[Review]:
    items = (payload.get("data") or {}).get("items") or []
    out = []
    for it in items:
        text = _clean_text(it.get("quote"))
        if not text:
            continue
        platform = it.get("platform")
        if isinstance(platform, dict):
            platform = platform.get("name")
        if kind == "critic":
            author = it.get("publicationName") or it.get("author")
            key = f"{it.get('publicationSlug') or author}|{it.get('url') or it.get('date')}"
        else:
            author = it.get("author")
            key = str(it.get("id") or f"{author}|{it.get('date')}")
        out.append(
            Review(
                text=text[:QUOTE_MAX_CHARS],
                score=it.get("score"),
                author=author,
                platform=platform,
                key=key,
            )
        )
    return out


class MetacriticClient:
    def __init__(self, client: httpx.AsyncClient | None = None, max_concurrency: int = 4):
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )
        self._sem = asyncio.Semaphore(max_concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str, params: dict | None = None, attempts: int = 4) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self._sem:
                    resp = await self._client.get(url, params=params)
                if resp.status_code == 404:
                    return resp
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise MetacriticError(f"HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                return resp
            except (httpx.HTTPError, MetacriticError) as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    break
                delay = 1.5 * 2**attempt + random.random()
                log.warning("metacritic retry %s in %.1fs: %s", attempt + 1, delay, exc)
                await asyncio.sleep(delay)
        raise MetacriticError(f"giving up on {url}: {last_exc}")

    async def _backend(self, path: str, **params) -> dict:
        resp = await self._get(f"{BACKEND}{path}", params={**params, "apiKey": API_KEY})
        if resp.status_code == 404:
            return {}
        return resp.json()

    async def new_releases(self) -> list[str]:
        resp = await self._get(NEW_RELEASES_URL)
        return parse_new_releases(resp.text)

    async def browse_page(self, page: int) -> list[str]:
        resp = await self._get(BROWSE_URL, params={"page": page} if page > 1 else None)
        return parse_browse_page(resp.text)

    async def game_details(self, slug: str) -> GameDetails:
        payload = await self._backend(f"/composer/metacritic/pages/games/{slug}/web")
        if not payload:
            raise MetacriticError(f"game {slug} not found")
        details = parse_product(payload)
        stats = await asyncio.gather(
            *(self._user_stats(slug, p.slug) for p in details.platforms if p.slug),
            return_exceptions=True,
        )
        with_slug = [p for p in details.platforms if p.slug]
        for platform, st in zip(with_slug, stats):
            if isinstance(st, Exception):
                log.warning("user score for %s/%s failed: %s", slug, platform.slug, st)
                continue
            platform.userscore, platform.user_reviews = st
        return details

    async def _user_stats(self, slug: str, platform_slug: str) -> tuple[float | None, int]:
        data = await self._backend(f"/reviews/metacritic/user/games/{slug}/platform/{platform_slug}/stats/web")
        item = (data.get("data") or {}).get("item") or {}
        count = item.get("reviewCount") or 0
        # with too few ratings the API still reports a score (often 0) but no sentiment,
        # and metacritic.com shows "tbd": do the same
        return (item.get("score") if count and item.get("sentiment") else None), count

    async def user_reviews(self, slug: str) -> ReviewSample:
        stats = await self._backend(f"/reviews/metacritic/user/games/{slug}/stats/web")
        item = (stats.get("data") or {}).get("item") or {}
        counts = {
            "positive": item.get("positiveCount") or 0,
            "neutral": item.get("neutralCount") or 0,
            "negative": item.get("negativeCount") or 0,
        }
        reviews: list[Review] = []
        # sentiment counts are reported for the lead platform only, so every
        # sentiment is requested anyway: other platforms may have reviews of it
        for sentiment, limit in USER_SAMPLE.items():
            payload = await self._backend(
                f"/reviews/metacritic/user/games/{slug}/web",
                offset=0,
                limit=limit,
                filterBySentiment=sentiment,
            )
            reviews.extend(r for r in parse_reviews(payload, "user") if not _is_junk(r.text))
        return ReviewSample("user", item.get("reviewCount") or sum(counts.values()), counts, reviews)

    async def critic_reviews(self, slug: str, counts: dict[str, int], total: int) -> ReviewSample:
        reviews: list[Review] = []
        for sentiment, want in CRITIC_SAMPLE.items():
            # the critic endpoint returns at most 10 items per call
            for offset in range(0, want, 10):
                payload = await self._backend(
                    f"/reviews/metacritic/critic/games/{slug}/web",
                    offset=offset,
                    limit=min(10, want - offset),
                    filterBySentiment=sentiment,
                )
                batch = parse_reviews(payload, "critic")
                reviews.extend(batch)
                if len(batch) < 10:
                    break
        return ReviewSample("critic", total, counts, reviews)


def _is_junk(text: str) -> bool:
    # one-word ratings and keyboard mashing carry no signal for a summary
    return len(text) < 25
