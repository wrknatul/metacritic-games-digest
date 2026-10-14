"""In-memory stand-ins for Metacritic and the LLM, so pipeline tests need no network."""

from app.llm import LetsPlayConclusion, ReviewSummary
from app.metacritic import GameDetails, PlatformInfo, Review, ReviewSample


class FakeMetacritic:
    def __init__(self, new_releases, pages):
        self._new = new_releases
        self._pages = pages  # {page_number: [slugs]}
        self.detail_calls = []

    async def new_releases(self):
        return list(self._new)

    async def browse_page(self, page):
        return list(self._pages.get(page, []))

    async def game_details(self, slug):
        self.detail_calls.append(slug)
        return GameDetails(
            slug=slug,
            title=slug.replace("-", " ").title(),
            cover_url=f"https://img.test/{slug}.jpg",
            developer="Studio " + slug[0].upper(),
            publisher=None,
            description=f"{slug} is a game about {slug.split('-')[0]}",
            release_date="2026-10-14",
            genres=["Action"] if slug[0] < "m" else ["Puzzle"],
            franchise=None,
            video_url=None,
            video_title=None,
            platforms=[
                PlatformInfo("PC", "pc", True, 70 + len(slug) % 20, 10, 7.5, 40),
                PlatformInfo("PlayStation 5", "playstation-5", False, None, 0, None, 0),
            ],
            critic_total=10,
            critic_counts={"positive": 8, "neutral": 2, "negative": 0},
        )

    async def critic_reviews(self, slug, counts, total):
        return ReviewSample("critic", total, counts, [Review(f"{slug} critic text number {i}", 80, "Pub", "PC", f"{slug}-c{i}") for i in range(3)])

    async def user_reviews(self, slug):
        return ReviewSample("user", 40, {"positive": 30, "neutral": 5, "negative": 5}, [Review(f"{slug} user text number {i}", 8, "u", "PC", f"{slug}-u{i}") for i in range(3)])


class FakeLLM:
    model = "fake/model"

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.summary_calls = 0

    async def summarize_reviews(self, title, kind, total, counts, reviews):
        self.summary_calls += 1
        return ReviewSummary(verdict=f"{kind} like {title}", likes=["gameplay"], dislikes=[])

    async def transcribe(self, audio, fmt="mp3"):
        return "speech " * 100

    async def transcribe_youtube(self, url, minutes):
        return "speech " * 100

    async def letsplay_conclusion(self, *a):
        return LetsPlayConclusion(conclusion="ok", impressions=[], sentiment="positive")

    async def aclose(self):
        pass
