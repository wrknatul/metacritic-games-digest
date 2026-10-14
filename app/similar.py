"""Similar games among the ones already in the database.

Score = 0.55 * TF-IDF cosine of descriptions + 0.30 * genre overlap
      + 0.10 * same franchise + 0.05 * same developer.
Deterministic, explainable (each match carries its reasons) and needs no API calls.
The index is rebuilt in memory when the set of games changes.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z][a-z0-9']+")
_STOP = set(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further game games
    get gets had has have having he her here hers him his how i if in into is it its itself just let lets
    like make makes me more most my new no nor not now of off on once one only or other our out over own
    play player players playing same she should so some such take than that the their them then there these
    they this those through to too under until up us very was way we well were what when where which while
    who whom why will with world would you your yours""".split()
)

W_TEXT, W_GENRE, W_FRANCHISE, W_DEVELOPER = 0.55, 0.30, 0.10, 0.05

# Metacritic genres are short labels: "FPS" and "Third Person Shooter" share nothing as strings,
# so genres are also compared word by word, with abbreviations expanded
GENRE_SYNONYMS = {
    "fps": "first person shooter",
    "tps": "third person shooter",
    "rpg": "role playing",
    "jrpg": "japanese role playing",
    "arpg": "action role playing",
    "mmorpg": "massively multiplayer role playing",
    "rts": "real time strategy",
    "moba": "multiplayer battle arena",
    "4x": "strategy",
    "roguelike": "roguelike rogue",
    "roguelite": "roguelike rogue",
    "metroidvania": "metroidvania platformer action",
}
GENRE_STOP = {"person", "first", "third", "game", "games", "general", "misc", "miscellaneous", "and", "the", "of"}


def _genre_words(genres: list[str]) -> set[str]:
    words: set[str] = set()
    for g in genres:
        for w in _TOKEN_RE.findall(g.lower().replace("-", " ")) + [g.lower()]:
            words.update(GENRE_SYNONYMS.get(w, w).split())
    return {w for w in words if w not in GENRE_STOP and len(w) > 2}


@dataclass
class GameFeatures:
    id: int
    slug: str
    title: str
    description: str
    genres: list[str]
    franchise: str | None
    developer: str | None


@dataclass
class SimilarGame:
    id: int
    slug: str
    title: str
    score: float
    reasons: list[str]


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


class SimilarityIndex:
    def __init__(self, games: list[GameFeatures]):
        self.games = {g.id: g for g in games}
        docs = {g.id: Counter(_tokens(f"{g.title} {g.description}")) for g in games}
        df: Counter = Counter()
        for tf in docs.values():
            df.update(tf.keys())
        n = max(len(docs), 1)
        idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vectors: dict[int, dict[str, float]] = {}
        for gid, tf in docs.items():
            vec = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vectors[gid] = {t: v / norm for t, v in vec.items()}

    def similar(self, game_id: int, k: int = 6, min_score: float = 0.08) -> list[SimilarGame]:
        me = self.games.get(game_id)
        if me is None:
            return []
        my_vec = self.vectors.get(game_id, {})
        my_genres = {g.lower() for g in me.genres}
        my_genre_words = _genre_words(me.genres)
        scored: list[SimilarGame] = []
        for other in self.games.values():
            if other.id == game_id:
                continue
            reasons = []
            vec = self.vectors.get(other.id, {})
            small, big = (my_vec, vec) if len(my_vec) < len(vec) else (vec, my_vec)
            cos = sum(v * big.get(t, 0.0) for t, v in small.items())
            other_genres = {g.lower() for g in other.genres}
            union = my_genres | other_genres
            genre = len(my_genres & other_genres) / len(union) if union else 0.0
            if genre:
                shared = [g for g in other.genres if g.lower() in my_genres]
                reasons.append("жанр: " + ", ".join(shared))
            else:
                other_words = _genre_words(other.genres)
                common = my_genre_words & other_words
                if common:
                    # related genres count less than the same genre
                    genre = 0.6 * len(common) / len(my_genre_words | other_words)
                    reasons.append(f"похожий жанр: {', '.join(other.genres)}")
            franchise = 1.0 if me.franchise and me.franchise == other.franchise else 0.0
            if franchise:
                reasons.append(f"серия: {me.franchise}")
            developer = 1.0 if me.developer and me.developer == other.developer else 0.0
            if developer:
                reasons.append(f"разработчик: {me.developer}")
            if cos >= 0.15:
                reasons.append("похожее описание")
            score = W_TEXT * cos + W_GENRE * genre + W_FRANCHISE * franchise + W_DEVELOPER * developer
            if score >= min_score:
                scored.append(SimilarGame(other.id, other.slug, other.title, round(score, 3), reasons))
        scored.sort(key=lambda s: (-s.score, s.title))
        return scored[:k]


class SimilarityCache:
    """Keeps one index per data version so card views do not rebuild it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._version = -1
        self._index: SimilarityIndex | None = None

    def get(self, version: int, loader) -> SimilarityIndex:
        with self._lock:
            if self._index is None or version != self._version:
                self._index = SimilarityIndex(loader())
                self._version = version
            return self._index
