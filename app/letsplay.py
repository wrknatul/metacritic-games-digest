"""Let's play lookup on YouTube and getting the blogger's speech as text.

Search goes through yt-dlp (no API key). The speech is turned into text by the first
strategy that works, because YouTube throttles datacenter IPs differently for each:
  captions     - YouTube's own speech recognition track (cheap, often rate limited)
  audio        - download the first minutes of audio and run speech-to-text in Gemini
  gemini_video - give Gemini the YouTube link, Google fetches the video itself
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass

log = logging.getLogger(__name__)

SEARCH_RESULTS = 15
MIN_DURATION_S = 5 * 60
_BAD_TITLE_RE = re.compile(
    r"\b(trailer|teaser|#shorts|soundtrack|ost|reveal|review|обзор|news|things you need|before you buy|explained|tier list|reaction)\b",
    re.I,
)
# what a let's play title usually says; long videos without these words count too
_PLAY_TITLE_RE = re.compile(
    r"(let'?s ?play|gameplay|walkthrough|playthrough|full game|part \d|episode \d|ep\.? ?\d|first look|blind|прохождение|летсплей|стрим|stream|live)",
    re.I,
)
LONG_FORM_S = 20 * 60
_STOPWORDS = {"the", "of", "a", "an", "and", "in", "on", "to", "for", "edition", "remastered", "remaster"}
_TOKEN_RE = re.compile(r"[a-z0-9а-яё]+")
_CUE_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s+-->")


class LetsPlayError(Exception):
    pass


class YouTubeBlocked(LetsPlayError):
    """YouTube refused the request (bot check, 403, 429) from this IP."""


class YouTubeUnreachable(LetsPlayError):
    """No network path to YouTube right now (DNS, refused, timeout): retry later, the game is fine."""


def _is_network_error(stderr: str) -> bool:
    s = stderr.lower()
    return any(x in s for x in ("unable to download api page", "errno", "timed out", "connection refused", "name or service not known", "nodename nor servname"))


@dataclass
class VideoCandidate:
    id: str
    title: str
    channel: str | None
    views: int
    duration: float | None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("'", ""))


def _significant(title: str) -> list[str]:
    toks = [t for t in _tokens(title) if t not in _STOPWORDS]
    # sequel numbers matter ("Planet Zoo 2" is not "Planet Zoo"), single letters do not
    keep = [t for t in toks if len(t) > 1 or t.isdigit()]
    return keep or toks


def title_matches(game_title: str, video_title: str) -> bool:
    need = _significant(game_title)
    if not need:
        return False
    have = set(_tokens(video_title))
    hits = sum(1 for t in need if t in have)
    if len(need) <= 3:
        return hits == len(need)
    return hits / len(need) >= 0.75


def pick_letsplay(game_title: str, candidates: list[VideoCandidate]) -> VideoCandidate | None:
    """The most viewed video that is about this game and long enough to be a let's play."""
    fitting = [
        c
        for c in candidates
        if title_matches(game_title, c.title)
        and (c.duration or 0) >= MIN_DURATION_S
        and not _BAD_TITLE_RE.search(c.title)
        and (_PLAY_TITLE_RE.search(c.title) or (c.duration or 0) >= LONG_FORM_S)
    ]
    return max(fitting, key=lambda c: c.views, default=None)


def parse_vtt(vtt: str, max_seconds: int) -> str:
    """Plain text of a WebVTT track up to max_seconds, with rolling-caption repeats removed."""
    out: list[str] = []
    for block in vtt.split("\n\n"):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        m = next((_CUE_TIME_RE.match(l) for l in lines if "-->" in l), None)
        if not m:
            continue
        h, mi, s, _ = (int(x) for x in m.groups())
        current_start = h * 3600 + mi * 60 + s
        if current_start > max_seconds:
            break
        for line in lines:
            if "-->" in line:
                continue
            text = re.sub(r"<[^>]+>", "", line).strip()
            # auto captions roll: each line is repeated in the next one or two cues
            if text and text not in out[-3:]:
                out.append(text)
    return " ".join(out)


def _is_blocked(stderr: str) -> bool:
    s = stderr.lower()
    return any(x in s for x in ("http error 403", "http error 429", "sign in to confirm", "too many requests", "403 forbidden"))


async def _run_ytdlp(args: list[str], timeout: float, cwd: str | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-warnings",
        "--no-progress",
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise LetsPlayError(f"yt-dlp timed out after {timeout:.0f}s")
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def search(game_title: str) -> list[VideoCandidate]:
    query = f"ytsearch{SEARCH_RESULTS}:{game_title} let's play"
    code, out, err = await _run_ytdlp(["--flat-playlist", "--dump-single-json", query], timeout=90)
    if code != 0:
        if _is_blocked(err):
            raise YouTubeBlocked(err.strip()[-300:])
        if _is_network_error(err):
            raise YouTubeUnreachable(err.strip()[-300:])
        raise LetsPlayError(f"search failed: {err.strip()[-300:]}")
    data = json.loads(out)
    result = []
    for e in data.get("entries") or []:
        if not e or not e.get("id"):
            continue
        result.append(
            VideoCandidate(
                id=e["id"],
                title=e.get("title") or "",
                channel=e.get("channel") or e.get("uploader"),
                views=int(e.get("view_count") or 0),
                duration=e.get("duration"),
            )
        )
    return result


def _caption_track(info: dict) -> dict | None:
    """The speech track in the language the blogger speaks: manual subtitles, else YouTube's ASR."""
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    lang = info.get("language")
    order = []
    if lang:
        order += [(subs, lang), (auto, f"{lang}-orig"), (auto, lang)]
    order += [(subs, "en"), (auto, "en-orig"), (auto, "en"), (subs, "ru"), (auto, "ru-orig")]
    for pool, key in order:
        formats = pool.get(key)
        vtt = next((f for f in formats or [] if f.get("ext") == "vtt" and f.get("url")), None)
        if vtt:
            return vtt
    return None


def _captions_sync(url: str, max_seconds: int) -> str | None:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            track = _caption_track(info)
            if track is None:
                return None
            # one request for the one track: asking for every language is what triggers 429
            raw = ydl.urlopen(track["url"]).read().decode("utf-8", errors="replace")
    except Exception as exc:  # yt-dlp wraps HTTP errors in its own exception types
        if _is_blocked(str(exc)):
            raise YouTubeBlocked(str(exc)[-300:]) from exc
        raise LetsPlayError(f"captions failed: {str(exc)[-300:]}") from exc
    return parse_vtt(raw, max_seconds) or None


async def fetch_captions(video: VideoCandidate, minutes: int) -> str | None:
    return await asyncio.wait_for(asyncio.to_thread(_captions_sync, video.url, minutes * 60), timeout=120)


async def fetch_audio(video: VideoCandidate, minutes: int) -> bytes | None:
    """First `minutes` of the audio track as 16 kHz mono mp3."""
    with tempfile.TemporaryDirectory() as tmp:
        code, _, err = await _run_ytdlp(
            [
                "-f",
                "bestaudio[abr<=96]/bestaudio",
                "--download-sections",
                f"*0-{minutes * 60}",
                "-x",
                "--audio-format",
                "mp3",
                "--postprocessor-args",
                "ExtractAudio:-ac 1 -ar 16000 -b:a 32k",
                "-o",
                "audio.%(ext)s",
                video.url,
            ],
            timeout=300,
            cwd=tmp,
        )
        path = os.path.join(tmp, "audio.mp3")
        if code != 0 or not os.path.exists(path):
            if _is_blocked(err):
                raise YouTubeBlocked(err.strip()[-300:])
            raise LetsPlayError(f"audio download failed: {err.strip()[-300:]}")
        with open(path, "rb") as fh:
            return fh.read()
