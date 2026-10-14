"""LLM calls through an OpenAI-compatible gateway: review summaries, speech-to-text, let's play conclusions."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)



class LLMError(Exception):
    pass


class BudgetExceeded(LLMError):
    pass


class LLMUnavailable(LLMError):
    """The gateway refuses every call (bad key, no balance): waiting helps, retrying each item does not."""


class ReviewSummary(BaseModel):
    verdict: str = Field(description="1-2 sentences: the overall opinion of this group of reviewers")
    likes: list[str] = Field(description="What reviewers like, up to 5 short points")
    dislikes: list[str] = Field(description="What reviewers dislike, up to 5 short points")


class LetsPlayConclusion(BaseModel):
    conclusion: str = Field(description="3-5 sentences: what the blogger thinks of the game")
    impressions: list[str] = Field(description="Up to 5 key impressions of the blogger")
    sentiment: Literal["positive", "mixed", "negative", "neutral"]


def _json_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    return {"name": model.__name__, "strict": True, "schema": schema}


SUMMARY_SYSTEM = """Ты аналитик игровых отзывов. Тебе дают выборку отзывов {who} об игре и распределение оценок.
Составь краткое резюме на русском языке: что {who_short} нравится в игре и что не нравится.
Правила:
- опирайся только на текст отзывов, ничего не выдумывай;
- пункты короткие (до 15 слов) и конкретные: геймплей, сюжет, графика, техническое состояние, контент, цена;
- объединяй повторяющиеся мысли, самые частые ставь первыми;
- учитывай распределение оценок: если негатива почти нет, не раздувай его;
- если претензий в отзывах нет, оставь dislikes пустым; если похвалы нет, оставь likes пустым;
- если отзывов мало (меньше 5), прямо скажи об этом в verdict."""

LETSPLAY_SYSTEM = """Ты анализируешь расшифровку летсплея: блогер играет в игру и комментирует.
По расшифровке сделай заключение на русском языке: какое впечатление игра произвела на блогера,
что он хвалит и что ругает. Опирайся только на расшифровку, не выдумывай.
Если в расшифровке почти нет мнений об игре (только прохождение или болтовня не по теме), так и скажи."""

TRANSCRIBE_PROMPT = (
    "Расшифруй речь в этом аудио дословно, на языке оригинала. "
    "Верни только текст речи, без таймкодов и пометок. Если речи нет, верни пустую строку."
)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://anymodel.org/v1",
        client: httpx.AsyncClient | None = None,
        budget_check: Callable[[], None] | None = None,
        usage_hook: Callable[[dict], None] | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        # usage accounting and provider routing are OpenRouter extensions
        self.is_openrouter = "openrouter.ai" in self.base_url
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
        self._budget_check = budget_check
        self._usage_hook = usage_hook

    def set_hooks(self, budget_check: Callable[[], None], usage_hook: Callable[[dict], None]) -> None:
        self._budget_check = budget_check
        self._usage_hook = usage_hook

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _complete(
        self,
        messages: list[dict],
        *,
        response_format: dict | None = None,
        max_tokens: int = 1500,
        provider: dict | None = None,
    ) -> str:
        if not self.enabled:
            raise LLMError("LLM_API_KEY is not set")
        if self._budget_check:
            self._budget_check()
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if self.is_openrouter:
            body["usage"] = {"include": True}
        if response_format:
            body["response_format"] = response_format
        if provider and self.is_openrouter:
            body["provider"] = provider
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Title": "Metacritic Games Digest",
                },
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code in (401, 402, 403):
            raise LLMUnavailable(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code != 200:
            raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if self._usage_hook and data.get("usage"):
            self._usage_hook(data["usage"])
        if data.get("error"):
            raise LLMError(f"LLM error: {data['error']}")
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response: {str(data)[:300]}") from exc

    async def _structured(self, messages: list[dict], model: type[BaseModel], max_tokens: int = 1500) -> BaseModel:
        fmt = {"type": "json_schema", "json_schema": _json_schema(model)}
        last_err: Exception | None = None
        for _ in range(2):
            content = await self._complete(messages, response_format=fmt, max_tokens=max_tokens)
            try:
                return model.model_validate(json.loads(_strip_fences(content)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = exc
                log.warning("LLM returned invalid %s, retrying: %s", model.__name__, exc)
        raise LLMError(f"LLM returned invalid {model.__name__} twice: {last_err}")

    async def summarize_reviews(self, game_title: str, kind: str, total: int, counts: dict, reviews: list[dict]) -> ReviewSummary:
        who, who_short = ("критиков (игровых изданий)", "критикам") if kind == "critic" else ("игроков", "игрокам")
        lines = [
            f"Игра: {game_title}",
            f"Всего отзывов {who}: {total}. Распределение: положительных {counts.get('positive', 0)}, "
            f"смешанных {counts.get('neutral', 0)}, отрицательных {counts.get('negative', 0)}.",
            f"Выборка ({len(reviews)} отзывов):",
        ]
        for r in reviews:
            score = "" if r.get("score") is None else f"[{r['score']}] "
            lines.append(f"- {score}{r['text']}")
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM.format(who=who, who_short=who_short)},
            {"role": "user", "content": "\n".join(lines)},
        ]
        return await self._structured(messages, ReviewSummary)

    async def transcribe(self, audio: bytes, fmt: str = "mp3") -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                    {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": fmt}},
                ],
            }
        ]
        return (await self._complete(messages, max_tokens=8000)).strip()

    async def transcribe_youtube(self, url: str, minutes: int) -> str:
        """Speech of the first minutes of a YouTube video; Google fetches the video itself."""
        prompt = (
            f"Это летсплей. Расшифруй дословно речь блогера за первые {minutes} минут ролика, "
            "на языке оригинала. Верни только текст речи, без таймкодов, без описания происходящего на экране. "
            "Если блогер молчит, верни пустую строку."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": url, "processing": "agentic"}},
                ],
            }
        ]
        # only Gemini on Google AI Studio accepts YouTube links
        provider = {"order": ["google-ai-studio"], "allow_fallbacks": False}
        return (await self._complete(messages, max_tokens=8000, provider=provider)).strip()

    async def letsplay_conclusion(self, game_title: str, video_title: str, channel: str | None, transcript: str) -> LetsPlayConclusion:
        messages = [
            {"role": "system", "content": LETSPLAY_SYSTEM},
            {
                "role": "user",
                "content": f"Игра: {game_title}\nРолик: {video_title}\nКанал: {channel or '?'}\n\nРасшифровка:\n{transcript[:30000]}",
            },
        ]
        return await self._structured(messages, LetsPlayConclusion)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
