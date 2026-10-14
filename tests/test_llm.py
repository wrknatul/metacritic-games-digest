import json

import httpx
import pytest

from app.llm import BudgetExceeded, LLMClient, LLMError


def reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}})


def client(handler, **kw):
    return LLMClient("key", "m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)


async def test_summary_parsed_and_usage_recorded():
    seen, usage = [], []

    def handler(request):
        seen.append(json.loads(request.content))
        return reply(json.dumps({"verdict": "Хорошо", "likes": ["сюжет"], "dislikes": []}))

    llm = client(handler, usage_hook=usage.append)
    s = await llm.summarize_reviews("Game", "user", 10, {"positive": 9}, [{"text": "great story", "score": 9}])
    assert s.verdict == "Хорошо" and s.likes == ["сюжет"]
    body = seen[0]
    assert body["response_format"]["type"] == "json_schema"
    assert "great story" in body["messages"][1]["content"]
    assert usage == [{"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}]


async def test_invalid_json_is_retried_once():
    answers = iter(["not json", "```json\n" + json.dumps({"verdict": "v", "likes": [], "dislikes": ["баги"]}) + "\n```"])
    llm = client(lambda r: reply(next(answers)))
    s = await llm.summarize_reviews("Game", "critic", 3, {}, [{"text": "buggy", "score": 40}])
    assert s.dislikes == ["баги"]


async def test_http_error_and_budget():
    llm = client(lambda r: httpx.Response(500, text="upstream down"))
    with pytest.raises(LLMError):
        await llm.summarize_reviews("Game", "critic", 1, {}, [{"text": "x", "score": 1}])

    def over_budget():
        raise BudgetExceeded("limit")

    llm = client(lambda r: reply("{}"), budget_check=over_budget)
    with pytest.raises(BudgetExceeded):
        await llm.summarize_reviews("Game", "critic", 1, {}, [{"text": "x", "score": 1}])
