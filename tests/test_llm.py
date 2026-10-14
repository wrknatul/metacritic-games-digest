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


FAKE_CLAUDE = """#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
schema = json.loads(args[args.index("--json-schema") + 1]) if "--json-schema" in args else None
assert "--tools" in args and args[args.index("--tools") + 1] == ""
out = {"type": "result", "subtype": "success", "is_error": False, "result": "текст",
       "usage": {"input_tokens": 7, "cache_creation_input_tokens": 100, "output_tokens": 20}}
if schema:
    out["structured_output"] = {"verdict": "ок: " + prompt.split(chr(10))[0], "likes": ["сюжет"], "dislikes": []}
print(json.dumps(out, ensure_ascii=False))
"""


async def test_claude_cli_backend_uses_json_schema_and_records_usage(tmp_path):
    from app.llm import ClaudeCLIClient

    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    usage = []
    llm = ClaudeCLIClient("haiku", binary=str(fake), usage_hook=usage.append)
    assert llm.enabled and llm.model == "claude-code:haiku"
    s = await llm.summarize_reviews("Game", "user", 3, {"positive": 3}, [{"text": "great story", "score": 9}])
    assert s.verdict == "ок: Игра: Game" and s.likes == ["сюжет"]
    assert usage == [{"prompt_tokens": 107, "completion_tokens": 20, "cost": 0.0}]
    with pytest.raises(LLMError):
        await llm.transcribe(b"audio")  # no audio through the CLI: captions are used instead
    await llm.aclose()


async def test_english_answer_is_retried_in_russian():
    answers = iter([
        json.dumps({"verdict": "Critics really love this game", "likes": ["story"], "dislikes": []}),
        json.dumps({"verdict": "Критикам нравится", "likes": ["сюжет"], "dislikes": []}),
    ])
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return reply(next(answers))

    s = await client(handler).summarize_reviews("Game", "critic", 3, {}, [{"text": "great story", "score": 90}])
    assert s.verdict == "Критикам нравится"
    assert "по-русски" in seen[1]["messages"][-1]["content"]


def test_russian_check_is_per_field():
    from app.llm import is_russian_text

    assert is_russian_text(["Критики хвалят кампанию", "Атмосфера", "Gears of War"])
    assert not is_russian_text(["Critics praise the campaign a lot", "Атмосферная кампания"])
