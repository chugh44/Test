"""Offline checks of the Claude integration: schema conversion and response handling, using a mocked SDK client."""

from types import SimpleNamespace

import pytest
from anthropic.lib._parse._transform import to_json_output_format

from stockintel.agents.consensus.schema import LLMThesis
from stockintel.agents.news.schema import LLMNewsAnalysis
from stockintel.llm.client import LLMClient


@pytest.mark.parametrize("schema", [LLMNewsAnalysis, LLMThesis])
def test_structured_output_schemas_convert(schema):
    fmt = to_json_output_format(schema)
    assert fmt["type"] == "json_schema" and fmt["schema"]["type"] == "object"


def _usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)

    def parse(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def _client(settings, responses):
    c = LLMClient(settings, enabled=False)
    fm = FakeMessages(responses)
    c._client = SimpleNamespace(beta=SimpleNamespace(messages=fm))
    return c, fm


def test_web_search_collects_real_result_urls_and_resumes_pause(settings):
    r1 = SimpleNamespace(stop_reason="pause_turn", content=[], model="m", usage=_usage(), _request_id="r1")
    result = SimpleNamespace(type="web_search_result", url="https://reuters.com/a", title="A", page_age="2026-11-20")
    r2 = SimpleNamespace(stop_reason="end_turn", model="m", usage=_usage(), _request_id="r2", content=[
        SimpleNamespace(type="server_tool_use"),
        SimpleNamespace(type="web_search_tool_result", content=[result]),
        SimpleNamespace(type="text", text="Summary", citations=[SimpleNamespace(url="https://reuters.com/a")]),
    ])
    c, fm = _client(settings, [r1, r2])
    out = c.web_search(purpose="t", system="s", prompt="p")
    assert out.text == "Summary"
    assert [r.url for r in out.results] == ["https://reuters.com/a"]
    assert out.cited_urls == ["https://reuters.com/a"]
    assert len(fm.calls) == 2  # the paused turn was resumed
    assert fm.calls[0]["fallbacks"] == "default" and fm.calls[0]["tools"][0]["type"] == "web_search_20260209"


def test_web_search_error_block_is_not_a_crash(settings):
    err = SimpleNamespace(stop_reason="end_turn", model="m", usage=_usage(), _request_id="r", content=[
        SimpleNamespace(type="web_search_tool_result", content=SimpleNamespace(error_code="max_uses_exceeded"))])
    c, _ = _client(settings, [err])
    out = c.web_search(purpose="t", system="s", prompt="p")
    assert out.results == [] and out.text == ""


def test_structured_refusal_returns_none(settings):
    refusal = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="other"), parsed_output=None,
                              model="m", usage=_usage(), _request_id="r")
    c, fm = _client(settings, [refusal])
    assert c.structured(purpose="t", system="s", user="u", schema=LLMThesis) is None
    assert c.calls and c.calls[-1].ok is False
    assert fm.calls[0]["output_format"] is LLMThesis and fm.calls[0]["output_config"] == {"effort": "medium"}
