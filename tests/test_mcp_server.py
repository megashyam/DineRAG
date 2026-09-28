"""
Tests for mcp_server.py tools against an httpx.MockTransport (no network).

Async tools run via asyncio.run; pytest-asyncio isn't required.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import mcp_server as m


@pytest.fixture(autouse=True)
def _reset_areas_cache(monkeypatch):
    monkeypatch.setattr(m, "_areas_cache", None)
    monkeypatch.setattr(m, "_areas_cached_at", 0.0)


def _mock(monkeypatch, handler):
    """Routes tool HTTP calls through handler; returns the list of recorded requests."""
    calls = []

    def recording(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(
        m,
        "_client_factory",
        lambda timeout=m.TIMEOUT: httpx.AsyncClient(
            transport=httpx.MockTransport(recording), timeout=timeout
        ),
    )
    return calls


def _ndjson(*frames) -> bytes:
    return "".join(json.dumps(f) + "\n" for f in frames).encode()


def _hit(name="Hiro Ramen House", text="passage: positive reviews for Hiro mention:\n  great   broth", score=5.34217):
    return {
        "score": score,
        "business_id": "b1",
        "restaurant": name,
        "text": text,
        "city": "philadelphia",
        "state": "PA",
        "address": "1102 Chestnut St",
        "latitude": 39.95,
        "longitude": -75.16,
    }


# ── Registration ─────────────────────────────────────────────────────────────


def test_tools_registered_with_schemas():
    tools = {t.name: t for t in asyncio.run(m.mcp.list_tools())}
    assert set(tools) == {"search_restaurant_reviews", "ask_restaurant_question", "list_covered_areas"}

    search = tools["search_restaurant_reviews"].input_schema
    assert search["required"] == ["query"]
    assert search["properties"]["top_k"]["default"] == 5

    ask = tools["ask_restaurant_question"].input_schema
    assert ask["required"] == ["query"]
    assert {"city", "state"} <= set(ask["properties"])

    assert tools["list_covered_areas"].input_schema.get("properties", {}) == {}
    for t in tools.values():
        assert t.description
        assert t.annotations.read_only_hint is True
        assert t.output_schema is not None


def test_import_does_not_pull_in_heavy_modules():
    # config.py imports torch; generator/retriever build Prometheus-instrumented
    # FastAPI apps. The MCP server must stay a thin HTTP client.
    code = (
        "import sys, mcp_server; "
        "bad = [n for n in ('torch', 'config', 'generator_groq', 'retriever') if n in sys.modules]; "
        "assert not bad, bad"
    )
    root = Path(m.__file__).resolve().parent
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, timeout=120)


# ── search_restaurant_reviews ────────────────────────────────────────────────


def test_search_shapes_results_and_sends_expected_body(monkeypatch):
    calls = _mock(
        monkeypatch,
        lambda req: httpx.Response(200, json={"results": [_hit()], "retrieval_ms": 12.0}),
    )
    out = asyncio.run(m.search_restaurant_reviews("ramen in Philadelphia", top_k=3))

    assert out["out_of_coverage"] is False
    assert out["results"] == [
        {
            "restaurant": "Hiro Ramen House",
            "city": "philadelphia",
            "state": "PA",
            "address": "1102 Chestnut St",
            "score": 5.3422,
            "excerpt": "positive reviews for Hiro mention: great broth",
        }
    ]
    (req,) = calls
    assert req.method == "POST"
    assert str(req.url) == f"{m.RETRIEVER_URL}/retrieve"
    # no_cache is eval-only and must never be sent.
    assert json.loads(req.content) == {"query": "ramen in Philadelphia", "top_k": 3}


def test_search_truncates_long_excerpts(monkeypatch):
    long_text = "passage: " + "delicious " * 200
    _mock(monkeypatch, lambda req: httpx.Response(200, json={"results": [_hit(text=long_text)]}))
    excerpt = asyncio.run(m.search_restaurant_reviews("ramen"))["results"][0]["excerpt"]
    assert len(excerpt) <= m.EXCERPT_CHARS
    assert excerpt.endswith("…")
    assert not excerpt.startswith("passage:")


@pytest.mark.parametrize("requested,sent", [(0, 1), (-4, 1), (7, 7), (500, m.MAX_TOP_K)])
def test_search_clamps_top_k(monkeypatch, requested, sent):
    calls = _mock(monkeypatch, lambda req: httpx.Response(200, json={"results": []}))
    asyncio.run(m.search_restaurant_reviews("ramen", top_k=requested))
    assert json.loads(calls[0].content)["top_k"] == sent


def test_search_out_of_coverage_points_to_list_covered_areas(monkeypatch):
    _mock(
        monkeypatch,
        lambda req: httpx.Response(
            200,
            json={"results": [], "retrieval_ms": 0.0, "out_of_coverage": True, "detected_location": "chicago"},
        ),
    )
    out = asyncio.run(m.search_restaurant_reviews("tacos in Chicago"))
    assert out["out_of_coverage"] is True
    assert out["results"] == []
    assert out["detected_location"] == "chicago"
    assert "chicago" in out["message"] and "list_covered_areas" in out["message"]


def test_search_rejects_blank_query(monkeypatch):
    calls = _mock(monkeypatch, lambda req: httpx.Response(200, json={"results": []}))
    with pytest.raises(ToolError, match="empty"):
        asyncio.run(m.search_restaurant_reviews("   "))
    assert calls == []


# ── ask_restaurant_question ──────────────────────────────────────────────────


def test_ask_joins_tokens_and_returns_sources(monkeypatch):
    stream = _ndjson(
        {"type": "ping"},
        {"type": "ping"},
        {"type": "meta", "data": {"retrieval_ms": 850, "results_count": 1, "reranked": True}},
        {
            "type": "sources",
            "data": [
                {
                    "name": "Dalessandro's",
                    "address": "600 Wendover St",
                    "lat": 40.0,
                    "lon": -75.2,
                    "city": "philadelphia",
                    "state": "PA",
                    "excerpt": "best cheesesteak",
                }
            ],
        },
        {"type": "provider", "data": {"provider": "claude", "model": "claude-haiku-4-5"}},
        {"type": "token", "data": "Try "},
        {"type": "token", "data": "Dalessandro's."},
    )
    calls = _mock(monkeypatch, lambda req: httpx.Response(200, content=stream))
    out = asyncio.run(m.ask_restaurant_question("best cheesesteak", city="Philadelphia", state="PA"))

    assert out == {
        "answer": "Try Dalessandro's.",
        "sources": [
            {"name": "Dalessandro's", "city": "philadelphia", "state": "PA", "address": "600 Wendover St"}
        ],
    }
    (req,) = calls
    assert str(req.url) == f"{m.GENERATOR_URL}/generate"
    assert json.loads(req.content) == {"query": "best cheesesteak", "city": "Philadelphia", "state": "PA"}


def test_ask_omits_unset_location(monkeypatch):
    stream = _ndjson({"type": "token", "data": "ok"})
    calls = _mock(monkeypatch, lambda req: httpx.Response(200, content=stream))
    asyncio.run(m.ask_restaurant_question("ramen"))
    assert json.loads(calls[0].content) == {"query": "ramen"}


def test_ask_handles_early_exit_stream(monkeypatch):
    # Greeting / off-topic / out-of-coverage replies: meta, empty sources, one
    # token, no provider frame.
    stream = _ndjson(
        {"type": "meta", "data": {"retrieval_ms": 0, "results_count": 0, "reranked": False}},
        {"type": "sources", "data": []},
        {"type": "token", "data": "DineRAG currently covers Philadelphia, Tampa, ..."},
    )
    _mock(monkeypatch, lambda req: httpx.Response(200, content=stream))
    out = asyncio.run(m.ask_restaurant_question("tacos in Chicago"))
    assert out["answer"].startswith("DineRAG currently covers")
    assert out["sources"] == []


def test_ask_error_frame_becomes_tool_error(monkeypatch):
    stream = _ndjson(
        {"type": "sources", "data": []},
        {"type": "error", "data": "All generation providers failed"},
    )
    _mock(monkeypatch, lambda req: httpx.Response(200, content=stream))
    with pytest.raises(ToolError, match="All generation providers failed"):
        asyncio.run(m.ask_restaurant_question("ramen"))


def test_ask_error_frame_reaches_client_as_is_error(monkeypatch):
    stream = _ndjson({"type": "error", "data": "boom"})
    _mock(monkeypatch, lambda req: httpx.Response(200, content=stream))
    # MCPServer.call_tool raises the ToolError; the protocol handler turns it
    # into an is_error result carrying the same message.
    with pytest.raises(ToolError, match="boom"):
        asyncio.run(m.mcp.call_tool("ask_restaurant_question", {"query": "ramen"}))


def test_ask_empty_answer_is_tool_error(monkeypatch):
    _mock(monkeypatch, lambda req: httpx.Response(200, content=_ndjson({"type": "ping"})))
    with pytest.raises(ToolError, match="empty answer"):
        asyncio.run(m.ask_restaurant_question("ramen"))


# ── list_covered_areas ───────────────────────────────────────────────────────


def test_covered_areas_cached(monkeypatch):
    calls = _mock(
        monkeypatch,
        lambda req: httpx.Response(200, json={"cities": ["philadelphia", "tampa"], "states": ["FL", "PA"]}),
    )
    first = asyncio.run(m.list_covered_areas())
    second = asyncio.run(m.list_covered_areas())

    assert first == second == {"cities": ["philadelphia", "tampa"], "states": ["FL", "PA"]}
    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert str(calls[0].url) == f"{m.RETRIEVER_URL}/debug/cities"


def test_covered_areas_refetched_after_ttl(monkeypatch):
    calls = _mock(monkeypatch, lambda req: httpx.Response(200, json={"cities": [], "states": []}))
    asyncio.run(m.list_covered_areas())
    m._areas_cached_at -= m.AREAS_TTL_S + 1
    asyncio.run(m.list_covered_areas())
    assert len(calls) == 2


def test_covered_areas_uses_longer_timeout(monkeypatch):
    seen = []

    def factory(timeout=m.TIMEOUT):
        seen.append(timeout)
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"cities": [], "states": []}))
        )

    monkeypatch.setattr(m, "_client_factory", factory)
    asyncio.run(m.list_covered_areas())
    assert seen == [m.AREAS_TIMEOUT]
    assert m.AREAS_TIMEOUT >= m.TIMEOUT


def test_covered_areas_failure_not_cached(monkeypatch):
    responses = iter([httpx.Response(500), httpx.Response(200, json={"cities": ["reno"], "states": ["NV"]})])
    _mock(monkeypatch, lambda req: next(responses))
    with pytest.raises(ToolError):
        asyncio.run(m.list_covered_areas())
    assert asyncio.run(m.list_covered_areas())["cities"] == ["reno"]


# ── Friendly errors ──────────────────────────────────────────────────────────


def _raise(exc_type):
    def handler(req):
        raise exc_type("simulated", request=req)

    return handler


@pytest.mark.parametrize("exc_type", [httpx.ReadTimeout, httpx.ConnectTimeout])
def test_timeouts_become_friendly_errors(monkeypatch, exc_type):
    _mock(monkeypatch, _raise(exc_type))
    with pytest.raises(ToolError, match="retriever timed out.*retry"):
        asyncio.run(m.search_restaurant_reviews("ramen"))


def test_connection_error_becomes_friendly_error(monkeypatch):
    _mock(monkeypatch, _raise(httpx.ConnectError))
    with pytest.raises(ToolError, match="generator is cold-starting or unavailable"):
        asyncio.run(m.ask_restaurant_question("ramen"))


def test_retriever_503_means_not_ready(monkeypatch):
    _mock(monkeypatch, lambda req: httpx.Response(503, json={"detail": "Retriever not initialized"}))
    with pytest.raises(ToolError, match="retriever is not ready"):
        asyncio.run(m.search_restaurant_reviews("ramen"))


@pytest.mark.parametrize("status", [500, 502, 504])
def test_5xx_becomes_friendly_error(monkeypatch, status):
    _mock(monkeypatch, lambda req: httpx.Response(status, text="<html>upstream</html>"))
    with pytest.raises(ToolError, match=f"generator is cold-starting or unavailable \\(HTTP {status}\\)"):
        asyncio.run(m.ask_restaurant_question("ramen"))


def test_4xx_surfaces_detail(monkeypatch):
    _mock(monkeypatch, lambda req: httpx.Response(422, json={"detail": "query: field required"}))
    with pytest.raises(ToolError, match="HTTP 422.*field required"):
        asyncio.run(m.ask_restaurant_question("ramen"))
