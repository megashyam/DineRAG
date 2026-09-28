"""
Stdio MCP server exposing the deployed DineRAG retriever and generator as tools.

Proxies to the services over HTTP; loads no models or indexes. Logs go to
stderr (stdout carries the protocol).

Run:  python ml_backend/mcp_server.py
Env:  DINERAG_RETRIEVER_URL, DINERAG_GENERATOR_URL (base URLs), DINERAG_TIMEOUT (s)
"""

# Doesn't import config.py (loads torch) or generator_groq.py / retriever.py
# (each calls attach_prometheus(); a second app per process raises
# "Duplicated timeseries").
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
# pydantic requires typing_extensions' TypedDict on Python < 3.12.
from typing_extensions import NotRequired, TypedDict

logger = logging.getLogger("dinerag.mcp")
# httpx logs every request at INFO; keep the server's stderr readable.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Base URLs, unlike config.RETRIEVER_URL, which includes the /retrieve path.
RETRIEVER_URL = os.environ.get(
    "DINERAG_RETRIEVER_URL", "https://megumind6172--food-rag-retriever-serve.modal.run"
).rstrip("/")
GENERATOR_URL = os.environ.get(
    "DINERAG_GENERATOR_URL", "https://megumind6172--food-rag-generator-serve.modal.run"
).rstrip("/")
# Modal cold starts have been observed at ~15-35s; leave headroom on top.
TIMEOUT = float(os.environ.get("DINERAG_TIMEOUT", "90"))

MAX_TOP_K = 20  # the service default (config.TOP_K=30) is too much context for a client LLM
EXCERPT_CHARS = 600
# /debug/cities scrolls the whole Qdrant collection (~30-65s observed), so it
# gets a longer timeout and its result is cached in-process.
AREAS_TTL_S = 3600
AREAS_TIMEOUT = max(TIMEOUT, 180.0)

_areas_cache: Optional["CoveredAreas"] = None
_areas_cached_at = 0.0

mcp = MCPServer(
    "dinerag",
    instructions=(
        "DineRAG answers restaurant questions from Yelp reviews, but only for a "
        "fixed set of US cities. Use search_restaurant_reviews for raw review "
        "evidence, ask_restaurant_question for a written recommendation, and "
        "list_covered_areas when unsure whether a city is covered."
    ),
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)


# Typed returns give clients a real output schema (a bare dict gets wrapped
# as an untyped {"result": {...}}).
class ReviewHit(TypedDict):
    restaurant: str
    city: Optional[str]
    state: Optional[str]
    address: Optional[str]
    score: float
    excerpt: str


class SearchResult(TypedDict):
    out_of_coverage: bool
    results: List[ReviewHit]
    detected_location: NotRequired[Optional[str]]
    message: NotRequired[str]


class Source(TypedDict):
    name: str
    city: Optional[str]
    state: Optional[str]
    address: Optional[str]


class Answer(TypedDict):
    answer: str
    sources: List[Source]


class CoveredAreas(TypedDict):
    cities: List[str]
    states: List[str]


def _client_factory(timeout: float = TIMEOUT) -> httpx.AsyncClient:
    """Returns an httpx client; patched in tests to use a MockTransport."""
    return httpx.AsyncClient(timeout=timeout)


# ── HTTP error handling ──────────────────────────────────────────────────────


@asynccontextmanager
async def _service_call(service: str):
    """Converts httpx timeouts and transport errors into retry-hint ToolErrors."""
    try:
        yield
    except ToolError:
        raise
    except httpx.TimeoutException:
        raise ToolError(
            f"DineRAG {service} timed out (it may be cold-starting); retry in ~30s."
        ) from None
    except httpx.TransportError as e:
        raise ToolError(
            f"DineRAG {service} is cold-starting or unavailable ({type(e).__name__}); retry in ~30s."
        ) from None


async def _check_status(service: str, resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    if resp.status_code == 503:
        raise ToolError(f"DineRAG {service} is not ready yet (still starting up); retry in ~30s.")
    if resp.status_code >= 500:
        raise ToolError(
            f"DineRAG {service} is cold-starting or unavailable (HTTP {resp.status_code}); retry in ~30s."
        )
    # Streamed responses aren't read yet; the body is small for a 4xx.
    await resp.aread()
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    raise ToolError(f"DineRAG {service} rejected the request (HTTP {resp.status_code}): {detail}")


def _require_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ToolError("query must not be empty.")
    return query


def _clean_excerpt(text: Optional[str]) -> str:
    """Strips the "passage: " prefix, collapses whitespace, truncates to EXCERPT_CHARS."""
    text = text or ""
    if text.startswith("passage: "):
        text = text[len("passage: "):]
    text = " ".join(text.split())
    if len(text) > EXCERPT_CHARS:
        text = text[: EXCERPT_CHARS - 1].rstrip() + "…"
    return text


# ── Tools ────────────────────────────────────────────────────────────────────
# Docstrings are the tool descriptions clients see. Admin endpoints and the
# eval-only no_cache flag are intentionally not exposed.


@mcp.tool(annotations=_READ_ONLY)
async def search_restaurant_reviews(query: str, top_k: int = 5) -> SearchResult:
    """Search Yelp restaurant reviews (hybrid dense + BM25, reranked).

    Returns matching restaurants with name, city, state, address, score and a
    review excerpt. Include the location in the query, e.g. "late night ramen
    in Philadelphia". Only some US cities are covered; see list_covered_areas.

    Args:
        query: Natural-language search, ideally including the city.
        top_k: Number of results to return (1-20, default 5).
    """
    query = _require_query(query)
    top_k = max(1, min(MAX_TOP_K, int(top_k)))

    async with _service_call("retriever"):
        async with _client_factory() as client:
            resp = await client.post(
                f"{RETRIEVER_URL}/retrieve", json={"query": query, "top_k": top_k}
            )
            await _check_status("retriever", resp)
            data = resp.json()

    if data.get("out_of_coverage"):
        location = data.get("detected_location") or "that location"
        return {
            "out_of_coverage": True,
            "detected_location": data.get("detected_location"),
            "results": [],
            "message": (
                f"DineRAG has no review data for '{location}'. "
                "Call list_covered_areas to see which cities and states are covered."
            ),
        }

    results = [
        {
            "restaurant": r.get("restaurant", "Unknown"),
            "city": r.get("city"),
            "state": r.get("state"),
            "address": r.get("address"),
            "score": round(float(r.get("score") or 0.0), 4),
            "excerpt": _clean_excerpt(r.get("text")),
        }
        for r in data.get("results", [])[:top_k]
    ]
    return {"out_of_coverage": False, "results": results}


@mcp.tool(annotations=_READ_ONLY)
async def ask_restaurant_question(
    query: str, city: Optional[str] = None, state: Optional[str] = None
) -> Answer:
    """Answer a restaurant question from Yelp reviews (retrieval + LLM), with cited restaurants.

    Slower than search_restaurant_reviews. Uncovered locations get a coverage
    notice instead of an answer.

    Args:
        query: The question, e.g. "best cheesesteak for a late dinner".
        city: Optional city to scope the search to, e.g. "Philadelphia".
        state: Optional two-letter state code, e.g. "PA".
    """
    query = _require_query(query)
    body: Dict[str, Any] = {"query": query}
    if city:
        body["city"] = city
    if state:
        body["state"] = state

    answer_parts: List[str] = []
    sources: List[Source] = []

    async with _service_call("generator"):
        async with _client_factory() as client:
            async with client.stream("POST", f"{GENERATOR_URL}/generate", json=body) as resp:
                await _check_status("generator", resp)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line)
                    except ValueError:
                        logger.warning("Skipping malformed generator frame: %r", line[:200])
                        continue
                    ftype = frame.get("type")
                    if ftype == "token":
                        answer_parts.append(frame.get("data") or "")
                    elif ftype == "sources":
                        sources = [
                            {
                                "name": s.get("name", "Unknown"),
                                "city": s.get("city"),
                                "state": s.get("state"),
                                "address": s.get("address"),
                            }
                            for s in frame.get("data") or []
                        ]
                    elif ftype == "error":
                        raise ToolError(f"DineRAG generator failed: {frame.get('data')}")

    answer = "".join(answer_parts).strip()
    if not answer:
        raise ToolError("DineRAG generator returned an empty answer; retry the question.")
    return {"answer": answer, "sources": sources}


@mcp.tool(annotations=_READ_ONLY)
async def list_covered_areas() -> CoveredAreas:
    """List the cities (lowercase) and states (two-letter codes) with review data.

    Use when unsure whether a location is covered or after an out_of_coverage result.
    """
    global _areas_cache, _areas_cached_at
    if _areas_cache is not None and time.monotonic() - _areas_cached_at < AREAS_TTL_S:
        return _areas_cache

    async with _service_call("retriever"):
        async with _client_factory(timeout=AREAS_TIMEOUT) as client:
            resp = await client.get(f"{RETRIEVER_URL}/debug/cities")
            await _check_status("retriever", resp)
            data = resp.json()

    _areas_cache = {"cities": data.get("cities", []), "states": data.get("states", [])}
    _areas_cached_at = time.monotonic()
    return _areas_cache


if __name__ == "__main__":
    mcp.run()
