import gc
import re
import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Generator, Callable, Tuple

from wrapt import partial
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from groq import Groq
from groq import APIError as GroqAPIError, RateLimitError as GroqRateLimitError
import anthropic

import config

_gen_lock = asyncio.Semaphore(1)


from ml_backend.observability import (
    setup_logging,
    attach_prometheus,
    Timer,
    QUERY_COUNTER,
    STAGE_LATENCY,
)
from loguru import logger


class RAGGenerator:
    """Hosted LLM generation with provider fallback in config.GENERATION_PROVIDER_ORDER."""

    def __init__(self):
        self.clients: Dict[str, Any] = {}
        self.device = config.DEVICE

    @property
    def client(self):
        """The Claude client if loaded, else the Groq client, else None."""
        return self.clients.get("claude") or self.clients.get("groq")

    def load_model(self):
        """Initializes each provider client whose API key is set.

        Raises:
            RuntimeError: If no provider could be initialized.
        """
        try:
            self.clients["claude"] = self._init_claude_client()
        except ValueError as e:
            logger.warning(f"[Generator] claude unavailable: {e}")

        try:
            self.clients["groq"] = self._init_groq_client()
        except ValueError as e:
            logger.warning(f"[Generator] groq unavailable: {e}")

        if not self.clients:
            raise RuntimeError(
                "No generation provider could be initialized — check ANTHROPIC_API_KEY / GROQ_API_KEY."
            )
        logger.info(f"[Generator] Available providers: {list(self.clients.keys())}")

    def _init_groq_client(self) -> Groq:
        """Returns a Groq client; raises ValueError if GROQ_API_KEY is unset."""
        logger.info(f"[Generator] Connecting to Groq model: {config.GROQ_MODEL_ID} ...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        client = Groq(api_key=api_key)
        logger.info(f"Groq {config.GROQ_MODEL_ID} Model loaded successfully.")
        return client

    def _init_claude_client(self) -> "anthropic.Anthropic":
        """Returns an Anthropic client; raises ValueError if ANTHROPIC_API_KEY is unset."""
        logger.info(
            f"[Generator] Connecting to Claude model: {config.CLAUDE_MODEL_ID} ..."
        )
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Claude {config.CLAUDE_MODEL_ID} Model loaded successfully.")
        return client

    def _build_prompt_content(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Returns the (system, user) prompt strings for query and context_snippets."""
        context_block = "\n".join(format_context_snippets(context_snippets))
        system_msg = config.GROQ_SYSTEM_PROMPT
        user_msg = f"User Query: {query}\n\nContext:\n{context_block}"
        return system_msg, user_msg

    def trim_context(self, snippets, max_chars=2500, trim_length=config.TRIM_LENGTH):
        """Returns copies of snippets with text cut to trim_length, stopping at max_chars total.

        Does not mutate snippets.
        """
        trimmed = []
        total = 0

        for s in snippets:
            text = (s.get("text") or "")[:trim_length]

            if total + len(text) > max_chars:
                break

            trimmed.append({**s, "text": text})
            total += len(text)

        return trimmed

    def generate_stream(
        self,
        query: str,
        context_snippets: List[Dict[str, Any]],
        on_provider: Optional[Callable[[str, str], None]] = None,
    ) -> Generator[str, None, None]:
        """Yields answer text chunks, falling back to the next provider on a pre-first-token failure.

        Args:
            on_provider: Called with (provider, model) before the first token.

        Raises:
            Exception: The last provider error if every provider failed.
        """
        streamers = {
            "claude": self._generate_stream_claude,
            "groq": self._generate_stream_groq,
        }

        last_exc = None
        for provider in config.GENERATION_PROVIDER_ORDER:
            client = self.clients.get(provider)
            streamer = streamers.get(provider)
            if client is None or streamer is None:
                continue

            try:
                token_iter = streamer(client, query, context_snippets)
                first_token = next(token_iter)
            except StopIteration:
                return
            except Exception as e:
                logger.warning(
                    f"[Generator] {provider} failed before first token, falling back: {e}"
                )
                last_exc = e
                continue

            if on_provider is not None:
                on_provider(provider, PROVIDER_MODELS.get(provider, "unknown"))
            yield first_token
            yield from token_iter
            return

        raise last_exc or RuntimeError("No generation provider available")

    def _generate_stream_groq(
        self, client: Groq, query: str, context_snippets: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Yields text chunks from a Groq chat completion stream.

        Raises:
            RateLimitError, APIError: From the Groq API.
        """
        system_msg, user_msg = self._build_prompt_content(
            query, self.trim_context(context_snippets)
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        with Timer("generator", "groq_stream_init"):
            try:
                stream = client.chat.completions.create(
                    model=config.GROQ_MODEL_ID,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    reasoning_effort="low",
                    stream=True,
                )
            except GroqRateLimitError as e:
                logger.warning(f"Groq rate limit: {e}")
                raise
            except GroqAPIError as e:
                logger.error(f"Groq API error: {e}")
                raise

        token_count = 0
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    token_count += 1
                    yield content
            except (IndexError, AttributeError):
                continue
        logger.debug(f"Groq stream complete — {token_count} token chunks emitted")

    def _generate_stream_claude(
        self,
        client: "anthropic.Anthropic",
        query: str,
        context_snippets: List[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Yields text chunks from an Anthropic Messages stream.

        Raises:
            anthropic.RateLimitError, anthropic.APIStatusError: From the Anthropic API.
        """
        system_msg, user_msg = self._build_prompt_content(
            query, self.trim_context(context_snippets)
        )

        with Timer("generator", "claude_stream_init"):
            try:
                stream = client.messages.create(
                    model=config.CLAUDE_MODEL_ID,
                    max_tokens=config.CLAUDE_MAX_TOKENS,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_msg}],
                    extra_body={
                        "temperature": config.TEMPERATURE,
                        "top_p": config.TOP_P,
                    },
                    stream=True,
                )
            except anthropic.RateLimitError as e:
                logger.warning(f"Claude rate limit: {e}")
                raise
            except anthropic.APIStatusError as e:
                logger.error(f"Claude API error: {e}")
                raise

        token_count = 0
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                token_count += 1
                yield event.delta.text
        logger.debug(f"Claude stream complete — {token_count} token chunks emitted")

    def classify_intent(self, query: str) -> str:
        """Returns the lowercased intent label, falling back across providers.

        Raises:
            Exception: The last provider error if every provider failed.
        """
        classifiers = {
            "claude": self._classify_intent_claude,
            "groq": self._classify_intent_groq,
        }

        last_exc = None
        for provider in config.GENERATION_PROVIDER_ORDER:
            client = self.clients.get(provider)
            classifier = classifiers.get(provider)
            if client is None or classifier is None:
                continue
            try:
                return classifier(client, query)
            except Exception as e:
                logger.warning(
                    f"[Generator] {provider} intent classification failed, falling back: {e}"
                )
                last_exc = e

        raise last_exc or RuntimeError("No generation provider available")

    def _classify_intent_groq(self, client: Groq, query: str) -> str:
        """Returns the lowercased intent label from Groq."""
        resp = client.chat.completions.create(
            model=config.GROQ_INTENT_MODEL,
            max_tokens=20,
            temperature=0.0,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return resp.choices[0].message.content.strip().lower()

    def _classify_intent_claude(self, client: "anthropic.Anthropic", query: str) -> str:
        """Returns the lowercased intent label from Claude."""
        resp = client.messages.create(
            model=config.CLAUDE_INTENT_MODEL,
            max_tokens=20,
            extra_body={"temperature": 0.0},
            system=INTENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        return resp.content[0].text.strip().lower()


PROVIDER_MODELS: Dict[str, str] = {
    "claude": config.CLAUDE_MODEL_ID,
    "groq": config.GROQ_MODEL_ID,
}


def format_context_snippets(context_snippets: List[Dict[str, Any]]) -> List[str]:
    """Returns one prompt context block per snippet; also sent as the /generate `context` frame."""
    formatted = []
    for res in context_snippets:
        name = res.get("restaurant") or res.get("name") or "Unknown"
        text = res.get("text") or res.get("chunks") or res.get("content") or ""
        city = res.get("city", "")

        formatted.append(
            f"Restaurant: {name}\n"
            f"Location: {res.get('address', 'Unknown')} ({city})\n"
            f"Description: {text}\n"
            f""
        )
    return formatted


INTENT_SYSTEM_PROMPT = (
    "Classify the user query into exactly one of these intents:\n"
    "- food_search: looking for restaurants, food, or dining recommendations\n"
    "- location_only: mentions only a city/location with no food intent\n"
    "- greeting: hello, hi, hey, how are you\n"
    "- identity: asking who/what the assistant is\n"
    "- off_topic: anything else unrelated to food/restaurants\n\n"
    "Reply with ONLY the intent label, nothing else."
)


gen_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the RAGGenerator on startup and releases it on shutdown."""
    setup_logging("generator")
    gen_state["generator"] = RAGGenerator()
    logger.info("Starting generator service (Claude → Groq fallback)...")
    gen_state["generator"].load_model()
    yield
    logger.info(" Shutting Down Generator Service ")
    del gen_state["generator"]
    gc.collect()


app = FastAPI(title="RAG Generator API", lifespan=lifespan)
attach_prometheus(app, "generator")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Accel-Buffering", "Transfer-Encoding"],
)


#  Models
class GenerateRequest(BaseModel):
    """Request body for /generate."""

    query: str
    city: Optional[str] = None
    state: Optional[str] = None
    top_k: int = config.TOP_K

    include_context: bool = False


#  Helper Functions


def _safe_excerpt(text: Optional[str], max_len: int = 400) -> str:
    """Returns the text after the first "--" (or all of it), truncated to max_len."""
    if not text:
        return ""
    parts = text.split("--")
    return (parts[1] if len(parts) > 1 else parts[0]).strip()[:max_len]


def is_non_food_query(query: str) -> bool:
    """Returns True if query matches any config.NON_FOOD_PATTERNS."""
    q = query.strip().lower()
    return any(re.search(p, q) for p in config.NON_FOOD_PATTERNS)


def _area_mentioned(area: str, query: str, query_lower: str) -> bool:
    # 2-letter entries (state codes like "in" for Indiana) match case-sensitively
    # against the original query — lowercased, "in" matches the preposition "in"
    # present in almost every query.
    if len(area) == 2:
        return re.search(r"\b" + re.escape(area.upper()) + r"\b", query) is not None
    return re.search(r"\b" + re.escape(area) + r"\b", query_lower) is not None


def is_out_of_coverage(query: str) -> bool:
    """Returns True if query names an OUT_OF_COVERAGE city and no COVERED_AREAS."""
    q = query.lower()
    if any(_area_mentioned(area, query, q) for area in config.COVERED_AREAS):
        return False

    return any(_area_mentioned(city, query, q) for city in config.OUT_OF_COVERAGE)


async def fetch_context(
    query: str, top_k: int
) -> Tuple[List[Dict[str, Any]], float, bool]:
    """Queries the retriever service.

    Returns:
        (results, retrieval_ms, out_of_coverage); ([], 0.0, False) on HTTP errors.
    """
    try:
        payload = {"query": query, "top_k": top_k, "do_rerank": config.DO_RERANK}
        logger.info(f"[API] Fetching context from {config.RETRIEVER_URL}...")

        with Timer("generator", "retriever_fetch"):
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(config.RETRIEVER_URL, json=payload)
                response.raise_for_status()

        data = response.json()
        logger.info(f"[API] Context fetched successfully: {data}")

        out_of_coverage = False
        if isinstance(data, list):
            results = data
            retrieval_ms = 0.0
        else:
            results = data.get("results", [])
            retrieval_ms = data.get("retrieval_ms", 0.0)
            out_of_coverage = data.get("out_of_coverage", False)

        logger.info(f"Retrieved {len(results)} snippets for '{query[:60]}'")
        return results, retrieval_ms, out_of_coverage

    except httpx.RequestError as e:
        logger.error(f"[API] Retrieval network error: {e}")
        return [], 0.0, False

    except httpx.HTTPStatusError as e:
        logger.error(f"[API] Retriever returned {e.response.status_code}: {e}")
        return [], 0.0, False


def _early_stream(
    message: str, ndjson_extra_newline: bool = False
) -> StreamingResponse:
    """Returns an NDJSON stream (meta, empty sources, one token frame) replying with message.

    Used for replies that skip retrieval. ndjson_extra_newline ends the token frame with "\\n\\n".
    """

    def stream():
        yield json.dumps(
            {
                "type": "meta",
                "data": {"retrieval_ms": 0, "results_count": 0, "reranked": False},
            }
        ) + "\n"
        yield json.dumps({"type": "sources", "data": []}) + "\n"
        tail = "\n\n" if ndjson_extra_newline else "\n"
        yield json.dumps({"type": "token", "data": message}) + tail

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/health")
def health_check():
    """Returns available providers, their order and models; 503 if none are loaded."""
    generator = gen_state.get("generator")
    if not generator or not generator.client:
        return Response(status_code=503)

    available = list(generator.clients.keys())
    return {
        "status": "active",
        "service": "Generator",
        "providers": available,
        "provider_order": [
            p for p in config.GENERATION_PROVIDER_ORDER if p in available
        ],
        "models": {p: PROVIDER_MODELS.get(p, "unknown") for p in available},
    }


#  Endpoints
@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    """Retrieves context and streams the answer as NDJSON frames.

    Non-food, non-retrieval-intent and out-of-coverage queries get a canned reply.

    Raises:
        HTTPException: 500 if the generator is not loaded.
    """
    generator: RAGGenerator = gen_state.get("generator")
    if not generator:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Refine Query
    full_query = req.query
    if req.city:
        full_query += f" in {req.city}"

    intent = await asyncio.get_event_loop().run_in_executor(
        None, partial(generator.classify_intent, req.query)
    )
    logger.info(f"[Intent] '{req.query}' -> {intent}")

    if intent in config.NON_RETRIEVAL_INTENTS:
        return _early_stream(config.INTENT_RESPONSE_MAP.get(intent, ""))

    # Skip retrieval for non-food queries
    if is_non_food_query(req.query):
        return _early_stream(
            config.INTENT_RESPONSE_MAP["greeting"], ndjson_extra_newline=True
        )

    if is_out_of_coverage(req.query):
        return _early_stream(config.COVERAGE_MESSAGE)

    # 2. Retrieve Context
    context_results, retrieval_ms, out_of_coverage = await fetch_context(
        full_query, req.top_k
    )
    # The retriever caught an out-of-coverage location the generator missed
    if out_of_coverage:
        return _early_stream(config.COVERAGE_MESSAGE)

    logger.info(f"[API] Retrieved {len(context_results)} snippets.")
    QUERY_COUNTER.labels("generator", "started").inc()

    # 3. Stream Response
    async def response_stream():
        """Yields NDJSON frames: ping, meta, sources, context, provider, token, error."""
        # A. Send Sources First (JSON)
        # strip the heavy 'text' field for the frontend source list to save bandwidth
        for _ in range(8):
            yield json.dumps({"type": "ping"}) + "\n"
        yield json.dumps(
            {
                "type": "meta",
                "data": {
                    "retrieval_ms": retrieval_ms,
                    "results_count": len(context_results),
                    "reranked": config.DO_RERANK,
                },
            }
        ) + "\n"

        sources = [
            {
                "name": r.get("restaurant", "Unknown"),
                "address": r.get("address"),
                "lat": r.get("latitude"),
                "lon": r.get("longitude"),
                "city": r.get("city"),
                "state": r.get("state"),
                "excerpt": _safe_excerpt(r.get("text")),
            }
            for r in context_results
        ]

        yield json.dumps({"type": "sources", "data": sources}) + "\n"

        # B. Send Generation Tokens
        if not context_results:
            yield json.dumps(
                {
                    "type": "token",
                    "data": "I couldn't find any restaurants matching that description.",
                }
            ) + "\n"
            QUERY_COUNTER.labels("generator", "no_results").inc()
            return

        if req.include_context:
            prompt_snippets = format_context_snippets(
                generator.trim_context(context_results)
            )
            yield json.dumps({"type": "context", "data": prompt_snippets}) + "\n"

        # generate_stream reports the provider it commits to just before its
        # first token, so the provider frame precedes that token.
        committed: List[Dict[str, str]] = []

        def _on_provider(provider: str, model: str) -> None:
            committed.append({"provider": provider, "model": model})

        # The lock is held for the generation call itself, so it limits
        # concurrent generations rather than just endpoint entry.
        async with _gen_lock:
            try:
                for token in generator.generate_stream(
                    req.query, context_results, on_provider=_on_provider
                ):
                    if committed:
                        yield json.dumps(
                            {"type": "provider", "data": committed.pop()}
                        ) + "\n"
                    yield json.dumps({"type": "token", "data": token}) + "\n"
                    await asyncio.sleep(0)
                QUERY_COUNTER.labels("generator", "success").inc()
            except (GroqRateLimitError, anthropic.RateLimitError):
                yield json.dumps(
                    {"type": "error", "data": "Rate limit hit — try again shortly."}
                ) + "\n"
                QUERY_COUNTER.labels("generator", "rate_limit").inc()
            except (GroqAPIError, anthropic.APIStatusError) as e:
                yield json.dumps({"type": "error", "data": f"API error: {e}"}) + "\n"
                QUERY_COUNTER.labels("generator", "error").inc()
            except Exception as e:
                logger.error(f"Generation error: {e}")
                yield json.dumps({"type": "error", "data": str(e)}) + "\n"
                QUERY_COUNTER.labels("generator", "error").inc()

    return StreamingResponse(
        response_stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
            "Transfer-Encoding": "chunked",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
