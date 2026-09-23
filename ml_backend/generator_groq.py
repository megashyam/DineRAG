import gc
import re
from threading import Thread
import logging
import os

from typing import List, Dict, Any, Generator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from wrapt import partial
import config
import requests
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import torch

from groq import Groq, APIError, RateLimitError

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
    """
    Service class responsible for managing LLM generation logic.
    Supports both a local, quantized Qwen model and external Groq API inference
    for Retrieval-Augmented Generation.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.client = None
        self.device = config.DEVICE

    def load_model_qwen_local(self):
        """
        Loads the specified Qwen model and tokenizer locally into memory
        using HuggingFace Transformers and BitsAndBytes quantization.

        Raises:
            Exception: If model loading fails due to OOM or missing weights.
        """
        logger.info(f"[Generator] Loading model: {config.MODEL_ID} on {self.device}...")

        try:
            bnb_config = BitsAndBytesConfig(**config.BNB_CONFIG)

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_ID, use_fast=True
            )

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                quantization_config=bnb_config,
                attn_implementation="sdpa",
                device_map="auto",
            )
            self.model.eval()
            logger.info("[Generator] Model loaded successfully.")
        except Exception as e:
            logger.error(f"[Generator] Critical Error loading model: {e}")
            raise e

    def load_model_groq(self):
        """
        Initializes the Groq client for external inference using
        the configured Groq API key and model ID.

        Raises:
            ValueError: If the GROQ_API_KEY environment variable is not set.
        """
        print(f"[Generator] Loading model: Groq-{config.GROQ_MODEL_ID} ...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)
        logger.info(f"Groq {config.GROQ_MODEL_ID} Model loaded successfully.")

    def _build_prompt_qwen_local(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Constructs the system and user prompts using retrieved context
        and tokenizes them for the local Qwen model.

        Args:
            query (str): The user's query string.
            context_snippets (List[Dict[str, Any]]): Retrieved snippets from the vector database.

        Returns:
            torch.Tensor: A tensor dictionary containing `input_ids` and `attention_mask`.
        """

        # 1. Format Context
        context_text_list = []
        for res in context_snippets:
            # Fallbacks for messy data keys
            name = res.get("restaurant") or res.get("name") or "Unknown"
            text = res.get("text") or res.get("chunks") or res.get("content") or ""
            # text = text[: config.MAX_SNIPPET_CHARS]
            city = res.get("city", "")

            snippet = (
                f"Restaurant: {name}\n"
                f"Location: {res.get('address', 'Unknown')} ({city})\n"
                f"Description: {text}\n"
                f"---"
            )
            context_text_list.append(snippet)

        context_block = "\n".join(context_text_list)

        # 2. System Instruction
        system_msg = config.QWEN_SYSTEM_PROMPT

        user_msg = f"User Query: {query}\n\nContext:\n{context_block}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # 3. Apply Template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            # truncation=True,
            # max_length=config.MAX_INPUT_TOKENS,
        ).to(self.device)

    def _build_prompt_groq(self, query: str, context_snippets: List[Dict[str, Any]]):
        """
        Constructs the chat message array for the Groq API, embedding
        the context into the user's prompt.

        Args:
            query (str): The user's query string.
            context_snippets (List[Dict[str, Any]]): Retrieved snippets from the vector database.

        Returns:
            List[Dict[str, str]]: A list of message dictionaries compatible with the OpenAI/Groq chat API format.
        """

        # 1. Format Context
        context_text_list = []
        for res in context_snippets:
            # Fallbacks for messy data keys
            name = res.get("restaurant") or res.get("name") or "Unknown"
            text = res.get("text") or res.get("chunks") or res.get("content") or ""
            # text = text[: config.MAX_SNIPPET_CHARS]
            city = res.get("city", "")

            snippet = (
                f"Restaurant: {name}\n"
                f"Location: {res.get('address', 'Unknown')} ({city})\n"
                f"Description: {text}\n"
                f"---"
            )
            context_text_list.append(snippet)

        context_block = "\n".join(context_text_list)

        # 2. System Instruction
        system_msg = config.GROQ_SYSTEM_PROMPT

        user_msg = f"User Query: {query}\n\nContext:\n{context_block}"

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def _run_generation_qwen_local(self, **kwargs):
        """
        Executes local generation.

        Args:
            **kwargs: Arbitrary keyword arguments passed to `model.generate()`.
        """

        with torch.no_grad():
            self.model.generate(**kwargs)

    def generate_stream_qwen(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """
        Streams tokens from the local LLM in a separate thread.

        Args:
            query (str): The user's query string.
            context_snippets (List[Dict[str, Any]]): Context chunks to include in the prompt.

        Yields:
            str: Next token(s) generated by the model.
        """

        # Cleanup before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        inputs = self._build_prompt(query, self.trim_context(context_snippets))

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            repetition_penalty=1.1,  # Slightly higher penalty for better lists
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=True,
        )

        thread = Thread(target=self._run_generation, kwargs=generation_kwargs)
        thread.start()

        try:

            for new_text in streamer:
                yield new_text

        finally:
            thread.join()
            del generation_kwargs, inputs, streamer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def trim_context(self, snippets, max_chars=2500, trim_length=config.TRIM_LENGTH):
        """
        Trims context snippets to prevent exceeding maximum token limits.

        Args:
            snippets (List[Dict[str, Any]]): List of context dictionaries.
            max_chars (int, optional): The maximum total characters allowed. Defaults to 2500.
            trim_length (int, optional): The maximum length for a single snippet. Defaults to config.TRIM_LENGTH.

        Returns:
            List[Dict[str, Any]]: A list of trimmed snippet dictionaries.
        """
        trimmed = []
        total = 0

        for s in snippets:
            text = (s.get("text") or "")[:trim_length]

            if total + len(text) > max_chars:
                break

            s["text"] = text
            trimmed.append(s)
            total += len(text)

        return trimmed

    def generate_stream_groq(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """
        Streams tokens from the Groq API completion endpoint.

        Args:
            query (str): The user query.
            context_snippets (List[Dict[str, Any]]): Relevant context snippets.

        Yields:
            str: Text chunks returned dynamically from the API stream.

        Raises:
            RuntimeError: If the Groq client is not initialized.
            RateLimitError: If the Groq API rate limits are exceeded.
            APIError: If a general Groq API error occurs.
        """

        if self.client is None:
            raise RuntimeError("Groq client not initialized")

        messages = self._build_prompt_groq(query, self.trim_context(context_snippets))

        with Timer("generator", "groq_stream_init"):
            try:
                stream = self.client.chat.completions.create(
                    model=config.GROQ_MODEL_ID,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    # max_completion_tokens=config.MAX_NEW_TOKENS,
                    reasoning_effort="none",
                    stream=True,
                )
            except RateLimitError as e:
                logger.warning(f"Groq rate limit: {e}")
                raise
            except APIError as e:
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
        logger.debug(f"Stream complete — {token_count} token chunks emitted")


# --- Lifespan Management ---
gen_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the FastAPI application startup and shutdown events,
    specifically loading the Groq model configuration upon startup.
    """
    setup_logging("generator")
    gen_state["generator"] = RAGGenerator()
    logger.info("Starting generator service (Groq)...")
    gen_state["generator"].load_model_groq()
    yield
    logger.info("--- Shutting Down Generator Service ---")
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


# --- Models ---
class GenerateRequest(BaseModel):
    """
    Pydantic schema representing the payload for generation requests.
    """

    query: str
    city: Optional[str] = None
    state: Optional[str] = None
    top_k: int = config.TOP_K


# --- Helper Functions ---


def _safe_excerpt(text: Optional[str], max_len: int = 400) -> str:
    if not text:
        return ""
    parts = text.split("--")
    return (parts[1] if len(parts) > 1 else parts[0]).strip()[:max_len]


def is_non_food_query(query: str) -> bool:
    """Determines if a query is likely non-food related based on regex patterns."""
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
    """checks if the query mentions any covered areas first, then checks for out of coverage cities."""
    q = query.lower()
    if any(_area_mentioned(area, query, q) for area in config.COVERED_AREAS):
        return False

    return any(_area_mentioned(city, query, q) for city in config.OUT_OF_COVERAGE)


def classify_intent(client, query: str) -> dict:
    """Returns intent classification using a tiny Groq call."""
    resp = client.chat.completions.create(
        model=config.GROQ_INTENT_MODEL,  # fastest/cheapest model, not your main one
        max_tokens=20,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user query into exactly one of these intents:\n"
                    "- food_search: looking for restaurants, food, or dining recommendations\n"
                    "- location_only: mentions only a city/location with no food intent\n"
                    "- greeting: hello, hi, hey, how are you\n"
                    "- identity: asking who/what the assistant is\n"
                    "- off_topic: anything else unrelated to food/restaurants\n\n"
                    "Reply with ONLY the intent label, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    intent = resp.choices[0].message.content.strip().lower()
    return intent


def fetch_context(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Calls the separate Retriever Microservice to fetch relevant data.

    Args:
        query (str): The specific search query to execute.
        top_k (int): Number of top results to retrieve.

    Returns:
        tuple: A tuple containing a list of result dictionaries and the retrieval latency in ms.
    """

    try:
        payload = {"query": query, "top_k": top_k, "do_rerank": config.DO_RERANK}
        logger.info(f"[API] Fetching context from {config.RETRIEVER_URL}...")
        with Timer("generator", "retriever_fetch"):
            response = requests.post(config.RETRIEVER_URL, json=payload, timeout=60)
            response.raise_for_status()
            logger.info(f"[API] Context fetched successfully: {response.json()}")

        data = response.json()

        if isinstance(data, list):
            results = data
            retrieval_ms = 0
        else:
            results = data.get("results", [])
            retrieval_ms = data.get("retrieval_ms", 0)

        logger.info(f"Retrieved {len(data)} snippets for '{query[:60]}'")
        return results, retrieval_ms
    except requests.exceptions.RequestException as e:
        print(f"[API] Retrieval Error: {e}")

        return [], 0


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify service and Groq client availability.

    Returns:
        Dict: A dictionary mapping the active status of the service.
    """
    generator = gen_state.get("generator")
    if not generator or not generator.client:
        from fastapi import Response

        return Response(status_code=503)

    return {"status": "active", "service": "Groq Generator"}


# --- Endpoints ---
@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    """
    Main endpoint for generating LLM responses. Orchestrates fetching context
    from the retriever service, appending it to the prompt, and streaming back
    a continuous NDJSON chunked response.

    Args:
        req (GenerateRequest): The incoming request payload containing the query and settings.

    Returns:
        StreamingResponse: An NDJSON stream of retrieval metadata, sources, and generated tokens.

    Raises:
        HTTPException: If the generator model is not loaded in state.
    """
    generator: RAGGenerator = gen_state.get("generator")
    if not generator:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Refine Query
    full_query = req.query
    if req.city:
        full_query += f" in {req.city}"

    intent = await asyncio.get_event_loop().run_in_executor(
        None, partial(classify_intent, generator.client, req.query)
    )

    if intent in config.NON_RETRIEVAL_INTENTS:
        intent_response = config.INTENT_RESPONSE_MAP.get(intent, {})

        def intent_stream():
            yield json.dumps(
                {
                    "type": "meta",
                    "data": {
                        "retrieval_ms": 0,
                        "results_count": 0,
                        "reranked": False,
                    },
                }
            ) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps({"type": "token", "data": intent_response}) + "\n"

        return StreamingResponse(
            intent_stream(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Skip retrieval for non-food queries
    if is_non_food_query(req.query):

        def greeting_stream():
            yield json.dumps(
                {
                    "type": "meta",
                    "data": {
                        "retrieval_ms": 0,
                        "results_count": 0,
                        "reranked": False,
                    },
                }
            ) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps(
                {
                    "type": "token",
                    "data": (
                        "Hi! I'm DineRAG, a restaurant recommendation assistant powered by real Yelp reviews 🍽️\n\n"
                        "Try asking:\n"
                        "- *Best tacos in Philadelphia*\n"
                        "- *Romantic Italian dinner in Nashville*\n"
                        "- *Late night ramen in Tampa*\n"
                        "- *Casual Indian restaurant in Pennsylvania*\n\n"
                        "I cover cities across: \n"
                        "📍 **Pennsylvania** — Philadelphia, King of Prussia, Norristown, Doylestown (PA)\n"
                        "📍 **California** — Santa Barbara, Goleta, Montecito, Carpinteria (CA)\n"
                        "📍 **New Jersey** — Cherry Hill, Camden, Voorhees, Haddonfield (NJ)\n"
                        "📍 **Florida** — Tampa, Clearwater, St. Petersburg, Brandon (FL)\n"
                        "📍 **Tennessee** — Nashville, Brentwood, Franklin, Hendersonville (TN)\n"
                        "📍 **Louisiana** — New Orleans, Metairie, Kenner, Chalmette (LA)\n"
                        "📍 **Indiana** — Indianapolis, Carmel, Fishers, Noblesville (IN)\n"
                        "📍 **Arizona** — Tucson, Oro Valley, Marana, Sahuarita (AZ)\n"
                        "📍 **Nevada** — Reno (NV)\n"
                        "📍 **Idaho** — Boise, Meridian, Eagle (ID)\n"
                        "📍 **Illinois** — Belleville, Collinsville, Mascoutah, Caseyville (IL)\n"
                        "📍 **Missouri** — Saint Louis, Chesterfield, Ballwin, Creve Coeur (MO)\n"
                        "📍 **Delaware** — Wilmington, Claymont, Christiana (DE)\n"
                        "📍 **Alberta** — Edmonton (AB)\n"
                    ),
                }
            ) + "\n" + "\n"

        return StreamingResponse(
            greeting_stream(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    if is_out_of_coverage(req.query):

        def coverage_stream():
            yield json.dumps(
                {
                    "type": "meta",
                    "data": {
                        "retrieval_ms": 0,
                        "results_count": 0,
                        "reranked": False,
                    },
                }
            ) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps(
                {"type": "token", "data": config.COVERAGE_MESSAGE}
            ) + "\n"

        return StreamingResponse(
            coverage_stream(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # 2. Retrieve Context
    with Timer("generator", "context_fetch"):
        context_results, retrieval_ms = fetch_context(full_query, req.top_k)

    logger.info(f"[API] Retrieved {len(context_results)} snippets.")
    QUERY_COUNTER.labels("generator", "started").inc()

    # 3. Stream Response

    async def response_stream():
        """
        Internal generator to yield Server-Sent Events (SSE) or NDJSON chunks
        for metadata, sources, generating tokens, and eventual exceptions.

        Yields:
            str: JSON encoded string chunks to be streamed directly to the client.
        """
        # A. Send Sources First (JSON)
        # We strip the heavy 'text' field for the frontend source list to save bandwidth
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

        # print(context_results[0].get("text").split("--")[1][:400])
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

        try:
            async with _gen_lock:
                for token in generator.generate_stream_groq(req.query, context_results):
                    yield json.dumps({"type": "token", "data": token}) + "\n"
            QUERY_COUNTER.labels("generator", "remove success").inc()
        except RateLimitError:
            yield json.dumps(
                {"type": "error", "data": "Rate limit hit — try again shortly."}
            ) + "\n"
            QUERY_COUNTER.labels("generator", "rate_limit").inc()
        except APIError as e:
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
