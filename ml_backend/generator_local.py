import gc
import re
from threading import Thread
import os
import logging


from typing import List, Dict, Any, Generator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import torch
import config
import httpx
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from groq import Groq, APIError, RateLimitError

import config

_gen_lock = asyncio.Semaphore(1)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from observability import (
    setup_logging,
    attach_prometheus,
    Timer,
    QUERY_COUNTER,
    STAGE_LATENCY,
)
from loguru import logger


class RAGGenerator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.client = None
        self.device = config.GEN_DEVICE

    def load_model_qwen(self):
        """Loads the quantized model into memory."""
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
            self.model = torch.compile(self.model)
            self.model.eval()
            logger.info("[Generator] Model loaded successfully.")
        except Exception as e:
            logger.error(f"[Generator] Critical Error loading model: {e}")
            raise e

    def _build_prompt_qwen(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Constructs the system and user prompts using retrieved context."""

        context_text_list = []
        for res in context_snippets:
            name = res.get("restaurant") or res.get("name") or "Unknown"
            text = res.get("text") or res.get("chunks") or res.get("content") or ""
            city = res.get("city", "")

            snippet = (
                f"Restaurant: {name}\n"
                f"Location: {res.get('address', 'Unknown')} ({city})\n"
                f"Description: {text}\n"
                f"---"
            )
            context_text_list.append(snippet)

        context_block = "\n".join(context_text_list)

        system_msg = config.GROQ_SYSTEM_PROMPT

        user_msg = f"User Query: {query}\n\nContext:\n{context_block}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

    def _run_generation(self, **kwargs):

        with torch.no_grad():
            self.model.generate(**kwargs)

    def generate_stream_qwen(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Streams tokens from the LLM in a separate thread."""

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        inputs = self._build_prompt_qwen(query, self.trim_context(context_snippets))

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
            repetition_penalty=1.1,
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


gen_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("generator")
    gen_state["generator"] = RAGGenerator()
    logger.info("Starting generator service (Groq)...")
    gen_state["generator"].load_model_qwen()
    yield
    logger.info("--- Shutting Down Generator Service ---")
    del gen_state["generator"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="RAG Generator API", lifespan=lifespan)
attach_prometheus(app, "generator")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Accel-Buffering", "Transfer-Encoding"],
)


class GenerateRequest(BaseModel):
    query: str
    city: Optional[str] = None
    state: Optional[str] = None
    top_k: int = config.TOP_K


def is_non_food_query(query: str) -> bool:
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
    q = query.lower()
    if any(_area_mentioned(area, query, q) for area in config.COVERED_AREAS):
        return False
    return any(_area_mentioned(city, query, q) for city in config.OUT_OF_COVERAGE)


def _safe_excerpt(text: Optional[str], max_len: int = 400) -> str:
    if not text:
        return ""
    parts = text.split("--")
    return (parts[1] if len(parts) > 1 else parts[0]).strip()[:max_len]


async def fetch_context(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Calls the separate Retriever Microservice."""

    try:
        payload = {"query": query, "top_k": top_k, "do_rerank": config.DO_RERANK}
        logger.info(f"[API] Fetching context from {config.RETRIEVER_URL}...")

        with Timer("generator", "retriever_fetch"):
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(config.RETRIEVER_URL, json=payload)
                response.raise_for_status()

        data = response.json()
        logger.info(f"[API] Context fetched successfully: {response.json()}")

        if isinstance(data, list):
            results = data
            retrieval_ms = 0.0
        else:
            results = data.get("results", [])
            retrieval_ms = data.get("retrieval_ms", 0.0)

        logger.info(f"Retrieved {len(data)} snippets for '{query[:60]}'")
        return results, retrieval_ms

    except httpx.RequestError as e:

        logger.error(f"[API] Retrieval network error: {e}")
        return [], 0.0

    except httpx.HTTPStatusError as e:
        logger.error(f"[API] Retriever returned {e.response.status_code}: {e}")
        return [], 0.0


@app.get("/health")
def health_check():
    generator = gen_state.get("generator")
    if not generator or not generator.model:
        from fastapi import Response

        return Response(status_code=503)
    return {"status": "active", "service": "Qwen Generator"}


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    generator: RAGGenerator = gen_state.get("generator")
    if not generator:
        raise HTTPException(status_code=500, detail="Model not loaded")

    logger.info(f"[generate_endpoint] req.query raw: '{req.query}'")
    full_query = req.query
    if req.city:
        full_query += f" in {req.city}"
    logger.info(f"[generate_endpoint] full_query: '{full_query}'")

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

    context_results, retrieval_ms = await fetch_context(full_query, req.top_k)

    logger.info(f"[API] Retrieved {len(context_results)} snippets.")
    QUERY_COUNTER.labels("generator", "started").inc()

    async def response_stream():

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
                for token in generator.generate_stream_qwen(req.query, context_results):
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
