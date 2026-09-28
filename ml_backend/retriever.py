from http import client

import os

import pickle
import re
import string
import time
import numpy as np
import pandas as pd
import spacy
import torch
from typing import Optional, List
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models


import config
from ml_backend.observability import (
    setup_logging,
    attach_prometheus,
    Timer,
    QUERY_COUNTER,
    CACHE_COUNTER,
    RESULTS_HISTOGRAM,
    RERANK_IMPROVEMENT,
)
from ml_backend.cache import get_cached, set_cached, cache_stats, evict_all
from loguru import logger

load_dotenv()


class HybridRetriever:
    """Hybrid retrieval: Qdrant dense search + full-corpus BM25, RRF fusion, cross-encoder rerank."""

    def __init__(self):
        self.qdrant: Optional[QdrantClient] = None
        self.reranker: Optional[CrossEncoder] = None
        self.nlp = None
        self.city_list = set()
        self.embedding_model = config.EMBEDDING_MODEL_NAME
        self.embedder = None

        self.bm25_full = None
        self.corpus_df: Optional[pd.DataFrame] = None
        self.corpus_city: Optional[pd.Series] = None
        self.corpus_state: Optional[pd.Series] = None
        self.ready = False

    def initialize(self):
        """Connects to Qdrant, loads the embedder, reranker, spaCy and BM25 index, then sets ready.

        Raises:
            Exception: If the Qdrant connection fails.
        """
        logger.info(f"Initializing Hybrid Retriever on {config.DEVICE}...")

        # 1. Qdrant
        try:
            self.qdrant = QdrantClient(
                url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60
            )
            self.qdrant.get_collections()  # Health check
            logger.info("Connected to Qdrant.")
            for field in ["city", "state", "restaurant", "doc_id", "chunk_id"]:
                self.qdrant.create_payload_index(
                    collection_name=config.COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info("Connected to Qdrant and indices verified.")
        except Exception as e:
            logger.error(f"Qdrant Connection Failed: {e}")
            raise e

        # 2. Reranker
        logger.info(f"Loading Reranker: {config.RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL_NAME, max_length=512, device=config.DEVICE
        )

        # Embedder
        logger.info(f"Loading E5 embedder on {config.DEVICE}...")
        self.embedder = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, device=config.DEVICE
        )
        logger.info("E5 embedder ready.")

        # 3. Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download

            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # 4. City Data
        try:
            if pd.io.common.file_exists(config.DATA_PATH):
                df = pd.read_csv(config.DATA_PATH)
                if "city" in df.columns:
                    self.city_list = set(df["city"].str.lower().dropna().unique())
                    logger.info(f"Loaded {len(self.city_list)} cities.")
        except Exception as e:
            logger.warning(f"Could not load city list: {e}")

        # 5. Full-corpus sparse index.
        try:
            if config.BM25_PATH.exists() and config.METADATA_PATH.exists():
                with open(config.BM25_PATH, "rb") as f:
                    self.bm25_full = pickle.load(f)
                self.corpus_df = pd.read_parquet(
                    config.METADATA_PATH,
                    columns=[
                        "chunk_id",
                        "chunk",
                        "business_id",
                        "restaurant",
                        "city",
                        "state",
                        "address",
                        "latitude",
                        "longitude",
                    ],
                )
                self.corpus_city = self.corpus_df["city"].fillna("").str.lower()
                self.corpus_state = self.corpus_df["state"].fillna("")
                logger.info(
                    f"Loaded full-corpus BM25 index ({len(self.corpus_df)} chunks)."
                )
            else:
                logger.warning(
                    "Full-corpus BM25 index or metadata not found — falling back "
                    "to rescoring just the dense shortlist."
                )
        except Exception as e:
            logger.warning(f"Could not load full-corpus BM25 index: {e}")

        self.ready = True

    def _get_embedding(self, text: str) -> List[float]:
        """Returns the normalized E5 query embedding for text."""
        with Timer("retriever", "embedding"):
            vec = self.embedder.encode(
                f"query: {text}",
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return vec.tolist()

    def _detect_raw_location(self, query: str) -> Optional[str]:
        """Returns the first location in query (pattern, then NER), lowercased, covered or not."""
        pattern_match = re.search(config.LOCATION_PATTERN, query, re.IGNORECASE)
        if pattern_match:
            return pattern_match.group(1).strip().lower()

        if self.nlp:
            for ent in self.nlp(query).ents:
                if ent.label_ == "GPE":
                    return ent.text.strip().lower()

        return None

    def _classify_location(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns (city, None), (None, state_code) or (None, None) for lowercased text."""
        if text in config.DATASET_CITIES:
            return text, None
        if text in config.US_STATES:
            return None, config.US_STATES[text]
        if text.upper() in config.DATASET_STATES:
            return None, text.upper()
        return None, None

    def _extract_location(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns the dataset (city, state) named in query.

        A state without a city maps to config.STATE_TO_PRIMARY_CITY.
        """
        city, state = None, None

        # 1. "in <Location>" pattern
        pattern_match = re.search(config.LOCATION_PATTERN, query, re.IGNORECASE)
        if pattern_match:
            city, state = self._classify_location(
                pattern_match.group(1).strip().lower()
            )

        # 2. NER on the full query, then on the tail of long queries
        segments = [query] + ([query[-60:]] if len(query) > 60 else [])
        for segment in segments:
            if city or state or not self.nlp:
                break
            for ent in self.nlp(segment).ents:
                if ent.label_ == "GPE":
                    ent_city, ent_state = self._classify_location(
                        ent.text.strip().lower()
                    )
                    city = ent_city or city
                    state = ent_state or state

        # 3. Dictionary fallback over individual tokens
        if not city and self.city_list:
            for word in query.lower().split():
                if word in config.DATASET_CITIES:
                    city = word
                    break

        # 4. State -> primary city mapping
        if state and not city:
            mapped = config.STATE_TO_PRIMARY_CITY.get(state)
            if mapped:
                city = mapped
                logger.info(f"State {state} -> city {city}")

        return city, state

    def detect_out_of_coverage(self, query: str) -> Optional[str]:
        """Returns the detected location if it is outside coverage, else None."""
        raw = self._detect_raw_location(query)
        if not raw:
            return None

        def _mentions(areas: List[str]) -> bool:
            return any(
                re.search(r"\b" + re.escape(area) + r"\b", raw) for area in areas
            )

        if _mentions(config.OUT_OF_COVERAGE) or not _mentions(config.COVERED_AREAS):
            return raw
        return None

    def search(
        self,
        query: str,
        top_k: int,
        initial_k: int,
        k_rrf: int,
        do_rerank: bool,
        max_duplicates: int,
    ):
        """Runs dense search, BM25, RRF fusion and optional reranking, filtered by detected location.

        Args:
            top_k: Results returned.
            initial_k: Dense candidates fetched from Qdrant.
            k_rrf: RRF constant.
            do_rerank: Rerank fused candidates with the cross-encoder.
            max_duplicates: Max chunks per restaurant in the results.

        Returns:
            (results, retrieval_ms).
        """

        t0 = time.perf_counter()
        # A. Extract Location & Build Filter
        city, state = self._extract_location(query)
        logger.info(f"Query: '{query}' | Extracted Loc: {city}, {state}")

        conditions = []
        if city:
            conditions.append(
                models.FieldCondition(
                    key="city", match=models.MatchValue(value=city.lower())
                )
            )
        if state and not city:
            conditions.append(
                models.FieldCondition(key="state", match=models.MatchValue(value=state))
            )

        q_filter = models.Filter(must=conditions) if conditions else None
        logger.debug(f"Qdrant Filter: {q_filter}")

        # B. Get Embeddings & Query Qdrant
        with Timer("retriever", "embedding"):
            query_vec = self._get_embedding(query)
            t1 = time.perf_counter()

        with Timer("retriever", "qdrant_query"):
            try:
                points = self.qdrant.query_points(
                    collection_name=config.COLLECTION_NAME,
                    query=query_vec,
                    query_filter=q_filter,
                    limit=initial_k,
                    with_payload=True,
                ).points
            except Exception as e:
                logger.error(f"Qdrant Query Failed: {e}")
                return [], float((time.perf_counter() - t0) * 1000)
            t2 = time.perf_counter()

        logger.info(f"Qdrant returned {len(points)} points.")

        # C. Prepare dense candidates, keyed by the stable chunk_id assigned at
        # ingestion time so they can be merged with sparse candidates below.
        dense_by_id: Dict[str, Dict[str, Any]] = {}
        for p in points:
            cid = p.payload.get("chunk_id") or p.id
            dense_by_id[cid] = {
                "text": p.payload.get("text_content", ""),
                "meta": p.payload,
                "vec_score": p.score,
            }

        # D. Sparse retrieval — BM25 over the full corpus when the full-corpus
        # index is loaded, falling back to rescoring just the dense shortlist.
        with Timer("retriever", "bm25"):
            translator = str.maketrans("", "", string.punctuation)
            clean_query = query.lower().translate(translator).split()
            expansions = {
                syn for tok in clean_query for syn in config.QUERY_SYNONYMS.get(tok, [])
            }
            clean_query = clean_query + [t for t in expansions if t not in clean_query]

            sparse_by_id: Dict[str, Dict[str, Any]] = {}
            if self.bm25_full is not None and self.corpus_df is not None:
                scores = self.bm25_full.get_scores(clean_query)
                mask = np.ones(len(scores), dtype=bool)
                if city:
                    mask &= (self.corpus_city == city.lower()).to_numpy()
                if state and not city:
                    mask &= (self.corpus_state == state).to_numpy()
                scores = np.where(mask, scores, -np.inf)
                top_sparse_idx = np.argsort(-scores)[:initial_k]
                for pos in top_sparse_idx:
                    if not np.isfinite(scores[pos]):
                        continue
                    row = self.corpus_df.iloc[pos]
                    cid = row["chunk_id"]
                    sparse_by_id[cid] = {
                        "text": row["chunk"],
                        "meta": row.to_dict(),
                        "bm25_score": float(scores[pos]),
                    }
            else:
                corpus_texts = [c["text"] for c in dense_by_id.values()]
                tokenized_corpus = [
                    doc.lower().translate(translator).split() for doc in corpus_texts
                ]
                if tokenized_corpus:
                    local_bm25 = BM25Okapi(tokenized_corpus)
                    local_scores = local_bm25.get_scores(clean_query)
                    for cid, sc in zip(dense_by_id.keys(), local_scores):
                        sparse_by_id[cid] = {
                            **dense_by_id[cid],
                            "bm25_score": float(sc),
                        }
            t3 = time.perf_counter()

        # E. Union of dense + sparse candidates, then Reciprocal Rank Fusion.
        all_ids = list(
            dict.fromkeys(list(dense_by_id.keys()) + list(sparse_by_id.keys()))
        )
        chunks = []
        for cid in all_ids:
            base = dense_by_id.get(cid) or sparse_by_id.get(cid)
            chunks.append({"id": cid, "text": base["text"], "meta": base["meta"]})
        corpus_texts = [c["text"] for c in chunks]

        vector_scores = np.array(
            [
                dense_by_id[c["id"]]["vec_score"] if c["id"] in dense_by_id else -np.inf
                for c in chunks
            ]
        )
        bm25_scores = np.array(
            [
                (
                    sparse_by_id[c["id"]]["bm25_score"]
                    if c["id"] in sparse_by_id
                    else -np.inf
                )
                for c in chunks
            ]
        )

        def _ranks(scores: np.ndarray) -> np.ndarray:
            # Rank only among finite (actually-present) scores; absent candidates
            # get no RRF term at all.
            present = np.isfinite(scores)
            ranks = np.full(len(scores), len(scores), dtype=float)
            ranks[present] = np.argsort(np.argsort(-scores[present]))
            return ranks

        vec_rank = _ranks(vector_scores)
        bm25_rank = _ranks(bm25_scores)

        rrf_scores = np.where(
            np.isfinite(vector_scores), 1 / (k_rrf + vec_rank), 0.0
        ) + np.where(np.isfinite(bm25_scores), 1 / (k_rrf + bm25_rank), 0.0)

        # Sort by RRF score descending
        candidate_indices = np.argsort(-rrf_scores)
        t4 = time.perf_counter()

        # F. Reranking (Cross Encoder)
        final_indices = candidate_indices
        final_scores = rrf_scores[candidate_indices]

        t5 = t4

        if do_rerank and self.reranker:
            top_candidates_idx = candidate_indices
            with Timer("retriever", "rerank"):
                # No manual char truncation — the CrossEncoder was constructed
                # with max_length=512 and truncates on its own tokenizer.
                pairs = [[query, corpus_texts[i]] for i in top_candidates_idx]

                if pairs:
                    rerank_scores = self.reranker.predict(pairs)
                    t5 = time.perf_counter()
                    sorted_rerank_idx = np.argsort(-rerank_scores)

                    final_indices = [top_candidates_idx[i] for i in sorted_rerank_idx]
                    final_scores = [rerank_scores[i] for i in sorted_rerank_idx]

                    if (
                        len(final_indices) > 0
                        and final_indices[0] != candidate_indices[0]
                    ):
                        RERANK_IMPROVEMENT.labels("retriever").inc()

        # G. Deduplication & Formatting
        results = []
        seen_biz = {}

        for idx, score in zip(final_indices, final_scores):
            c = chunks[idx]
            b_id = c["meta"].get("business_id")

            if seen_biz.get(b_id, 0) < max_duplicates:
                results.append(
                    {
                        "score": float(score),
                        "business_id": b_id,
                        "restaurant": c["meta"].get("restaurant", "Unknown"),
                        "text": c["text"],
                        "city": c["meta"].get("city"),
                        "state": c["meta"].get("state"),
                        "address": c["meta"].get("address"),
                        "latitude": c["meta"].get("latitude"),
                        "longitude": c["meta"].get("longitude"),
                    }
                )
                seen_biz[b_id] = seen_biz.get(b_id, 0) + 1

            if len(results) >= top_k:
                break
        logger.info(f"Search returning {len(results)} deduplicated results.")
        t_end = time.perf_counter()
        retrieval_ms = float((t_end - t0) * 1000)

        try:
            logger.info(
                f"[PERF] embed={int((t1-t0)*1000)}ms | "
                f"qdrant={int((t2-t1)*1000)}ms | "
                f"bm25={int((t3-t2)*1000)}ms | "
                f"rrf={int((t4-t3)*1000)}ms | "
                f"rerank={int((t5-t4)*1000)}ms | "
                f"total={int((t5-t0)*1000)}ms"
            )
        except Exception:
            pass
        return results, retrieval_ms


# Global Service Instance
retriever = HybridRetriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the retriever on startup."""
    # Startup
    setup_logging("retriever")
    try:
        retriever.initialize()
        logger.info("Retriever ready.")
    except Exception as e:
        logger.error(f"Critical Startup Error: {e}")
    yield
    # Shutdown
    logger.info("Shutting down retriever service...")


app = FastAPI(title="Retrieval Microservice", lifespan=lifespan)
attach_prometheus(app, "retriever")


# --- Models ---
class RetrieveRequest(BaseModel):
    """Request body for /retrieve."""

    query: str
    top_k: int = config.TOP_K
    initial_k: int = config.INITIAL_K
    k_rrf: int = config.RRF_K
    max_duplicates: int = config.MAX_DUPLICATES
    do_rerank: bool = config.DO_RERANK
    no_cache: bool = False


class RetrieveResponse(BaseModel):
    """Response body for /retrieve."""

    results: List[Dict[str, Any]]
    retrieval_ms: float
    out_of_coverage: bool = False
    detected_location: Optional[str] = None


# --- Endpoints ---
@app.get("/health")
def health_check(response: Response):
    """Returns service status; 503 until the retriever is initialized."""
    if not retriever.ready or retriever.qdrant is None:
        response.status_code = 503
        return {"status": "not_ready"}
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest, request: Request):
    """Returns cached or fresh search results; out-of-coverage queries return none.

    Raises:
        HTTPException: 503 if not initialized; 500 on search errors.
    """
    if not retriever.ready or not retriever.qdrant:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    # Cache check
    cached = None
    if not req.no_cache:
        cached = get_cached(
            req.query,
            req.top_k,
            req.do_rerank,
            req.k_rrf,
            req.initial_k,
            req.max_duplicates,
        )
    if cached is not None:
        CACHE_COUNTER.labels("hit").inc()
        QUERY_COUNTER.labels("retriever", "cache_hit").inc()
        return {"results": cached, "retrieval_ms": 0}

    # If no cached result, perform full retrieval
    CACHE_COUNTER.labels("miss").inc()

    # Coverage check
    detected_location = retriever.detect_out_of_coverage(req.query)
    if detected_location:
        logger.info(
            f"Retriever blocked out-of-coverage location: '{detected_location}'"
        )
        return {
            "results": [],
            "retrieval_ms": 0.0,
            "out_of_coverage": True,
            "detected_location": detected_location,
        }

    # Full retrieval pipeline
    try:
        results, retrieval_ms = retriever.search(
            query=req.query,
            top_k=req.top_k,
            initial_k=req.initial_k,
            k_rrf=req.k_rrf,
            do_rerank=req.do_rerank,
            max_duplicates=req.max_duplicates,
        )

    except Exception as e:
        QUERY_COUNTER.labels("retriever", "error").inc()
        import traceback

        logger.error(
            f"Retrieval failed for '{req.query}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    RESULTS_HISTOGRAM.observe(len(results))
    QUERY_COUNTER.labels("retriever", "success").inc()

    if results and not req.no_cache:
        set_cached(
            req.query,
            req.top_k,
            req.do_rerank,
            results,
            req.k_rrf,
            req.initial_k,
            req.max_duplicates,
        )

    logger.info(f"Retrieved {(len(results))} for query: '{req.query}'")
    logger.info(f"Request from IP: {request.headers.get('x-forwarded-for', 'unknown')}")

    return {"results": results, "retrieval_ms": retrieval_ms}


@app.get("/cache/stats")
def cache_stats_endpoint():
    """Returns retrieval cache statistics."""
    return cache_stats()


@app.post("/cache/clear")
def clear_cache_():
    """Evicts every entry from the retrieval cache."""
    evict_all()
    return {"status": "cache cleared"}


@app.get("/debug/cities")
def get_unique_cities():
    """Returns sorted distinct cities and states in the Qdrant collection (full scroll)."""
    cities = set()
    states = set()
    offset = None

    while True:
        points, offset = retriever.qdrant.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            city = point.payload.get("city")
            state = point.payload.get("state")
            if city:
                cities.add(city.lower().strip())
            if state:
                states.add(state.upper().strip())
        if offset is None:
            break

    return {"cities": sorted(cities), "states": sorted(states)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
