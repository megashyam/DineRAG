# DineRAG: Production-Grade RAG System on the Yelp Business Review Dataset Entirely from Scratch

**DineRAG** is a high-performance Retrieval-Augmented Generation (RAG) system engineered to provide grounded, location-aware restaurant recommendations  **built entirely from scratch without LangChain.** It leverages a **Hybrid Search Architecture** (Dense Vectors + Sparse Keywords) fused with a Cross-Encoder Reranker to retrieve precise context from the Yelp Academic Dataset, which is then synthesized by a 4-bit quantized LLM.

The system is architected as a set of decoupled, asynchronous microservices to ensure scalability and fault tolerance.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Inference-FFD21E?logo=huggingface&logoColor=black)
![Transformers](https://img.shields.io/badge/Transformers-NLP%20Models-FFD21E?logo=huggingface&logoColor=black)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-Embedding%20Models-FF6F00)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-FFB000)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-Quantization-FF6F00)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-DC244C)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend%20API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker&logoColor=white)
![Modal](https://img.shields.io/badge/Modal-Serverless%20GPU%20Compute-000000)
![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)
![Airflow](https://img.shields.io/badge/Airflow-Workflow%20Orchestration-017CEE?logo=apacheairflow&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white)
![DeepEval](https://img.shields.io/badge/DeepEval-LLM%20Evaluation-6E56CF)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?logo=nextdotjs&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Deployment-000000?logo=vercel&logoColor=white)





![Demo GIF](data/demo.gif)

> ⚡ **Live Demo:** [yelp-restaurant-rag.vercel.app](https://yelp-restaurant-rag.vercel.app)
> First request after inactivity takes ~15–20s (serverless cold start while E5 loads into memory). Subsequent queries respond faster.

---

## Why From Scratch?

To demonstrate a real understanding of everything that happens under the hood of RAG Frameworks. DineRAG is built with full control and zero abstractions. There is no managed loader, no pre-built retriever, no abstracted LLM call.


## Index

* [Why From Scratch?](#why-from-scratch)

* [Architecture Overview](#architecture-overview)

* [Data Pipeline — From Raw Yelp JSON to Production Grade Vector DB](#data-pipeline--from-raw-yelp-json-to-production-grade-vector-db)

  * [Challenges Solved](#challenges-solved)

* [Under the Hood](#under-the-hood)

  * [Streaming Filter Cascade (`preprocessor.py`)](#streaming-filter-cascade-preprocessorpy)

  * [Composite Restaurant Scoring (`preprocessor.py`)](#composite-restaurant-scoring-preprocessorpy)

  * [Balanced Sentiment Sampling (`pipeline.py`)](#balanced-sentiment-sampling-pipelinepy)

  * [Token-Bounded Chunking (`chunker.py`)](#token-bounded-chunking-chunkerpy)

  * [Typed Chunk Structure (`chunker.py`)](#typed-chunk-structure-chunkerpy)

* [Retrieval](#retrieval)

  * [Hybrid Search](#hybrid-search)

  * [Location Extraction](#location-extraction)

* [Query Routing](#query-routing)

  * [Intent Classification](#intent-classification)

  * [Coverage Guard (Two Layers)](#coverage-guard-two-layers)

* [Inference](#inference)

  * [Production - Groq API](#production---groq-api)

  * [Local / Offline - Qwen2.5-3B NF4](#local--offline---qwen25-3b-nf4)

  * [Comparison](#comparison)

* [Retrieval Quality Evaluation](#retrieval-quality-evaluation)

* [Deployment](#deployment)

  * [Serverless Backend (Modal)](#serverless-backend-modal)

  * [Frontend](#frontend)

* [Performance](#performance)

  * [Retrieval (shared across both paths)](#retrieval-shared-across-both-paths)

  * [End-to-End (warm, deployed)](#end-to-end-warm-deployed)

* [Project Structure](#project-structure)

* [Performance Optimizations Implemented](#performance-optimizations-implemented)

* [Setup](#setup)

  * [Requirements](#requirements)

  * [Install](#install)

  * [Configure](#configure)

  * [Run Data Pipeline](#run-data-pipeline)

  * [Run Services](#run-services)

  * [Run Evaluation](#run-evaluation)

  * [Run Tests](#run-tests)

* [API](#api)

  * [Generate (user-facing)](#generate-user-facing)

  * [Retrieve (debug)](#retrieve-debug)

---


## Architecture Overview

```
Raw Yelp JSON (~10GB)
    → Preprocessor (scoring, filtering, sentiment sampling)
    → Chunker (token-bounded, typed chunks)
    → Embedder (E5-large-v2, BM25 index)
    → Qdrant Cloud (vector DB)
         ↓
Retriever Microservice (FastAPI)
    → Geo filter (spaCy NER)
    → Qdrant vector search (HNSW)
    → Full-corpus BM25 (independent sparse retrieval, query synonym expansion)
    → RRF fusion
    → CrossEncoder reranking
         ↓
Generator Microservice (FastAPI)
    → Intent Classification (llama-3.1-8b-instant via Groq)
         ↓ food_search / location_only → proceed
         ↓ greeting / identity / off_topic → short-circuit, no retrieval
    → Coverage Guard (COVERED_AREAS / OUT_OF_COVERAGE blocklists)
         ↓ out-of-coverage → short-circuit, no retrieval
    → Groq API (gpt-oss-20b, production)
    → Local Qwen2.5-3B NF4 (offline mode)
         ↓
Next.js Frontend (Vercel)
```

## Data Pipeline — From Raw Yelp JSON to Production Grade Vector DB

The entire data pipeline is hand-built from the raw 
[Yelp Academic Dataset](https://www.yelp.com/dataset) ~10GB across multiple 
JSON files with no managed loaders, pre-processed datasets, or data APIs.

### Challenges solved:

**1. Multi-file joins at scale**: Streamed joins between `business.json` and `review.json` using `business_id`. without loading full datasets into RAM.

**2. Category filtering**: Parsed Yelp’s free-text category lists to isolate restaurants from 1000+ business types.

**3. Custom Restaurant Score**: Built a weighted ranking system combining rating, review count, and recency for more reliable quality scoring.

**4. Balanced sentiment sampling**: Sampled reviews across positive, neutral, and negative buckets to capture tradeoffs, not just high ratings.

**5. Token-bounded chunking**: Used E5's own tokenizer for splitting so chunk budgets match what the embedding model  sees.

**6. Typed chunk structure**: Created structured chunks (profile, positive, negative) instead of raw text to improve retrieval specificity.

**7. Embedding at scale**: Generated E5-large-v2 embeddings in batches and stored them as .pt tensors for fast reload.

**8. Stable deduplication**: Used UUID5-based deterministic IDs to ensure idempotent reprocessing without duplicates.

**9. Generator-based ingestion**: Streamed Qdrant uploads in batches of 256 using generators to keep memory usage constant.

---

## Under the Hood

### Streaming Filter Cascade (`preprocessor.py`)
Reviews are filtered at parse time in cheapest-to-expensive order to avoid loading 20GB+ into RAM:
1. **Business ID** — O(1) set lookup. Filters 95% of records instantly.
2. **Date filter** — lexicographic ISO string compare.
3. **Word count** — `text.split()` only runs on the survivors of (1) and (2).

### Composite Restaurant Scoring (`preprocessor.py`)
Raw star ratings are statistically broken for ranking; 3 reviews at 5 outranks
2000 reviews at 4.7 naively. I Built a two-component weighted score:
- `restaurant_score = stars × log1p(review_count)` — long-term reputation signal
- `reviews_score = mean_stars × log1p(sum_stars)` — recent review quality

Both are Min-Max normalized before combining with tunable coefficients.
City caps are adaptive, dense cities (600+ restaurants) get higher limits than sparse ones.

### Balanced Sentiment Sampling (`pipeline.py`)
Naive top-N sampling produces all 5-star reviews. The LLM then has no context
on wait times, portion sizes, or service issues. Reviews are sampled proportionally
across positive/neutral/negative buckets with ceiling rounding to preserve rare
sentiment classes.

### Token-Bounded Chunking (`chunker.py`)
Chunking tokenizes with E5's own tokenizer, so the token budget matches what
the embedder sees. Individual reviews are capped and truncated at the token
level (not by character count) before being packed into a batch, so a single
long review can't blow the chunk budget on its own.

### Typed Chunk Structure (`chunker.py`)
The chunks are not raw review blocks. Each restaurant produces structured chunk types:
- **Business Profile** — name, location, category, hours
- **Attribute chunks** — parsed from Yelp's nested stringified dicts (e.g. WiFi, parking, alcohol)
- **Vibe chunks** — atmosphere descriptors extracted from nested attribute maps
- **Positive / Neutral / Negative review batches** — sentiment-separated for retrieval precision

This allows the retriever to surface different facets of the same restaurant
depending on query intent, a "romantic atmosphere" query hits vibe chunks,
a "avoid if in a rush" query hits negative review chunks.

---

## Retrieval

### Hybrid Search

**Dense retrieval** — E5-large-v2 embeddings queried against Qdrant HNSW index. Captures semantic similarity ("cheap eats" -> "affordable prices").

**Sparse retrieval** — BM25 over the full chunk corpus. A small synonym table expands the sparse query (e.g. "bbq" <-> "barbecue", "brunch" <-> "breakfast").

**RRF fusion** — Reciprocal Rank Fusion combines both rankings without score normalization:
```
rrf_score = 1/(k + vec_rank) + 1/(k + bm25_rank)    k=60
```

**Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` processes `[CLS] Query [SEP] Document` jointly, enabling deep interaction modeling between query and document tokens.

### Location Extraction
spaCy NER extracts city/state from queries. Fallback dictionary matching handles abbreviations ("Philly" -> Philadelphia, "NOLA" -> New Orleans) and state→city routing ("Pennsylvania" -> Philadelphia metro).

---

## Query Routing

Before any retrieval occurs, every query passes through two filtering layers:

### Intent Classification
`llama-3.1-8b-instant` via Groq classifies the query (~200ms) into one of:

| Intent | Action |
|:---|:---|
| `food_search` | Proceed to coverage check + retrieval |
| `location_only` | Proceed to retrieval with geo filter |
| `greeting` | Return intro message, skip retrieval |
| `identity` | Return DineRAG description, skip retrieval |
| `off_topic` | Return redirect message, skip retrieval |


### Coverage Guard (Two Layers)
Out-of-coverage cities are blocked before retrieval at two independent points:

- **Generator-level** — `is_out_of_coverage()` checks the query against `COVERED_AREAS` / `OUT_OF_COVERAGE` lists before calling the retriever
- **Retriever-level** — `_detect_raw_location()` catches anything that slips through, returns `out_of_coverage: true` with empty results so irrelevant source cards never appear in the UI

Both layers use raw location detection (NER + regex) that is **not** filtered against the dataset so an unknown city like "Athens" is detected and blocked rather than silently returning unfiltered results from other cities.

---

## Inference

Two generator paths are implemented production uses Groq, local runs a fully quantized model offline.

### Production - Groq API
`gpt-oss-20b` via Groq. Zero GPU cost, zero VRAM. Response is currently buffered by Modal's ASGI proxy — full response arrives after ~4s rather than true token streaming.

| Metric | Value |
|:---|:---|
| TTFT | ~4s (Modal buffered) |
| Tokens/sec | ~400k |
| GPU cost | 0 |

### Local / Offline - Qwen2.5-3B NF4

Qwen2.5-3B-Instruct in 4-bit NF4 quantization via `bitsandbytes`. Measured on RTX 3060 6GB:

| Metric | Value |
|:---|:---|
| VRAM baseline | 0 MB |
| VRAM at load | 2221 MB (~2.2GB) |
| VRAM peak generation | 2369 MB (~2.3GB) |
| Headroom remaining | ~3.7GB |
| Tokens/sec | 11.8 |
| TTFT (warm) | ~3.7s |
| TTFT (cold, first run) | ~37s (torch.compile JIT) |

Config:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

- `attn_implementation="sdpa"` — PyTorch scaled dot-product attention
- `torch.backends.cuda.matmul.allow_tf32 = True` — TF32 on Ampere
- `TextIteratorStreamer` + background thread — `model.generate()` runs in a thread, main thread yields tokens without blocking the event loop
- `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` after each generation prevents VRAM fragmentation across requests

**Why not 7B?** Qwen2.5-7B NF4 theoretically fits (~3.5GB weights) but spills onto CPU at inference time on 6GB. Measured result: TTFT 58s, 0.6 tokens/sec, ~22 minutes for a full response. So 3B NF4 is the best choice for 6GB VRAM.

### Comparison

| Path | Model | VRAM | TTFT (warm) | Tokens/sec | Streaming |
|:---|:---|:---|:---|:---|:---|
| Production | gpt-oss-20b via Groq | 0 | ~4s | N/A (buffered) | Buffered |
| Local | Qwen2.5-3B NF4 | 2.2GB | ~3.7s | 11.8 | Real token stream |
| Local (attempted) | Qwen2.5-7B NF4 | 6GB+ (CPU offload) | ~58s | 0.6 | — |

---

## Retrieval Quality Evaluation

47 queries across all 12 dataset cities. Human-labeled relevance judgments. Fuzzy name matching with accent normalization, city/branch suffix stripping, and token-subset matching. Bootstrap 95% CI.

| Strategy | MRR@5 | 95% CI | Hit@3 | Hit@5 | P@5 | Avg Latency |
|:---|:---|:---|:---|:---|:---|:---|
| Hybrid + Rerank | 0.760 | [0.678, 0.841] | 0.979 | 1.000 | 0.413 | 13ms |
| Hybrid (no rerank) | 0.639 | [0.524, 0.753] | 0.745 | 0.830 | 0.349 | 14ms |

**Key Findings**

- **Ground-truth refinement:** Updated evaluation labels to include valid retrieved restaurants that were previously excluded, added missing location aliases (e.g., `philly`, `nola`, `indy`), and introduced BM25-side synonym expansion for sparse retrieval gaps, reducing false retrieval failures to zero.
- **Reranking impact:** Hybrid retrieval with CrossEncoder reranking consistently outperformed the non-reranked baseline, improving MRR@5 by **+0.121**, Hit@5 by **+0.170**, and P@5 by **+0.064** with comparable latency.
- **RRF sensitivity:** Sweeping `RRF_K` across 10–150 showed negligible performance differences (<0.01 across metrics), indicating that the CrossEncoder reranker contributes most of the final ranking improvement once relevant candidates are retrieved.
- **Query category performance:** Cuisine-based queries achieved the strongest results (**MRR@5 = 0.922**).
- **Geographic filtering:** Location-aware retrieval achieved **100% metro-area accuracy** across the evaluation benchmark.
- **Caching fixes:** Improved query cache reliability by including retrieval parameters (`k_rrf`, `initial_k`, `max_duplicates`) in cache keys and clearing stale cached results after ranking logic changes.

Eval script: `ml_backend/evaluation.py`.

### Generation Quality (DeepEval)

`eval_generation.py` scores the generator's actual output, judged by an LLM, on two metrics:

- **Faithfulness** — answer claims are checked against the retrieved context.
- **Answer Relevancy** — answer is checked against the query.

Judge model: `GroqJudge`, a DeepEval model wrapping this project's own Groq client — no OpenAI key required.

```bash
python eval_generation.py --url http://127.0.0.1:9000 --limit 15
```

### Experiment Tracking (MLflow)

`eval.py` and `eval_generation.py` both support `--mlflow`, logging runs to a local MLflow store instead of stdout only. Retrieval and generation quality become comparable across config changes — RRF_K sweeps, reranker on/off, judge model swaps.

```bash
python eval.py --url http://127.0.0.1:8000 --mlflow
python eval_generation.py --url http://127.0.0.1:9000 --mlflow

mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Runs are under `Model training` in the MLflow UI, not the `GenAI` traces view.

---

## Deployment

### Serverless Backend (Modal)

| Service | Resources |
|:---|:---|
| Retriever | 2 CPU, 2GB RAM |
| Generator | 1 CPU, 0.5GB RAM |

- E5-large-v2 and CrossEncoder pre-cached in a Modal Volume cold starts load from volume (~5-10s), not the internet
- `keep_warm=1` on retriever keeps one container warm
- Query result cache (diskcache / SQLite, 6-hour TTL) repeated queries skip retrieval entirely

### Frontend
Next.js on Vercel. Token streaming, interactive Leaflet map, restaurant cards with Maps/Yelp links, retrieval latency display.

---

## Performance

### Retrieval (shared across both paths)

| Step | Latency |
|:---|:---|
| E5 embedding (CPU) | ~200–500ms |
| Qdrant vector search | ~133–700ms |
| BM25 + RRF | ~0ms (pre-full-corpus-BM25 measurement, needs re-benchmarking) |
| CrossEncoder rerank | ~200–400ms |
| **Total retrieval** | **~700ms–1.2s** |

### End-to-End (warm, deployed)

| Path | Generator | Total |
|:---|:---|:---|
| Production (Groq) | ~4s buffered | ~5–6s |
| Local (2.5-3B NF4) | ~3.7s TTFT + ~60s generation | ~64s |

Variance is inherent to free-tier serverless infrastructure. A paid Qdrant cluster + dedicated CPU would bring down retrieval time consistently.

---


## Project Structure

```
DineRAG/
├── ml_backend/
│   ├── config.py             # Centralized configuration
│   ├── retriever.py          # Hybrid search, RRF, geo-filter (FastAPI)
│   ├── generator_groq.py     # Groq generator, production (FastAPI)
│   ├── generator_local.py    # Local quantized  generator (FastAPI)
│   ├── cache.py              # diskcache SQLite query result cache
│   ├── observability.py      # Prometheus metrics + loguru logging
│   ├── evaluation.py         # MRR, Hit@K, P@K, bootstrap CI (+ optional MLflow logging)
│   └── eval_generation.py    # Faithfulness / answer relevancy via DeepEval (Groq judge)
├── data_pipeline/
│   ├── preprocessor.py       # Raw Yelp NDJSON filtering + scoring
│   ├── chunker.py            # Token-bounded semantic chunking
│   ├── embedder.py           # E5-large-v2 embeddings + full-corpus BM25 index
│   └── ingester.py           # Generator-based Qdrant ingestion
├── deployment/
│   ├── modal_retriever.py    # Modal serverless — retriever
│   └── modal_generator.py    # Modal serverless — generator
├── dags/
│   └── dinerag_pipeline_dag.py  # Airflow DAG 
├── .github/workflows/ci.yml  # pytest on push/PR
├── tests/                    # pytest suite — cache, chunking, eval metrics, generator helpers
├── conftest.py
├── requirements.txt
└── ui/                        # Next.js frontend
```

---

## Performance Optimizations Implemented

1. **Parquet & PyTorch Formats:** Replaced standard CSV/JSON/Pickle intermediate files with **Parquet** (for metadata) and **.pt Tensors** (for vectors). 
2. **Pandas Vectorization:** `chunker.py` used `df.explode()` and `df.melt()` instead of expensive `for` loops. This moves the iteration logic to C-level Pandas optimizations, speeding up processing.
3. **Generator-Based Ingestion:** The `ingestor.py` does not load the full dataset into RAM. It lazily reads from the disk and yields batches to the Qdrant client, allowing the ingestion of datasets larger than system RAM.
4. **Query cache** — diskcache (SQLite-backed) with 6-hour TTL; repeated queries skip retrieval entirely
5. **Qdrant co-location** — database and compute in same cloud region eliminates cross-cloud latency (~200ms saved)

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (for local inference)
- CUDA 11.8 or 12.1
- Docker (for local Qdrant)

### Install

```bash
git clone https://github.com/your-username/dinerag.git
cd dinerag
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

### Configure

`.env`:
```ini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
RETRIEVER_URL=http://127.0.0.1:8000/retrieve
GROQ_API_KEY=gsk_...
GROQ_MODEL_ID=gpt-oss-20b
```

`.env.local` (frontend):
```ini
NEXT_PUBLIC_GENERATOR_URL=http://localhost:9000
NEXT_PUBLIC_STADIA_API_KEY=your_stadia_key
```

### Run Data Pipeline

```bash
python data_pipeline/preprocessor.py
python data_pipeline/chunker.py
python data_pipeline/embedder.py
python data_pipeline/ingester.py
```

**Run as an orchestrated Airflow DAG** (`dags/dinerag_pipeline_dag.py`), 4 tasks in sequence instead of 4 manual invocations. 

```bash
pip install -r orchestration/requirements-airflow.txt   # separate venv, WSL2 on Windows
export AIRFLOW_HOME=~/airflow-home
export AIRFLOW__CORE__DAGS_FOLDER="<repo-root>/dags"
airflow db migrate
airflow standalone
```

### Run Services

```bash
uvicorn ml_backend.retriever:app --port 8000

uvicorn ml_backend.generator_groq:app --port 9000

# Terminal 3
cd ui && npm run dev
```

### Run Evaluation

```bash
python ml_backend/evaluation.py --url http://127.0.0.1:8000 --top_k 8 --verbose
```

### Run Tests

```bash
pytest tests/
```

Runs automatically on every push/PR to `main` via `.github/workflows/ci.yml`.

---

## API

### Generate (user-facing)
`POST /generate`
```json
{ "query": "best tacos in Philadelphia", "top_k": 8 }
```

Response — NDJSON stream:
```
{"type": "ping"}
{"type": "meta", "data": {"retrieval_ms": 850, "results_count": 8, "reranked": true}}
{"type": "sources", "data": [...]}
{"type": "token", "data": "Here"}
{"type": "token", "data": " are"}
...
```

### Retrieve (debug)
`POST /retrieve`
```json
{ "query": "late night ramen", "top_k": 5, "do_rerank": true }
```

