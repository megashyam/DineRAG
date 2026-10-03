# DineRAG: Production-Grade RAG System on the Yelp Business Review Dataset Entirely from Scratch

**DineRAG** is a high-performance Retrieval-Augmented Generation (RAG) system engineered to provide grounded, location-aware restaurant recommendations  **built entirely from scratch without any RAG framework.** It leverages a **Hybrid Search Architecture** (Dense Vectors + Sparse Keywords) fused with a Cross-Encoder Reranker to retrieve precise context from the Yelp Academic Dataset, which is then synthesized by a dual 4-bit quantized LLM offline route and Claude Haiku online route.

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

To demonstrate a real understanding of everything that happens under the hood of RAG Frameworks. DineRAG is built with full control and zero abstractions. No RAG frameworks. There is no managed loader, no pre-built retriever, no abstracted LLM call.


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

  * [Production - Claude Haiku 4.5 with Groq Fallback](#production---claude-haiku-45-with-groq-fallback)

  * [Local / Offline - Qwen2.5-3B NF4](#local--offline---qwen25-3b-nf4)

  * [Comparison](#comparison)

* [Retrieval Quality Evaluation](#retrieval-quality-evaluation)

* [Deployment](#deployment)

  * [Serverless Backend (Modal)](#serverless-backend-modal)

  * [Frontend](#frontend)

  * [MCP Server (Claude / any MCP client)](#mcp-server-claude--any-mcp-client)

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
                  ┌──────────────────────────────┐
                  │    Raw Yelp JSON (~10GB)     │
                  └──────────────┬───────────────┘
                                 ▼
          ┌──────────────────────────────────────────────┐
          │  OFFLINE DATA PIPELINE                       │
          │  Preprocessor   scoring, filtering, sampling │
          │  Chunker        token-bounded, typed chunks  │
          │  Embedder       E5-large-v2 + BM25 index     │
          └──────────────────────┬───────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │   Qdrant Cloud (vector DB)   │
                  └──────────────┬───────────────┘
                                 ▼
          ┌──────────────────────────────────────────────┐
          │  RETRIEVER SERVICE (FastAPI)                 │
          │  Geo filter        spaCy NER                 │
          │  Dense search      Qdrant HNSW               │
          │  Sparse search     full-corpus BM25          │
          │  Fusion            RRF                       │
          │  Rerank            CrossEncoder              │
          └──────────────────────┬───────────────────────┘
                                 ▼
          ┌──────────────────────────────────────────────┐
          │  GENERATOR SERVICE (FastAPI)                 │
          │  Intent check    greeting/off_topic → canned │
          │  Coverage guard  out of coverage → canned    │
          │  LLM   Claude Haiku 4.5 → Groq gpt-oss-20b   │
          │        Qwen2.5-3B NF4 (local / offline)      │
          └──────────────────────┬───────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  Next.js Frontend (Vercel)   │
                  └──────────────────────────────┘
```

## Data Pipeline — From Raw Yelp JSON to Production Grade Vector DB

The pipeline starts from the raw [Yelp Academic Dataset](https://www.yelp.com/dataset), about 10GB spread over several JSON files. There are no data loaders, preprocessed versions or data APIs involved.

### Challenges solved:

**1. Multi-file joins at scale**: `business.json` and `review.json` are joined on `business_id` as they're streamed, so neither file is ever fully in memory.

**2. Category filtering**: Yelp categories are free text, so restaurants are picked out of 1000+ business types by parsing those lists.

**3. Custom Restaurant Score**: Restaurants are ranked by a weighted score built from rating, review count and recency. Raw stars alone aren't reliable.

**4. Balanced sentiment sampling**: Reviews are sampled from positive, neutral and negative buckets, so the model hears about the downsides too.

**5. Token-bounded chunking**: Chunks are split with E5's own tokenizer, so the token budget matches what the embedding model sees.

**6. Typed chunk structure**: Each restaurant becomes several kinds of chunks (profile, positive, negative and so on) instead of one block of raw text, which makes retrieval more precise.

**7. Embedding at scale**: E5-large-v2 embeddings are generated in batches and saved as `.pt` tensors, so they reload quickly.

**8. Stable deduplication**: Point IDs are UUID5 hashes, so re-running the pipeline doesn't create duplicates.

**9. Generator-based ingestion**: Uploads to Qdrant go through a generator in batches of 256, so memory use stays flat no matter how big the dataset is.

---

## Under the Hood

### Streaming Filter Cascade (`preprocessor.py`)
Reviews are filtered while they're parsed, cheapest check first, so the 20GB+ of reviews never has to sit in RAM:
1. **Business ID**: a set lookup that drops about 95% of records right away.
2. **Date filter**: a plain ISO date string comparison.
3. **Word count**: `text.split()` only runs on reviews that passed the first two checks.

### Composite Restaurant Scoring (`preprocessor.py`)
Ranking by raw stars doesn't work. A place with three 5-star reviews would beat one with 2,000 reviews averaging 4.7. So each restaurant gets two scores:
- `restaurant_score = stars × log1p(review_count)` for long-term reputation
- `reviews_score = mean_stars × log1p(sum_stars)` for the quality of recent reviews

Both are min-max normalized and then combined with tunable weights. Each city has a cap on how many restaurants it keeps, and the cap is higher for dense cities (600+ restaurants) than for sparse ones.

### Balanced Sentiment Sampling (`pipeline.py`)
If you just take the top N reviews, you get nothing but 5-star reviews, and the LLM never hears about wait times, portion sizes or bad service. Reviews are sampled proportionally from positive, neutral and negative buckets. Counts are rounded up so rare sentiment classes don't disappear.

### Token-Bounded Chunking (`chunker.py`)
Chunks are measured with E5's own tokenizer, so the token budget matches what the embedder sees. Each review is truncated by tokens, not characters, before it's packed into a chunk. That way one long review can't use up the whole chunk.

### Typed Chunk Structure (`chunker.py`)
Each restaurant becomes several typed chunks instead of one big block of reviews:
- **Business Profile**: name, location, category, hours
- **Attribute chunks**: parsed from Yelp's nested stringified dicts (WiFi, parking, alcohol, etc.)
- **Vibe chunks**: atmosphere descriptors pulled from the nested attribute maps
- **Positive / Neutral / Negative review batches**: reviews grouped by sentiment

This lets a query find the part of a restaurant it's actually asking about. "Romantic atmosphere" matches vibe chunks, and "avoid if in a rush" matches negative review chunks.

---

## Retrieval

### Hybrid Search

**Dense retrieval**: E5-large-v2 embeddings searched in Qdrant's HNSW index. This catches matches with different wording, like "cheap eats" and "affordable prices".

**Sparse retrieval**: BM25 over every chunk. A small synonym table expands the query, for example "bbq" ↔ "barbecue" and "brunch" ↔ "breakfast".

**RRF fusion**: Reciprocal Rank Fusion merges the two ranked lists. It only uses ranks, so the two kinds of scores don't need normalizing:
```
rrf_score = 1/(k + vec_rank) + 1/(k + bm25_rank)    k=60
```

**Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2` reads the query and each chunk together as `[CLS] Query [SEP] Document`. That makes it more accurate than comparing two separate embeddings.

### Location Extraction
spaCy NER finds the city or state in the query. A fallback dictionary handles nicknames ("Philly" → Philadelphia, "NOLA" → New Orleans) and maps states to cities ("Pennsylvania" → the Philadelphia metro).

---

## Query Routing

Not every message needs a search. Each query goes through two checks before any retrieval happens.

### Intent Classification
Claude Haiku 4.5 (`claude-haiku-4-5`) sorts the query into one of these intents. If the Claude call fails, Groq's `openai/gpt-oss-20b` does it instead:

| Intent | Action |
|:---|:---|
| `food_search` | Proceed to coverage check + retrieval |
| `location_only` | Proceed to retrieval with geo filter |
| `greeting` | Return intro message, skip retrieval |
| `identity` | Return DineRAG description, skip retrieval |
| `off_topic` | Return redirect message, skip retrieval |


### Coverage Guard (Two Layers)
Questions about cities outside the dataset are stopped in two places:

- **Generator-level**: `is_out_of_coverage()` checks the query against the `COVERED_AREAS` / `OUT_OF_COVERAGE` lists before calling the retriever.
- **Retriever-level**: `_detect_raw_location()` catches anything that gets past the first check. It returns `out_of_coverage: true` with no results, so the UI never shows cards for unrelated restaurants.

Both checks detect any location (NER + regex), not just the cities in the dataset. So a query about "Athens" gets blocked, instead of quietly returning results from some other city.

---

## Inference

There are two ways to generate answers: the production path (Claude, falling back to Groq) and a local path that runs a quantized model offline.

### Production - Claude Haiku 4.5 with Groq Fallback
`claude-haiku-4-5` is tried first, and if it fails the request goes to `openai/gpt-oss-20b` on Groq. You can change the order with `GENERATION_PROVIDER_ORDER` (default `claude,groq`). The response stream includes a `provider` frame that says which model answered. No GPU is needed.

Groq path (Modal's ASGI proxy buffers the response):

| Metric | Value |
|:---|:---|
| TTFT (Groq) | ~4s (Modal buffered) |
| Tokens/sec (Groq) | ~400k |
| GPU cost | 0 |

### Local / Offline - Qwen2.5-3B NF4

Qwen2.5-3B-Instruct, quantized to 4-bit NF4 with `bitsandbytes`. These numbers are from an RTX 3060 (6GB):

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

- `attn_implementation="sdpa"`: PyTorch's scaled dot-product attention
- `torch.backends.cuda.matmul.allow_tf32 = True`: TF32 math on Ampere GPUs
- `TextIteratorStreamer` with a background thread: `model.generate()` runs in its own thread, and tokens are streamed from the main thread without blocking the event loop
- `gc.collect()`, `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()` run after every generation so VRAM doesn't fragment over many requests

**Why not 7B?** On paper, Qwen2.5-7B NF4 fits (about 3.5GB of weights), but on a 6GB card it spills onto the CPU during inference. I measured a 58s TTFT and 0.6 tokens/sec, so a full answer took about 22 minutes. 3B NF4 is the best fit for 6GB of VRAM.

### Comparison

| Path | Model | VRAM | TTFT (warm) | Tokens/sec | Streaming |
|:---|:---|:---|:---|:---|:---|
| Production (fallback) | gpt-oss-20b via Groq | 0 | ~4s | N/A (buffered) | Buffered by Modal proxy |
| Local | Qwen2.5-3B NF4 | 2.2GB | ~3.7s | 11.8 | Real token stream |
| Local (attempted) | Qwen2.5-7B NF4 | 6GB+ (CPU offload) | ~58s | 0.6 | — |

---

## Retrieval Quality Evaluation

The test set has 70 queries covering all 12 cities in the dataset, plus some queries that don't name a city. I report MRR@5, Hit@3, Hit@5 and P@5 (always divided by k), each with a bootstrap 95% confidence interval.

**Labels.** Relevance is judged per restaurant (`business_id`). I pooled the top 10 results from every strategy and RRF setting, added the older name-based labels, and marked each of the 1,023 candidates y/n (`eval_labels.py` → `eval_data/label_pool.csv` → `eval_data/qrels.json`). Results nobody judged count as not relevant.

**Measurement.** Caching is turned off (`no_cache=true`). Latency is recorded on both the client and the server, as mean, p50 and p95.

Results from 2026-09-25, with `k_rrf=60`, `initial_k=30`, `max_duplicates=1` and client-side latency:

| Strategy | MRR@5 | 95% CI | Hit@3 | Hit@5 | P@5 | Mean latency | p95 latency |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Hybrid + Rerank | 0.957 | [0.914, 0.993] | 0.986 | 0.986 | 0.883 | 554ms | 706ms |
| Hybrid (no rerank) | 0.948 | [0.900, 0.986] | 0.986 | 0.986 | 0.823 | 477ms | 591ms |

**Key Findings**

- **Reranking:** raises P@5 from 0.823 to 0.883 and adds about 75ms. The MRR@5 gain (+0.010, 95% CI [−0.043, +0.062], W/L/T 4/3/63) is not significant.
- **RRF sensitivity:** the choice of `k_rrf` barely matters. These runs skip reranking, because the reranked output doesn't depend on `k_rrf`:

  | `k_rrf` | 10 | 30 | 60 | 100 | 150 |
  |:---|:---|:---|:---|:---|:---|
  | MRR@5 | 0.944 | 0.946 | 0.948 | 0.948 | 0.948 |
  | P@5 | 0.809 | 0.823 | 0.823 | 0.823 | 0.823 |
  | Hit@5 | 1.000 | 0.986 | 0.986 | 0.986 | 0.986 |
- **Geographic filtering:** 100% of results come from the requested city, because the city filter is strict.
- **Caching:** `k_rrf`, `initial_k` and `max_duplicates` are included in the cache key, so changing them never returns stale cached results.

```bash
# 1. Build the candidate pool
python eval_labels.py pool --url http://127.0.0.1:8000
python eval_labels.py import

# 2. Evaluate (uncached), and sweep RRF k
python eval.py --url http://127.0.0.1:8000 --mlflow --out results.json
python eval.py --url http://127.0.0.1:8000 --mlflow --sweep-rrf 10,30,60,100,150
```

The eval scripts are `eval.py` and `eval_labels.py`.

### Experiment Tracking (MLflow)

Add `--mlflow` to `eval.py` or `eval_generation.py` to log the run to MLflow.

```bash
python eval.py --url http://127.0.0.1:8000 --mlflow
python eval_generation.py --url http://127.0.0.1:9000 --mlflow

mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
---

## Deployment

### Serverless Backend (Modal)

| Service | Resources |
|:---|:---|
| Retriever | 2 CPU, 2GB RAM |
| Generator | 1 CPU, 0.5GB RAM |

- The E5-large-v2 and CrossEncoder weights are stored in a Modal Volume, so a cold start loads them from disk (about 5–10s) instead of downloading them.
- `keep_warm=1` keeps one retriever container running at all times.
- Query results are cached in diskcache (SQLite) for 6 hours, so a repeated query skips retrieval entirely.

### Frontend
A Next.js app on Vercel. It streams answers token by token, shows results on an interactive Leaflet map, has restaurant cards with Google Maps and Yelp links, and displays how long retrieval took.

### MCP Server (Claude / any MCP client)
`mcp_server.py` lets Claude or any other MCP client use DineRAG as a set of tools over stdio. It talks to the deployed services over HTTP, so the only install is `pip install mcp==2.2.0 httpx`.

| Tool | Calls | Returns |
|:---|:---|:---|
| `search_restaurant_reviews(query, top_k=5)` | `POST /retrieve` | Top restaurants (1-20) with city, address, score and a review excerpt, or an out-of-coverage notice |
| `ask_restaurant_question(query, city?, state?)` | `POST /generate` (streamed) | The generated answer plus the restaurants it cited |
| `list_covered_areas()` | `GET /debug/cities` | Covered cities and states (cached for 1 hour) |

Claude Code:
```bash
claude mcp add dinerag -- python /abs/path/to/mcp_server.py
```

Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "dinerag": {
      "command": "python",
      "args": ["/abs/path/to/mcp_server.py"]
    }
  }
}
```

By default it uses the live Modal deployment. To point it at services running locally, set:

| Variable | Default | Local |
|:---|:---|:---|
| `DINERAG_RETRIEVER_URL` | `https://megumind6172--food-rag-retriever-serve.modal.run` | `http://localhost:8000` |
| `DINERAG_GENERATOR_URL` | `https://megumind6172--food-rag-generator-serve.modal.run` | `http://localhost:9000` |
| `DINERAG_TIMEOUT` | `90` (seconds; allows for Modal cold starts) | |

To try the tools in the MCP Inspector: `npx @modelcontextprotocol/inspector python mcp_server.py`

---

## Performance

### Retrieval (shared across both paths)

| Step | Latency |
|:---|:---|
| E5 embedding (CPU) | ~200–500ms |
| Qdrant vector search | ~133–700ms |
| BM25 + RRF | ~0ms (measured before BM25 covered the full corpus; needs re-measuring) |
| CrossEncoder rerank | ~200–400ms |
| **Total retrieval** | **~700ms–1.2s** |

### End-to-End (warm, deployed)

| Path | Generator | Total |
|:---|:---|:---|
| Production (Groq, measured) | ~4s buffered | ~5–6s |
| Local (Qwen2.5-3B NF4) | ~3.7s TTFT + ~60s generation | ~64s |

These numbers jump around because everything runs on free-tier serverless. A paid Qdrant cluster and dedicated CPUs would make retrieval faster and more consistent.

---


## Project Structure

```
DineRAG/
├── ml_backend/
│   ├── config.py             # Centralized configuration
│   ├── retriever.py          # Hybrid search, RRF, geo-filter (FastAPI)
│   ├── generator_groq.py     # Production generator, Claude → Groq fallback (FastAPI)
│   ├── generator_local.py    # Local quantized  generator (FastAPI)
│   ├── cache.py              # diskcache SQLite query result cache
│   ├── observability.py      # Prometheus metrics + loguru logging
│   ├── eval.py               # MRR, Hit@K, P@K, paired bootstrap CIs, RRF sweep (+ optional MLflow logging)
│   ├── eval_labels.py        # Builds the pooled candidate CSV for y/n judging → qrels.json
│   ├── eval_generation.py    # Faithfulness / answer relevancy via DeepEval (Groq judge)
│   └── mcp_server.py         # MCP server (stdio) exposing the deployed services as tools
├── data_pipeline/
│   ├── preprocessor.py       # Raw Yelp NDJSON filtering + scoring
│   ├── chunker.py            # Token-bounded semantic chunking
│   ├── embedder.py           # E5-large-v2 embeddings + full-corpus BM25 index
│   └── ingester.py           # Generator-based Qdrant ingestion
├── deployment/
│   ├── modal_retriever.py    # Modal serverless — retriever
│   └── modal_generator.py    # Modal serverless — generator
├── dags/
│   └── dinerag_pipeline_dag.py  # Airflow DAG wrapping the data_pipeline stages
├── orchestration/
│   └── requirements-airflow.txt  # Airflow's own pin — install in a separate venv
├── .github/workflows/ci.yml  # pytest on push/PR
├── eval_data/                # label_pool.csv (judged candidates) + qrels.json (relevance labels)
├── tests/                    # pytest suite — cache, chunking, eval metrics/labels, generator helpers, MCP server
├── conftest.py
├── requirements.txt
└── ui/                        # Next.js frontend
```

---

## Performance Optimizations Implemented

1. **Parquet & PyTorch Formats:** Intermediate files are **Parquet** (metadata) and **.pt tensors** (vectors) instead of CSV, JSON or pickle.
2. **Pandas Vectorization:** `chunker.py` uses `df.explode()` and `df.melt()` instead of Python `for` loops, so the looping happens inside pandas' C code and runs much faster.
3. **Generator-Based Ingestion:** `ingester.py` never loads the whole dataset into RAM. It reads from disk lazily and hands batches to the Qdrant client, so it can ingest datasets bigger than system memory.
4. **Query cache:** diskcache (backed by SQLite) with a 6-hour TTL. Repeated queries skip retrieval entirely.
5. **Qdrant co-location:** the database and the compute run in the same cloud region, which saves about 200ms of cross-cloud latency.

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
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
# requirements.txt pins CPU torch; for GPU (local inference), install the matching
# CUDA build first, e.g.:
#   pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m spacy download en_core_web_sm
```

### Configure

`.env`:
```ini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
RETRIEVER_URL=http://127.0.0.1:8000/retrieve
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
# Optional (default: claude,groq)
GENERATION_PROVIDER_ORDER=claude,groq
```

You need at least one of `ANTHROPIC_API_KEY` or `GROQ_API_KEY`.

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

**Or run it with Airflow** (`dags/dinerag_pipeline_dag.py`). Airflow 3.3.0 needs its own venv, and on Windows it has to run under WSL2.

```bash
pip install -r orchestration/requirements-airflow.txt   # separate venv, WSL2 on Windows
export AIRFLOW_HOME=~/airflow-home
export AIRFLOW__CORE__DAGS_FOLDER="<repo-root>/dags"
airflow db migrate
airflow standalone
```

### Run Services

```bash
# Terminal 1
uvicorn ml_backend.retriever:app --port 8000

# Terminal 2 — Claude/Groq (production) or generator_local (offline/local GPU)
uvicorn ml_backend.generator_groq:app --port 9000

# Terminal 3
cd ui && npm run dev
```

### Run Evaluation

```bash
python eval.py --url http://127.0.0.1:8000 --verbose
```

### Run Tests

```bash
pytest tests/
```

CI runs the test suite on every push and pull request (`.github/workflows/ci.yml`).

---

## API

### Generate (user-facing)
`POST /generate`
```json
{ "query": "best tacos in Philadelphia", "top_k": 8 }
```

The response is an NDJSON stream:
```
{"type": "ping"}
{"type": "meta", "data": {"retrieval_ms": 850, "results_count": 8, "reranked": true}}
{"type": "sources", "data": [...]}
{"type": "provider", "data": {"provider": "claude", "model": "claude-haiku-4-5"}}
{"type": "token", "data": "Here"}
{"type": "token", "data": " are"}
...
```

### Retrieve (debug)
`POST /retrieve`
```json
{ "query": "late night ramen", "top_k": 5, "do_rerank": true }
```

Optional fields: `k_rrf`, `initial_k`, `max_duplicates`, `no_cache`.

