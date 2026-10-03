"""
Modal deployment for the DineRAG Retriever service.


Deploy:
    modal deploy deployment/modal_retriever.py
"""

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "qdrant-client",
        "sentence-transformers",
        "spacy",
        "rank-bm25",
        "pandas",
        "pyarrow",
        "numpy",
        "python-dotenv",
        "diskcache",
        "loguru",
        "prometheus-fastapi-instrumentator",
        "prometheus-client",
    )
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_file(_ROOT / "ml_backend/retriever.py", "/app/ml_backend/retriever.py")
    .add_local_file(_ROOT / "ml_backend/cache.py", "/app/ml_backend/cache.py")
    .add_local_file(
        _ROOT / "ml_backend/observability.py", "/app/ml_backend/observability.py"
    )
    .add_local_file(_ROOT / "config.py", "/app/config.py")
)


model_vol = modal.Volume.from_name("food-rag-model-cache", create_if_missing=True)
cache_vol = modal.Volume.from_name("food-rag-query-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("food-rag-data", create_if_missing=True)


app = modal.App("food-rag-retriever", image=image)


@app.function(
    volumes={"/model-cache": model_vol},
    timeout=600,
)
def download_models():
    """Downloads the E5 embedder and cross-encoder into the model-cache Volume. Run once before deploying."""
    import os

    os.environ["HF_HOME"] = "/model-cache"

    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("Downloading intfloat/e5-large-v2 (~1.2GB)...")
    SentenceTransformer("intfloat/e5-large-v2")

    print("Downloading cross-encoder/ms-marco-MiniLM-L-6-v2...")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    print("Committing to volume...")
    model_vol.commit()
    print("Done. Models are cached — cold starts will be fast from now on.")


_data_upload_image = image.add_local_file(
    _ROOT / "data/bm25_ranks.pkl", "/tmp/bm25_ranks.pkl"
).add_local_file(
    _ROOT / "data/retriever_metadata.parquet", "/tmp/retriever_metadata.parquet"
)


@app.function(
    image=_data_upload_image,
    volumes={"/data": data_vol},
    timeout=600,
)
def upload_data():
    """Copies the BM25 index and chunk metadata into the data Volume."""
    import shutil

    shutil.copy("/tmp/bm25_ranks.pkl", "/data/bm25_ranks.pkl")
    shutil.copy("/tmp/retriever_metadata.parquet", "/data/retriever_metadata.parquet")

    print("Committing to volume...")
    data_vol.commit()
    print("Done. Full-corpus BM25 index is available to the retriever.")


@app.function(
    volumes={
        "/model-cache": model_vol,
        "/cache": cache_vol,
        "/data": data_vol,
    },
    secrets=[modal.Secret.from_name("food-rag-secrets")],
    cpu=2.0,
    memory=2000,
    timeout=120,
    scaledown_window=500,
)
@modal.asgi_app()
def serve():
    import os
    import sys

    sys.path.insert(0, "/app")

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["CACHE_DIR"] = "/cache"

    import config

    config.BM25_PATH = Path("/data/bm25_ranks.pkl")
    config.METADATA_PATH = Path("/data/retriever_metadata.parquet")

    from ml_backend.retriever import app

    return app
