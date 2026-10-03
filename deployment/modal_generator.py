"""
Modal deployment for the DineRAG Generator service.

Deploy (from the repo root):
    modal deploy deployment/modal_generator.py
"""

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "groq",
        "anthropic",
        "httpx",
        "python-dotenv",
        "loguru",
        "prometheus-fastapi-instrumentator",
        "prometheus-client",
        "wrapt",
    )
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_file(
        _ROOT / "ml_backend/generator_groq.py", "/app/ml_backend/generator_groq.py"
    )
    .add_local_file(
        _ROOT / "ml_backend/observability.py", "/app/ml_backend/observability.py"
    )
    .add_local_file(_ROOT / "config.py", "/app/config.py")
)


app = modal.App("food-rag-generator", image=image)


@app.function(
    secrets=[modal.Secret.from_name("food-rag-secrets")],
    cpu=1.0,
    memory=500,
    timeout=120,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    import sys

    sys.path.insert(0, "/app")

    from ml_backend.generator_groq import app

    return app
