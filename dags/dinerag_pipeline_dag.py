from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airflow.sdk import dag, task


@dag(
    dag_id="dinerag_data_pipeline",
    description="Yelp NDJSON -> preprocess -> chunk -> embed -> ingest (Qdrant)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["dinerag", "data-pipeline"],
    default_args={"retries": 0},
)
def dinerag_data_pipeline():
    @task
    def preprocess():
        from data_pipeline.preprocessor import YelpRestaurantPipeline

        YelpRestaurantPipeline().run()

    @task
    def chunk():
        from data_pipeline.chunker import YelpChunking

        YelpChunking().run()

    @task
    def embed():
        from data_pipeline.embedder import YelpEmbedder

        YelpEmbedder().run()

    @task
    def ingest():
        from data_pipeline.ingester import YelpIngestorQdrant

        YelpIngestorQdrant().run()

    preprocess() >> chunk() >> embed() >> ingest()


dinerag_data_pipeline()
