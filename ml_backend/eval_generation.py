import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from tabulate import tabulate
from groq import Groq

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from evaluation import TEST_QUERIES

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


class GroqJudge(DeepEvalBaseLLM):
    def load_model(self):
        return Groq(api_key=config.GROQ_API_KEY)

    def generate(self, prompt: str) -> str:
        resp = self.model.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            reasoning_effort="low",
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self.name


def fetch_generation(base_url: str, query: str, top_k: int) -> Tuple[str, List[str]]:
    answer_parts: List[str] = []
    context: List[str] = []

    with httpx.Client(timeout=90.0) as client:
        with client.stream(
            "POST", f"{base_url}/generate", json={"query": query, "top_k": top_k}
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                frame = json.loads(line)
                if frame["type"] == "sources":
                    context = [
                        s.get("excerpt", "") for s in frame["data"] if s.get("excerpt")
                    ]
                elif frame["type"] == "token":
                    answer_parts.append(frame["data"])

    return "".join(answer_parts), context


def evaluate_generation(
    url: str,
    top_k: int = 5,
    limit: Optional[int] = 15,
    judge_model: str = config.GROQ_MODEL_ID,
    out: Optional[str] = None,
    use_mlflow: bool = False,
):
    queries = TEST_QUERIES[:limit] if limit else TEST_QUERIES

    print(f"\n{'='*72}")
    print("  DineRAG Generation Quality Evaluation (DeepEval)")
    print(f"{'='*72}")
    print(f"  Generator : {url}")
    print(f"  Judge     : {judge_model} (via Groq)")
    print(f"  Queries   : {len(queries)}")
    print(f"{'='*72}\n")

    judge = GroqJudge(judge_model)
    faithfulness = FaithfulnessMetric(threshold=0.5, model=judge, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=0.5, model=judge, async_mode=False)

    rows, records = [], []
    for i, q in enumerate(queries):
        query = q["query"]
        print(f"  [{i+1:02d}/{len(queries)}] {query}")

        try:
            answer, context = fetch_generation(url, query, top_k)
        except Exception as e:
            print(f"       generation failed: {e}")
            continue

        if not answer or not context:
            print("       skipped (empty answer or no retrieved context)")
            continue

        test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=context)

        try:
            faithfulness.measure(test_case)
            relevancy.measure(test_case)
        except Exception as e:
            print(f"       judge scoring failed: {e}")
            continue

        print(f"       Faithfulness={faithfulness.score:.2f}  Relevancy={relevancy.score:.2f}")

        rows.append([query[:55], f"{faithfulness.score:.2f}", f"{relevancy.score:.2f}"])
        records.append(
            {
                "query": query,
                "answer": answer,
                "context": context,
                "faithfulness": faithfulness.score,
                "faithfulness_reason": faithfulness.reason,
                "answer_relevancy": relevancy.score,
                "answer_relevancy_reason": relevancy.reason,
            }
        )

    print(f"\n\n{'='*72}")
    print("  RESULTS")
    print(f"{'='*72}\n")
    print(
        tabulate(
            rows,
            headers=["Query", "Faithfulness", "Answer Relevancy"],
            tablefmt="rounded_outline",
        )
    )

    if records:
        avg_faith = statistics.mean(r["faithfulness"] for r in records)
        avg_rel = statistics.mean(r["answer_relevancy"] for r in records)
        print(f"\n  Mean Faithfulness    : {avg_faith:.3f}  (n={len(records)})")
        print(f"  Mean Answer Relevancy: {avg_rel:.3f}")

        low_faith = [r for r in records if r["faithfulness"] < 0.5]
        if low_faith:
            print(f"\n  {len(low_faith)} answer(s) below faithfulness threshold (0.5):")
            for r in low_faith:
                print(f"    - {r['query']}")
                print(f"      {r['faithfulness_reason']}")

        if use_mlflow:
            _log_to_mlflow(judge_model, top_k, records, avg_faith, avg_rel)
    else:
        print("\n  No records scored — nothing to summarize.")

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"\n  Results written to: {out}")

    return records


def _log_to_mlflow(judge_model: str, top_k: int, records: list, avg_faith: float, avg_rel: float):
    import mlflow

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlruns_dir = Path(__file__).resolve().parent.parent / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}")

    mlflow.set_experiment("dinerag-generation-eval")
    with mlflow.start_run(run_name=f"judge={judge_model}"):
        mlflow.log_params({"judge_model": judge_model, "top_k": top_k, "n_queries": len(records)})
        mlflow.log_metrics({"faithfulness": avg_faith, "answer_relevancy": avg_rel})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DineRAG Generation Quality Evaluator (DeepEval)")
    parser.add_argument("--url", default="http://127.0.0.1:9000", help="Generator base URL")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of test queries to run (each costs several LLM judge calls). "
        "Use 0 to run the full set.",
    )
    parser.add_argument("--judge-model", default=config.GROQ_MODEL_ID)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    evaluate_generation(
        url=args.url,
        top_k=args.top_k,
        limit=None if args.limit == 0 else args.limit,
        judge_model=args.judge_model,
        out=args.out,
        use_mlflow=args.mlflow,
    )
