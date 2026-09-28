"""
Generation quality evaluation (DeepEval).

Scores generator answers for Faithfulness (claims supported by the retrieved
context) and Answer Relevancy (addresses the query), using a Groq or claude-*
LLM judge.
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import httpx
from tabulate import tabulate
from groq import Groq, GroqError

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from evaluation import TEST_QUERIES, bootstrap_ci

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


#  Retry helper


def _with_retries(fn, *, retries: int, base_delay: float, retryable: tuple, label: str):
    """Calls fn() with exponential backoff on `retryable`; re-raises the last error."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except retryable as e:
            last_exc = e
            if attempt == retries:
                break
            delay = base_delay * (2**attempt)
            print(
                f"       ! {label} failed ({e}); retrying in {delay:.1f}s [{attempt + 1}/{retries}]"
            )
            time.sleep(delay)
    raise last_exc


#  Groq-backed judge model


class JudgeQuotaExhausted(Exception):
    """Judge hit a Groq per-day limit; stops the run instead of retrying."""


def _is_daily_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "per day" in msg and ("rate limit" in msg or "rate_limit" in msg)


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by Groq chat completions."""

    def load_model(self):

        return Groq(api_key=config.GROQ_API_KEY, max_retries=8)

    def generate(self, prompt: str) -> str:
        def _call() -> str:
            try:
                return _request()
            except GroqError as e:
                if _is_daily_limit(e):
                    raise JudgeQuotaExhausted(str(e)) from e
                raise

        def _request() -> str:
            resp = self.model.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                reasoning_effort="low",
            )
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("judge returned empty content")
            return content

        return _with_retries(
            _call,
            retries=3,
            base_delay=2.0,
            retryable=(GroqError, httpx.HTTPError, RuntimeError),
            label="judge call",
        )

    async def a_generate(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self.name


class ClaudeJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the Anthropic Messages API.

    The model must differ from the generator's; the self-judge guard enforces this.
    """

    def load_model(self):
        return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=6)

    def generate(self, prompt: str) -> str:
        def _call() -> str:
            resp = self.model.messages.create(
                model=self.name,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "refusal":
                raise RuntimeError("judge refused the request")
            content = "".join(b.text for b in resp.content if b.type == "text")
            if not content:
                raise RuntimeError("judge returned empty content")
            return content

        return _with_retries(
            _call,
            retries=2,
            base_delay=2.0,
            retryable=(RuntimeError, anthropic.APIConnectionError),
            label="judge call",
        )

    async def a_generate(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self.name


def make_judge(judge_model: str) -> DeepEvalBaseLLM:
    """Returns a ClaudeJudge for claude-* models, else a GroqJudge."""
    if judge_model.startswith("claude-"):
        return ClaudeJudge(judge_model)
    return GroqJudge(judge_model)


class GenerationError(Exception):
    """Generator streamed an `error` frame."""


def fetch_generation(base_url: str, query: str, top_k: int) -> Dict:
    """Collects a /generate stream into {answer, context, context_source, provider, model}.

    context is the prompt context if the generator sent it, else source excerpts.
    Retries httpx errors; raises GenerationError on an `error` frame.
    """

    def _call() -> Dict:
        answer_parts: List[str] = []
        prompt_context: Optional[List[str]] = None
        excerpts: List[str] = []
        provider = model = None

        with httpx.Client(timeout=90.0) as client:
            with client.stream(
                "POST",
                f"{base_url}/generate",
                json={"query": query, "top_k": top_k, "include_context": True},
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    frame = json.loads(line)
                    ftype = frame.get("type")
                    if ftype == "context":
                        prompt_context = list(frame["data"])
                    elif ftype == "sources":
                        excerpts = [
                            s["excerpt"] for s in frame["data"] if s.get("excerpt")
                        ]
                    elif ftype == "provider":
                        provider = frame["data"].get("provider")
                        model = frame["data"].get("model")
                    elif ftype == "token":
                        answer_parts.append(frame["data"])
                    elif ftype == "error":
                        raise GenerationError(frame.get("data"))

        return {
            "answer": "".join(answer_parts),
            "context": prompt_context if prompt_context is not None else excerpts,
            "context_source": "prompt" if prompt_context is not None else "excerpts",
            "provider": provider,
            "model": model,
        }

    return _with_retries(
        _call,
        retries=2,
        base_delay=2.0,
        retryable=(httpx.HTTPError,),
        label="generator call",
    )


def fetch_generator_info(base_url: str) -> Optional[Dict]:
    """Returns the generator's /health payload, or None if unreachable or it lacks `models`."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=15.0)
        resp.raise_for_status()
        info = resp.json()
    except Exception as e:
        print(f"  ! could not read generator /health: {e}")
        return None
    if "models" not in info:
        return None
    return info


def primary_generation_model(
    info: Optional[Dict],
) -> Tuple[Optional[str], Optional[str]]:
    """Returns the generator's first (provider, model), falling back to local config."""
    local_models = {"claude": config.CLAUDE_MODEL_ID, "groq": config.GROQ_MODEL_ID}
    if info:
        order = info.get("provider_order") or info.get("providers") or []
        if order:
            return order[0], info["models"].get(order[0])
    for p in config.GENERATION_PROVIDER_ORDER:
        if p in local_models:
            return p, local_models[p]
    return None, None


#  Query sampling ─


def stratified_sample(
    queries: List[dict], limit: Optional[int], seed: int = 42
) -> List[dict]:
    """Returns up to limit queries (all if 0/None), interleaved round-robin by city.

    Interleaving keeps partial runs (small --limit, or quota cut-off) spread across cities.
    """
    if not limit or limit >= len(queries):
        limit = len(queries)

    rng = random.Random(seed)
    buckets: Dict[Optional[str], List[dict]] = defaultdict(list)
    for q in queries:
        buckets[q.get("city")].append(q)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    cities = list(buckets.keys())
    rng.shuffle(cities)

    sample: List[dict] = []
    i = 0
    while len(sample) < limit and any(buckets.values()):
        city = cities[i % len(cities)]
        if buckets[city]:
            sample.append(buckets[city].pop())
        i += 1
    return sample


#  Evaluator


def evaluate_generation(
    url: str,
    top_k: int = 5,
    limit: Optional[int] = 15,
    judge_model: str = config.GROQ_MODEL_ID,
    out: Optional[str] = None,
    use_mlflow: bool = False,
    seed: int = 42,
    allow_self_judge: bool = False,
    generator_info: Optional[Dict] = None,
) -> dict:
    queries = stratified_sample(TEST_QUERIES, limit, seed=seed)

    print(f"\n{'='*72}")
    print("  DineRAG Generation Quality Evaluation (DeepEval)")
    print(f"{'='*72}")
    print(f"  Generator : {url}")
    if generator_info:
        order = generator_info.get("provider_order") or generator_info.get("providers")
        models = generator_info.get("models", {})
        print(f"  Providers : {' → '.join(f'{p} ({models.get(p)})' for p in order)}")
    via = "Anthropic" if judge_model.startswith("claude-") else "Groq"
    print(f"  Judge     : {judge_model} (via {via})")
    print(f"  Queries   : {len(queries)} / {len(TEST_QUERIES)} (seed={seed})")
    print(f"{'='*72}\n")

    judge = make_judge(judge_model)
    faithfulness = FaithfulnessMetric(threshold=0.5, model=judge, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=0.5, model=judge, async_mode=False)

    rows, records, failures = [], [], []
    stopped_early_at: Optional[int] = None
    for i, q in enumerate(queries):
        query = q["query"]
        print(f"  [{i+1:02d}/{len(queries)}] {query}")

        try:
            gen = fetch_generation(url, query, top_k)
        except GenerationError as e:
            print(f"       ! generator returned an error frame: {e}")
            failures.append(
                {"query": query, "stage": "generation_error", "error": str(e)}
            )
            continue
        except Exception as e:
            print(f"       ! generation failed: {e}")
            failures.append({"query": query, "stage": "generation", "error": str(e)})
            continue

        answer, context = gen["answer"], gen["context"]
        if not answer or not context:
            print("       skipped (empty answer or no retrieved context)")
            failures.append(
                {
                    "query": query,
                    "stage": "empty",
                    "error": "empty answer or no retrieved context",
                }
            )
            continue
        if gen["context_source"] != "prompt":
            print(
                "       ! generator sent no `context` frame (older build?) — judging against "
                "source excerpts, which are NOT what the model saw"
            )

        test_case = LLMTestCase(
            input=query, actual_output=answer, retrieval_context=context
        )

        try:
            faithfulness.measure(test_case)
            relevancy.measure(test_case)
        except JudgeQuotaExhausted as e:
            print(
                f"       ! judge daily quota exhausted — stopping after {i} queries: {e}"
            )
            failures.append({"query": query, "stage": "judge_quota", "error": str(e)})
            stopped_early_at = i
            break
        except Exception as e:
            print(f"       ! judge scoring failed: {e}")
            failures.append({"query": query, "stage": "judge", "error": str(e)})
            continue

        self_judged = gen["model"] == judge_model
        flag = "  ! self-judged" if self_judged else ""
        print(
            f"       Faithfulness={faithfulness.score:.2f}  Relevancy={relevancy.score:.2f}  "
            f"[{gen['provider'] or '?'}]{flag}"
        )

        rows.append(
            [
                query[:55],
                gen["provider"] or "?",
                f"{faithfulness.score:.2f}",
                f"{relevancy.score:.2f}",
                "yes" if self_judged else "",
            ]
        )
        records.append(
            {
                "query": query,
                "answer": answer,
                "context": context,
                "context_source": gen["context_source"],
                "provider": gen["provider"],
                "model": gen["model"],
                "self_judged": self_judged,
                "faithfulness": faithfulness.score,
                "faithfulness_reason": faithfulness.reason,
                "answer_relevancy": relevancy.score,
                "answer_relevancy_reason": relevancy.reason,
            }
        )

    attempted = len(queries)
    provider_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        provider_counts[r["provider"] or "unknown"] += 1
    self_judged_records = [r for r in records if r["self_judged"]]

    counted = (
        records if allow_self_judge else [r for r in records if not r["self_judged"]]
    )
    scored = len(counted)
    completion_rate = scored / attempted if attempted else 0.0

    print(f"\n\n{'='*72}")
    print("  RESULTS")
    print(f"{'='*72}\n")
    print(
        tabulate(
            rows,
            headers=[
                "Query",
                "Provider",
                "Faithfulness",
                "Answer Relevancy",
                "Self-judged",
            ],
            tablefmt="rounded_outline",
        )
    )

    print(f"\n  Attempted  : {attempted}")
    print(f"  Scored     : {scored}")
    print(f"  Completion : {completion_rate:.1%}")
    if stopped_early_at is not None:
        print(
            f"  Stopped    : judge daily quota hit at query {stopped_early_at + 1}; "
            f"{attempted - stopped_early_at - 1} not attempted"
        )
    print(f"  Providers  : {dict(provider_counts)}")
    if self_judged_records:
        action = (
            "included (--allow-self-judge)"
            if allow_self_judge
            else "excluded from means"
        )
        print(
            f"  Self-judged: {len(self_judged_records)} (generated by {judge_model}) — {action}"
        )
    if failures:
        by_stage: Dict[str, int] = defaultdict(int)
        for f in failures:
            by_stage[f["stage"]] += 1
        print(f"  Failures   : {dict(by_stage)}")

    summary = {
        "judge_model": judge_model,
        "top_k": top_k,
        "seed": seed,
        "attempted": attempted,
        "scored": scored,
        "completion_rate": completion_rate,
        "stopped_early_at": stopped_early_at,
        "provider_counts": dict(provider_counts),
        "n_self_judged": len(self_judged_records),
        "self_judged_excluded": bool(self_judged_records) and not allow_self_judge,
        "generator_info": generator_info,
        "failures": failures,
        "avg_faithfulness": None,
        "avg_answer_relevancy": None,
        "faithfulness_ci": None,
        "answer_relevancy_ci": None,
        "records": records,
    }

    if counted:
        faith_vals = [r["faithfulness"] for r in counted]
        rel_vals = [r["answer_relevancy"] for r in counted]
        avg_faith = statistics.mean(faith_vals)
        avg_rel = statistics.mean(rel_vals)
        faith_ci = bootstrap_ci(faith_vals)
        rel_ci = bootstrap_ci(rel_vals)

        summary.update(
            avg_faithfulness=avg_faith,
            avg_answer_relevancy=avg_rel,
            faithfulness_ci=faith_ci,
            answer_relevancy_ci=rel_ci,
        )

        print(
            f"\n  Mean Faithfulness    : {avg_faith:.3f}  95% CI [{faith_ci[0]:.3f}, {faith_ci[1]:.3f}]  (n={len(counted)})"
        )
        print(
            f"  Mean Answer Relevancy: {avg_rel:.3f}  95% CI [{rel_ci[0]:.3f}, {rel_ci[1]:.3f}]"
        )

        low_faith = [r for r in counted if r["faithfulness"] < 0.5]
        if low_faith:
            print(
                f"\n  ! {len(low_faith)} answer(s) below faithfulness threshold (0.5):"
            )
            for r in low_faith:
                print(f"    - {r['query']}")
                print(f"      {r['faithfulness_reason']}")

        if use_mlflow:
            _log_to_mlflow(judge_model, top_k, summary)
    else:
        print("\n  No records scored — nothing to summarize.")

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results written to: {out_path}")

    return summary


def _log_to_mlflow(judge_model: str, top_k: int, summary: dict):
    import os

    import mlflow

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlruns_dir = Path(_REPO_ROOT) / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}")

    mlflow.set_experiment("dinerag-generation-eval")
    with mlflow.start_run(run_name=f"judge={judge_model}"):
        mlflow.log_params(
            {
                "judge_model": judge_model,
                "top_k": top_k,
                "seed": summary["seed"],
                "n_attempted": summary["attempted"],
                "judge_context": "prompt_snippets",
                "self_judged_excluded": summary["self_judged_excluded"],
                "stopped_early": summary["stopped_early_at"] is not None,
            }
        )
        mlflow.log_metrics(
            {
                **{f"n_provider_{p}": n for p, n in summary["provider_counts"].items()},
                "n_self_judged": summary["n_self_judged"],
                "faithfulness": summary["avg_faithfulness"],
                "answer_relevancy": summary["avg_answer_relevancy"],
                "faithfulness_ci_lo": summary["faithfulness_ci"][0],
                "faithfulness_ci_hi": summary["faithfulness_ci"][1],
                "answer_relevancy_ci_lo": summary["answer_relevancy_ci"][0],
                "answer_relevancy_ci_hi": summary["answer_relevancy_ci"][1],
                "n_scored": summary["scored"],
                "completion_rate": summary["completion_rate"],
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DineRAG Generation Quality Evaluator (DeepEval)"
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:9000", help="Generator base URL"
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of test queries to run, stratified round-robin across cities (each query "
        "costs several LLM judge calls). Use 0 to run the full set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for stratified query sampling."
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model: a Groq model ID, or a claude-* model (judged via the Anthropic API). "
        "Must differ from the generator's primary model (read "
        "from its /health) unless --allow-self-judge is passed — grading your own output has a "
        "known self-preference bias. Answers that fall back to the judge's model are excluded.",
    )
    parser.add_argument(
        "--allow-self-judge",
        action="store_true",
        help="Allow the judge model to be the same as a generation model, and keep self-judged "
        "answers in the means.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to write full results JSON. Defaults to a timestamped file under eval_runs/.",
    )
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument(
        "--min-completion-rate",
        type=float,
        default=0.9,
        help="Exit 1 if the fraction of attempted queries that were scored falls below this.",
    )
    parser.add_argument(
        "--fail-under-faithfulness",
        type=float,
        default=None,
        help="Exit 1 if mean faithfulness falls below this.",
    )
    parser.add_argument(
        "--fail-under-relevancy",
        type=float,
        default=None,
        help="Exit 1 if mean answer relevancy falls below this.",
    )
    args = parser.parse_args()

    judge_model = args.judge_model or config.GROQ_MODEL_ID
    generator_info = fetch_generator_info(args.url)
    primary_provider, primary_model = primary_generation_model(generator_info)
    if judge_model == primary_model and not args.allow_self_judge:
        print(
            f"Refusing to run: judge model ({judge_model}) is the generator's primary model "
            f"({primary_provider}). A model grading its own output has a known self-preference "
            "bias. Pass --judge-model <a-different-model>, or --allow-self-judge to run anyway.",
            file=sys.stderr,
        )
        sys.exit(2)
    fallback_models = (generator_info or {}).get("models", {}).values()
    if judge_model in fallback_models and not args.allow_self_judge:
        print(
            f"  Note: judge {judge_model} is the generator's fallback model — any answer that "
            "falls back to it will be flagged and excluded from the means."
        )

    out = args.out
    if out is None:
        out_dir = Path(_REPO_ROOT) / "eval_runs"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = str(out_dir / f"generation_eval_{timestamp}.json")

    summary = evaluate_generation(
        url=args.url,
        top_k=args.top_k,
        limit=None if args.limit == 0 else args.limit,
        judge_model=judge_model,
        out=out,
        use_mlflow=args.mlflow,
        seed=args.seed,
        allow_self_judge=args.allow_self_judge,
        generator_info=generator_info,
    )

    exit_code = 0

    if summary["completion_rate"] < args.min_completion_rate:
        print(
            f"\n  ✗ completion rate {summary['completion_rate']:.1%} is below "
            f"--min-completion-rate {args.min_completion_rate:.1%}",
            file=sys.stderr,
        )
        exit_code = 1

    if args.fail_under_faithfulness is not None and (
        summary["avg_faithfulness"] is None
        or summary["avg_faithfulness"] < args.fail_under_faithfulness
    ):
        print(
            f"\n  ✗ mean faithfulness below --fail-under-faithfulness {args.fail_under_faithfulness}",
            file=sys.stderr,
        )
        exit_code = 1

    if args.fail_under_relevancy is not None and (
        summary["avg_answer_relevancy"] is None
        or summary["avg_answer_relevancy"] < args.fail_under_relevancy
    ):
        print(
            f"\n  ✗ mean answer relevancy below --fail-under-relevancy {args.fail_under_relevancy}",
            file=sys.stderr,
        )
        exit_code = 1

    sys.exit(exit_code)
