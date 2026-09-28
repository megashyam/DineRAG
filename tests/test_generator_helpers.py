"""
Tests for generator_groq.py:
  - _safe_excerpt on chunks with and without "--"
  - is_non_food_query / is_out_of_coverage regression cases
  - trim_context does not mutate its input
  - generate_stream provider fallback and on_provider callback
"""

import pytest

import config
# Import only one FastAPI-app module per test process: each calls
# attach_prometheus(), and a second raises "Duplicated timeseries".
from generator_groq import (
    PROVIDER_MODELS,
    RAGGenerator,
    _safe_excerpt,
    format_context_snippets,
    is_non_food_query,
    is_out_of_coverage,
)


class TestSafeExcerpt:
    def test_extracts_review_body_after_double_dash(self):
        text = "passage: positive reviews for X mention:\n -- Great tacos here!"
        assert _safe_excerpt(text) == "Great tacos here!"

    def test_does_not_crash_on_text_without_double_dash(self):
        # Business-profile/attribute/vibe chunks don't use "-- " formatting.
        text = "passage: X is a Italian restaurant located at 123 Main St in Testville, TS."
        assert _safe_excerpt(text) == text

    def test_does_not_crash_on_none(self):
        assert _safe_excerpt(None) == ""

    def test_does_not_crash_on_empty_string(self):
        assert _safe_excerpt("") == ""

    def test_truncates_to_max_len(self):
        text = "-- " + ("a" * 1000)
        assert len(_safe_excerpt(text, max_len=50)) == 50


class TestNonFoodQuery:
    @pytest.mark.parametrize(
        "query",
        ["breakfast", "brunch", "dinner", "burgers", "sashimi",
         "best hot dog places in philadelphia", "sushi restaurants near me",
         "restaurant with parking in boise", "cheapest restaurants in tampa"],
    )
    def test_legitimate_food_queries_are_not_blocked(self, query):
        # These were previously false-positive-blocked by NON_FOOD_PATTERNS
        # (e.g. the old `^[a-z]{6,}$` pattern, or bare "dog"/"parking"/
        # "cheapest" matches) and were fixed with inline `# FIX:` patches in
        # config.py. This test pins that behavior so it can't silently regress.
        assert not is_non_food_query(query)

    @pytest.mark.parametrize(
        "query",
        ["hi", "hello", "thanks", "asdfgh", "what is the capital of france",
         "write a python script", "who won the super bowl"],
    )
    def test_off_topic_queries_are_blocked(self, query):
        assert is_non_food_query(query)


class TestOutOfCoverage:
    def test_covered_city_is_not_out_of_coverage(self):
        assert not is_out_of_coverage("best tacos in Philadelphia")

    def test_uncovered_city_is_out_of_coverage(self):
        assert is_out_of_coverage("best pizza in Chicago")

    def test_query_with_no_location_is_in_coverage(self):
        assert not is_out_of_coverage("upscale steakhouse")

    def test_philly_nickname_is_covered(self):
        assert not is_out_of_coverage("best ramen in Philly")


class TestTrimContext:
    def test_does_not_mutate_input(self):
        snippets = [{"restaurant": "A", "text": "x" * 1000}]
        trimmed = RAGGenerator().trim_context(snippets, trim_length=10)
        assert trimmed[0]["text"] == "x" * 10
        assert snippets[0]["text"] == "x" * 1000
        assert trimmed[0] is not snippets[0]

    def test_stops_at_total_char_budget(self):
        snippets = [{"text": "a" * 100}, {"text": "b" * 100}, {"text": "c" * 100}]
        trimmed = RAGGenerator().trim_context(snippets, max_chars=250, trim_length=100)
        assert [s["text"][0] for s in trimmed] == ["a", "b"]


class TestFormatContextSnippets:
    def test_formats_each_result(self):
        out = format_context_snippets(
            [{"restaurant": "Gumbo Shop", "address": "630 St Peter St", "city": "new orleans",
              "text": "Great gumbo."}]
        )
        assert out == [
            "Restaurant: Gumbo Shop\n"
            "Location: 630 St Peter St (new orleans)\n"
            "Description: Great gumbo.\n"
            "---"
        ]

    def test_prompt_uses_the_same_snippets(self):
        results = [{"restaurant": "A", "address": "1 Main", "city": "reno", "text": "ok"}]
        _, user_msg = RAGGenerator()._build_prompt_content("q", results)
        assert format_context_snippets(results)[0] in user_msg


class TestGenerateStreamFallback:
    def _generator(self, monkeypatch, order, streamers):
        gen = RAGGenerator()
        gen.clients = {name: object() for name in streamers}
        for name, fn in streamers.items():
            monkeypatch.setattr(gen, f"_generate_stream_{name}", fn)
        monkeypatch.setattr(config, "GENERATION_PROVIDER_ORDER", order)
        return gen

    @staticmethod
    def _ok(*tokens):
        def streamer(client, query, ctx):
            yield from tokens
        return streamer

    @staticmethod
    def _fails_before_first_token(client, query, ctx):
        raise RuntimeError("auth failed")
        yield  # pragma: no cover — makes this a generator

    def test_uses_first_provider_and_reports_it(self, monkeypatch):
        gen = self._generator(
            monkeypatch, ["claude", "groq"],
            {"claude": self._ok("hi", " there"), "groq": self._ok("groq")},
        )
        seen = []
        tokens = list(gen.generate_stream("q", [], on_provider=lambda p, m: seen.append((p, m))))
        assert tokens == ["hi", " there"]
        assert seen == [("claude", PROVIDER_MODELS["claude"])]

    def test_falls_back_when_first_provider_fails_before_first_token(self, monkeypatch):
        gen = self._generator(
            monkeypatch, ["claude", "groq"],
            {"claude": self._fails_before_first_token, "groq": self._ok("from groq")},
        )
        seen = []
        tokens = list(gen.generate_stream("q", [], on_provider=lambda p, m: seen.append((p, m))))
        assert tokens == ["from groq"]
        assert seen == [("groq", PROVIDER_MODELS["groq"])]

    def test_respects_configured_order(self, monkeypatch):
        gen = self._generator(
            monkeypatch, ["groq", "claude"],
            {"claude": self._ok("claude"), "groq": self._ok("groq")},
        )
        assert list(gen.generate_stream("q", [])) == ["groq"]

    def test_raises_last_error_when_all_providers_fail(self, monkeypatch):
        gen = self._generator(
            monkeypatch, ["claude", "groq"],
            {"claude": self._fails_before_first_token, "groq": self._fails_before_first_token},
        )
        with pytest.raises(RuntimeError, match="auth failed"):
            list(gen.generate_stream("q", []))


class _SignatureCheckedClaude:
    """Fake anthropic client; messages.create rejects kwargs the installed SDK's signature doesn't accept."""

    def __init__(self, response):
        import inspect

        import anthropic

        self._sig = inspect.signature(anthropic.resources.Messages.create)
        self._response = response
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self._sig.bind(None, **kwargs)
        self.calls.append(kwargs)
        return self._response


class TestClaudeCallsMatchSdk:
    def test_intent_classification_call_is_accepted_by_sdk(self):
        from types import SimpleNamespace

        client = _SignatureCheckedClaude(
            SimpleNamespace(content=[SimpleNamespace(text=" RESTAURANT ")])
        )
        assert RAGGenerator()._classify_intent_claude(client, "tacos") == "restaurant"
        assert client.calls[0]["extra_body"] == {"temperature": 0.0}

    def test_stream_call_is_accepted_by_sdk(self):
        from types import SimpleNamespace

        events = [
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hi")),
            SimpleNamespace(type="message_stop"),
        ]
        client = _SignatureCheckedClaude(iter(events))
        snippets = [{"restaurant": "A", "city": "x", "text": "good food"}]
        out = list(RAGGenerator()._generate_stream_claude(client, "tacos", snippets))
        assert out == ["Hi"]
        assert client.calls[0]["extra_body"]["temperature"] == config.TEMPERATURE
