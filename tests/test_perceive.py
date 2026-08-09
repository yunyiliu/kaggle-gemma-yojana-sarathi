"""Tests for the boundary between the model and the rest of the system.

These use scripted model replies rather than a live model. They are not testing whether
Gemma extracts Hindi well - that is measured separately in reports/perception_eval.md.
They are testing that whatever the model returns, only well-formed facts get past this
layer, because everything downstream trusts what comes out of here.
"""
from __future__ import annotations

import json

from src.facts import Facts
from src.model import scripted_chat
from src.perceive import BIGHA_TO_HECTARE, coerce, perceive


def _run(payload) -> Facts:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    facts, _warnings, _raw = perceive("...", scripted_chat([text]))
    return facts


def test_invented_field_is_dropped():
    """The model cannot introduce a fact no rule reads."""
    facts, warnings, _ = perceive(
        "...", scripted_chat([json.dumps({"bpl": True, "caste_category": "OBC"})]))
    assert facts.bpl is True
    assert not hasattr(facts, "caste_category")
    assert any("caste_category" in w for w in warnings)


def test_null_is_treated_as_unknown_not_false():
    facts = _run({"owns_farmland": None})
    assert facts.owns_farmland is None


def test_explicit_false_survives():
    facts = _run({"owns_farmland": False})
    assert facts.owns_farmland is False


def test_bighas_converted_in_code_not_by_the_model():
    facts = _run({"farmland_bighas": 2})
    assert facts.farmland_hectares == round(2 * BIGHA_TO_HECTARE, 3)
    assert facts.owns_farmland is True


def test_income_period_is_derived_consistently():
    assert _run({"annual_income": 60_000}).monthly_income == 5_000
    assert _run({"monthly_income": 5_000}).annual_income == 60_000


def test_wrong_type_is_dropped_with_a_warning():
    facts, warnings, _ = perceive(
        "...", scripted_chat([json.dumps({"applicant_age": "बहुत"})]))
    assert facts.applicant_age is None
    assert any("applicant_age" in w for w in warnings)


def test_unparseable_output_yields_no_facts_rather_than_wrong_ones():
    facts, warnings, _ = perceive("...", scripted_chat(["I think she qualifies!"]))
    assert facts.known() == {}
    assert warnings


def test_json_inside_a_code_fence_is_read():
    facts = _run('```json\n{"bpl": true}\n```')
    assert facts.bpl is True


def test_prose_around_the_json_is_tolerated():
    facts = _run('Here is the JSON:\n{"bpl": true}\nHope that helps.')
    assert facts.bpl is True


def test_question_context_reaches_the_model():
    """A bare 'yes' is meaningless without the question, so it has to be in the prompt."""
    seen: list = []

    def spy(messages):
        seen.append(messages)
        return "{}"

    perceive("हाँ", spy, question="क्या आपके पास बीपीएल कार्ड है?", expect_fact="bpl")
    last = seen[0][-1]["content"]
    assert "बीपीएल" in last
    assert "bpl" in last
