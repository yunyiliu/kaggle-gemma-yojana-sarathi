"""Tests for the part that decides entitlement.

These run without a model. That is the point of the split: the component that can deny
someone a pension is ordinary Python with ordinary tests, and none of its behaviour
depends on what a language model felt like emitting today.
"""
from __future__ import annotations

import pytest
import yaml

from src.engine import (SchemeFileError, evaluate, load_schemes, next_question,
                        summarise)
from src.facts import Facts

SCHEMES = load_schemes("schemes/schemes.yaml")


# ---------------------------------------------------------------- the three-way split

def test_unknown_is_not_denial():
    """The failure this guards against is a tool that quietly says no.

    A widow who has stated nothing but her widowhood must come back as 'uncertain' for
    the widow pension, never 'ineligible'. Collapsing unknown into no is how entitlements
    go unclaimed.
    """
    res = {r.id: r for r in evaluate(Facts(applicant_is_widow=True), SCHEMES)}
    assert res["widow_pension"].status == "uncertain"
    assert "annual_income" in res["widow_pension"].missing


def test_definite_failure_is_ineligible():
    facts = Facts(applicant_is_widow=True, applicant_age=35, annual_income=90_000,
                  remarried=False)
    res = {r.id: r for r in evaluate(facts, SCHEMES)}
    assert res["widow_pension"].status == "ineligible"
    assert res["widow_pension"].failed[0].fact == "annual_income"


def test_all_rules_met_is_eligible():
    facts = Facts(applicant_is_widow=True, applicant_age=35, annual_income=40_000,
                  remarried=False)
    res = {r.id: r for r in evaluate(facts, SCHEMES)}
    assert res["widow_pension"].status == "eligible"


def test_false_and_unknown_differ():
    """`False` is an answer; `None` is a question still to ask."""
    unknown = evaluate(Facts(), SCHEMES)
    said_no = evaluate(Facts(owns_farmland=False), SCHEMES)
    by_id = lambda rs: {r.id: r.status for r in rs}
    assert by_id(unknown)["pm_kisan"] == "uncertain"
    assert by_id(said_no)["pm_kisan"] == "ineligible"


# ---------------------------------------------------------------- question selection

def test_question_prefers_the_fact_that_unlocks_most_money():
    """With nothing known, the BPL card gates the largest total benefit on the list."""
    fact, diag = next_question(Facts(), SCHEMES)
    assert fact == "bpl"
    assert set(diag["resolves"]["bpl"]) >= {"nfbs", "ayushman"}


def test_questions_stop_when_nothing_can_change():
    """Every scheme decided one way or the other means there is nothing left to ask."""
    facts = Facts(applicant_is_widow=False, applicant_age=30, annual_income=9_000_000,
                  bpl=False, residence="urban", house_type="pucca",
                  owns_motor_vehicle=True, has_girl_child=False,
                  has_school_age_child=False, owns_farmland=False,
                  primary_earner_died=False)
    fact, _ = next_question(facts, SCHEMES)
    assert fact is None


def test_answering_the_chosen_question_reduces_uncertainty():
    facts = Facts()
    before = len(summarise(evaluate(facts, SCHEMES))["uncertain"])
    for _ in range(6):
        fact, _ = next_question(facts, SCHEMES)
        if fact is None:
            break
        facts = facts.with_(fact, _plausible_value(fact))
    after = len(summarise(evaluate(facts, SCHEMES))["uncertain"])
    assert after < before


def _plausible_value(fact: str):
    return {"bpl": True, "applicant_age": 35, "annual_income": 40_000,
            "residence": "rural", "house_type": "kutcha"}.get(fact, False)


# ---------------------------------------------------------------- the rule file itself

def test_every_rule_references_a_declared_fact():
    """A typo in schemes.yaml must fail at load, not silently never match."""
    bad = {"schemes": [{"id": "x", "name_en": "X", "source": "s", "documents": [],
                        "rules": [{"fact": "aplicant_age", "op": "gte", "value": 60}]}]}
    with pytest.raises(SchemeFileError, match="unknown fact"):
        _validate(bad)


def test_every_fact_used_by_a_rule_is_askable():
    """If a rule depends on a fact, there has to be a way to ask a person about it."""
    from src.facts import QUESTIONS
    for s in SCHEMES["schemes"]:
        for r in s["rules"]:
            assert r["fact"] in QUESTIONS, f"{s['id']} needs a question for {r['fact']}"


def test_duplicate_scheme_ids_rejected():
    bad = {"schemes": [
        {"id": "x", "name_en": "X", "source": "s", "documents": [], "rules": []},
        {"id": "x", "name_en": "Y", "source": "s", "documents": [], "rules": []}]}
    with pytest.raises(SchemeFileError, match="duplicate"):
        _validate(bad)


def test_every_scheme_cites_a_source():
    for s in SCHEMES["schemes"]:
        assert s["source"].strip(), f"{s['id']} has no source document"


def _validate(obj, tmp=None):
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkstemp(suffix=".yaml")[1])
    p.write_text(yaml.safe_dump(obj))
    try:
        return load_schemes(p)
    finally:
        p.unlink()
