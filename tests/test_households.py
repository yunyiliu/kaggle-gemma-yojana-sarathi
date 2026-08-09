"""End-to-end tests of the part that decides entitlement.

The other test files check components. These check the answer: eight complete households
in, the hand-derived list of schemes out. No model is involved, which is the point - the
component that can deny someone a pension is ordinary Python and is tested like it.
"""
from __future__ import annotations

import random

import pytest

from src.engine import evaluate, load_schemes, next_question, summarise
from src.facts import Facts
from tests.households import HOUSEHOLDS

import scripts_ask


@pytest.fixture(scope="module")
def schemes():
    return load_schemes("schemes/schemes.yaml")


@pytest.mark.parametrize("h", HOUSEHOLDS, ids=lambda h: h["id"])
def test_full_facts_reach_the_hand_derived_answer(h, schemes):
    got = summarise(evaluate(Facts(**h["facts"]), schemes))
    assert sorted(r.id for r in got["eligible"]) == sorted(h["expect_eligible"])
    assert not got["uncertain"], "a complete household should leave nothing uncertain"


@pytest.mark.parametrize("h", HOUSEHOLDS, ids=lambda h: h["id"])
def test_every_ruled_out_scheme_names_the_rule_that_ruled_it_out(h, schemes):
    """An exclusion nobody can trace is an exclusion nobody can appeal."""
    for res in evaluate(Facts(**h["facts"]), schemes):
        if res.status == "ineligible":
            assert res.failed, f"{res.id} is ineligible but names no failing rule"


@pytest.mark.parametrize("name", sorted(scripts_ask.STRATEGIES))
def test_question_order_never_changes_the_conclusion(name, schemes):
    """Asking in a different order may cost more questions. It must not change the answer.

    This is the property that makes it safe to optimise the interview at all: the order
    is a matter of the person's time, never of what they are entitled to.
    """
    pick = scripts_ask.STRATEGIES[name]
    for h in HOUSEHOLDS:
        r = scripts_ask.run(h, schemes, pick, random.Random(0))
        assert r["reached"] == sorted(h["expect_eligible"]), \
            f"{name} on {h['id']} reached {r['reached']}"


def test_interview_terminates_and_stops_asking_dead_questions(schemes):
    """next_question returns None once nothing further can move an outcome."""
    for h in HOUSEHOLDS:
        facts = Facts()
        for _ in range(len(h["facts"]) + 1):
            fact, _diag = next_question(facts, schemes)
            if fact is None:
                break
            facts = facts.with_(fact, h["facts"].get(fact))
        else:
            pytest.fail(f"{h['id']}: interview did not terminate")
        assert next_question(facts, schemes)[0] is None


def test_a_household_entitled_to_nothing_is_told_so(schemes):
    """The tool has to be able to say no, and say it without leaving anything uncertain."""
    h = next(x for x in HOUSEHOLDS if x["id"] == "urban_salaried")
    got = summarise(evaluate(Facts(**h["facts"]), schemes))
    assert got["eligible"] == []
    assert not got["uncertain"]
    assert len(got["ineligible"]) == len(schemes["schemes"])
