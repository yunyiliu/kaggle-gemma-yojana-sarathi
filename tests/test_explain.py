"""Tests for the writing step.

The model writes the message but does not choose its content. These check that the
content it is handed is complete and that a message inventing a scheme is caught.
"""
from __future__ import annotations

from src.engine import evaluate, load_schemes, summarise
from src.explain import (build_prompt, document_checklist, render_brief,
                         verify_no_new_schemes)
from src.facts import Facts
from src.model import scripted_chat
from src.explain import explain

SCHEMES = load_schemes("schemes/schemes.yaml")


def _widow():
    facts = Facts(applicant_is_widow=True, applicant_age=35, annual_income=40_000,
                  remarried=False, bpl=True)
    return summarise(evaluate(facts, SCHEMES))


def test_brief_carries_amounts_documents_and_office():
    s = _widow()
    brief = render_brief(s["eligible"], s["uncertain"])
    assert "Widow Pension" in brief or "निराश्रित" in brief
    assert "1,000" in brief
    assert "Death certificate of the husband" in brief
    assert "sspy-up.gov.in" in brief


def test_prompt_forbids_adding_schemes():
    s = _widow()
    system = build_prompt(s["eligible"], s["uncertain"])[0]["content"]
    assert "not in the list" in system.lower()


def test_invented_scheme_in_the_message_is_flagged():
    s = _widow()
    bad = "आपको निराश्रित महिला पेंशन और पीएम किसान सम्मान निधि दोनों मिलेंगे।"
    problems = verify_no_new_schemes(bad, s["eligible"], SCHEMES)
    assert any("PM Kisan" in p for p in problems)


def test_a_faithful_message_is_not_flagged():
    s = _widow()
    good = "आप निराश्रित महिला पेंशन के लिए आवेदन कर सकती हैं।"
    assert verify_no_new_schemes(good, s["eligible"] + s["uncertain"], SCHEMES) == []


def test_documents_are_grouped_so_each_is_collected_once():
    s = _widow()
    checklist = document_checklist(s["eligible"])
    names = [d for d, _ in checklist]
    assert len(names) == len(set(names))
    # the document needed by the most schemes comes first, so one trip covers the most
    counts = [len(schemes) for _, schemes in checklist]
    assert counts == sorted(counts, reverse=True)


def test_explain_returns_the_models_words_and_any_problems():
    s = _widow()
    text, problems = explain(s["eligible"], s["uncertain"], SCHEMES,
                             scripted_chat(["आप पेंशन के लिए आवेदन कर सकती हैं।"]))
    assert "पेंशन" in text
    assert problems == []
