"""Deterministic eligibility engine.

This is the part of the system that decides who is entitled to what, and it contains no
model.  Being confidently wrong here has asymmetric costs in both directions: telling
someone they qualify when they do not costs them a day's wage and a bus fare to a
counter that turns them away; telling them they do not qualify when they do costs them
money they are owed, possibly for years, and they have no way to know.  Neither error is
one to hand to a language model that cannot show its working.

So the model never sees these rules.  It fills in a form (src/perceive.py) and it writes
the result up (src/explain.py).  Everything between is this file plus schemes/schemes.yaml,
where a caseworker can read any decision back to the line that produced it.

The engine returns three things, not one:

  eligible    every rule that can be checked passes
  ineligible  some rule definitively fails, with the rule that failed
  uncertain   nothing fails, but a rule cannot be evaluated for want of a fact

The third is the useful one.  A tool that silently drops "uncertain" into "no" is the
failure mode that keeps entitlements unclaimed, and the missing facts it reports are what
drives the follow-up question - see `next_question`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .facts import FACT_NAMES, QUESTIONS, Facts

Status = Literal["eligible", "ineligible", "uncertain"]


class SchemeFileError(ValueError):
    pass


@dataclass
class RuleResult:
    fact: str
    op: str
    value: Any
    actual: Any
    passed: bool | None          # None = cannot tell, the fact is unknown
    note: str | None = None

    def describe(self) -> str:
        if self.passed is None:
            return f"{self.fact} unknown (needs {self.op} {self.value})"
        verb = "ok" if self.passed else "fails"
        return f"{self.fact}={self.actual!r} {verb} ({self.op} {self.value})"


@dataclass
class SchemeResult:
    scheme: dict
    status: Status
    rules: list[RuleResult]
    missing: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.scheme["id"]

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.rules if r.passed is False]


def _check(op: str, actual: Any, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    if op == "lte":
        return float(actual) <= float(expected)
    if op == "gte":
        return float(actual) >= float(expected)
    if op == "lt":
        return float(actual) < float(expected)
    if op == "gt":
        return float(actual) > float(expected)
    if op == "between":
        lo, hi = expected
        return float(lo) <= float(actual) <= float(hi)
    raise SchemeFileError(f"unknown operator: {op}")


VALID_OPS = {"eq", "ne", "in", "not_in", "lte", "gte", "lt", "gt", "between"}


def load_schemes(path: str | Path = "schemes/schemes.yaml") -> dict:
    """Load and validate the rule file.

    Validation is strict and happens at load: a rule that references a fact the
    vocabulary does not declare is a startup error, not a scheme that silently never
    matches.  That failure mode is invisible in testing and denies people benefits in
    production, so it is made loud here.
    """
    data = yaml.safe_load(Path(path).read_text())
    if "schemes" not in data:
        raise SchemeFileError("no 'schemes' key")
    seen: set[str] = set()
    for s in data["schemes"]:
        for required in ("id", "name_en", "rules", "documents", "source"):
            if required not in s:
                raise SchemeFileError(f"scheme {s.get('id', '?')} is missing '{required}'")
        if s["id"] in seen:
            raise SchemeFileError(f"duplicate scheme id: {s['id']}")
        seen.add(s["id"])
        for r in s["rules"]:
            if r["fact"] not in FACT_NAMES:
                raise SchemeFileError(
                    f"scheme {s['id']} references unknown fact {r['fact']!r}; "
                    f"add it to src/facts.py or fix the spelling")
            if r["op"] not in VALID_OPS:
                raise SchemeFileError(f"scheme {s['id']} uses unknown op {r['op']!r}")
            if r["fact"] not in QUESTIONS:
                raise SchemeFileError(
                    f"fact {r['fact']!r} has no question in src/facts.py - every fact a "
                    f"rule depends on must be answerable by asking the person")
    return data


def evaluate(facts: Facts, schemes: dict) -> list[SchemeResult]:
    """Evaluate every scheme against what is known."""
    known = facts.known()
    out: list[SchemeResult] = []
    for s in schemes["schemes"]:
        rules: list[RuleResult] = []
        missing: list[str] = []
        for r in s["rules"]:
            name = r["fact"]
            if name not in known:
                rules.append(RuleResult(name, r["op"], r["value"], None, None,
                                        r.get("note")))
                missing.append(name)
                continue
            actual = known[name]
            try:
                passed = _check(r["op"], actual, r["value"])
            except (TypeError, ValueError):
                # a fact of the wrong shape is a perception bug; treat it as unknown
                # rather than as a failure, so it surfaces as a question not a denial
                rules.append(RuleResult(name, r["op"], r["value"], actual, None,
                                        r.get("note")))
                missing.append(name)
                continue
            rules.append(RuleResult(name, r["op"], r["value"], actual, passed,
                                    r.get("note")))
        if any(r.passed is False for r in rules):
            status: Status = "ineligible"
        elif missing:
            status = "uncertain"
        else:
            status = "eligible"
        out.append(SchemeResult(s, status, rules, missing))
    return out


def next_question(facts: Facts, schemes: dict) -> tuple[str | None, dict]:
    """Choose the single fact worth asking about next.

    This is what makes the conversation short.  A person seeking help has limited time
    and patience, and every question that cannot change any answer is a question that
    costs their goodwill for nothing.

    The score for a fact is the total benefit currently sitting in 'uncertain' that
    knowing it could resolve, weighted by how close each scheme is to a decision.  A fact
    that would settle a ₹1,20,000 housing grant outranks one that would settle a ₹1,000
    monthly pension, and a scheme with one unknown left outranks one with four.

    Returns (fact_name, diagnostics).  fact_name is None when nothing further can move
    an outcome, which is the signal to stop asking and report.
    """
    results = evaluate(facts, schemes)
    scores: dict[str, float] = {}
    detail: dict[str, list[str]] = {}
    for res in results:
        if res.status != "uncertain":
            continue
        value = _benefit_weight(res.scheme)
        # a scheme one fact away is worth more per question than one four facts away
        share = value / len(res.missing)
        for m in res.missing:
            scores[m] = scores.get(m, 0.0) + share
            detail.setdefault(m, []).append(res.id)
    if not scores:
        return None, {"scores": {}, "resolves": {}}
    best = max(scores, key=lambda k: scores[k])
    return best, {"scores": scores, "resolves": detail}


def _benefit_weight(scheme: dict) -> float:
    """Rough annual rupee value of a scheme, for ranking questions only.

    Parsed from the human-readable benefit string so the YAML stays readable by the
    people who have to audit it.  A monthly figure is annualised; anything unparseable
    falls back to a small positive weight so the scheme still generates questions.
    """
    text = str(scheme.get("benefit_en", "")).replace(",", "")
    digits = "".join(c if c.isdigit() else " " for c in text).split()
    if not digits:
        return 1000.0
    amount = float(max(digits, key=len))
    if "month" in text.lower():
        amount *= 12
    return amount if math.isfinite(amount) and amount > 0 else 1000.0


def summarise(results: list[SchemeResult]) -> dict[str, list[SchemeResult]]:
    return {
        "eligible": [r for r in results if r.status == "eligible"],
        "uncertain": [r for r in results if r.status == "uncertain"],
        "ineligible": [r for r in results if r.status == "ineligible"],
    }


def audit_trail(res: SchemeResult) -> str:
    """The decision, written so a caseworker can check it against the source document."""
    lines = [f"{res.scheme['name_en']} [{res.id}] -> {res.status.upper()}",
             f"  source: {res.scheme['source']}"]
    for r in res.rules:
        mark = {True: "PASS", False: "FAIL", None: "????"}[r.passed]
        lines.append(f"  [{mark}] {r.describe()}")
        if r.note:
            lines.append(f"         note: {r.note}")
    return "\n".join(lines)
