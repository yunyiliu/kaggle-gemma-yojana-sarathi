"""Measure the interview: does asking in benefit order actually help, and by how much.

`next_question()` claims that ordering questions by the benefit they would resolve makes
the conversation shorter and front-loads the money. That is a claim about an algorithm,
so it can be checked without a model and without a person - which is what this does.

Each household in tests/households.py answers truthfully whatever it is asked. Four
strategies pick the questions:

  benefit   what the engine ships: score each unknown fact by the benefit sitting in
            'uncertain' that knowing it would resolve, divided across the facts each
            scheme is still waiting on
  coverage  ask whatever unblocks the most schemes, ignoring how much they are worth.
            This is the ablation that matters: it isolates the benefit weighting from
            the mere fact of asking something relevant
  fixed     the vocabulary order, skipping facts no undecided scheme needs. This is what
            a paper form does
  random    a shuffle, averaged over seeds. The floor

Two numbers come out, and the second is the one that matters at a doorstep.

**Questions to a final answer.** How long the interview runs before no remaining question
can change any outcome.

**Benefit secured after k questions.** Interviews get abandoned. A visitor is called away,
a child starts crying, the queue at the block office moves. If the conversation stops at
question three, how much of the household's true entitlement has already been established?
A strategy that reaches the same place in the same number of questions but establishes the
₹5,00,000 health cover first is not equal - it is better, and only this second measurement
can see the difference.
"""
from __future__ import annotations

import argparse
import random
import sys

sys.path.insert(0, ".")
from src.engine import (evaluate, load_schemes, next_question,  # noqa: E402
                        summarise, _benefit_weight)
from src.facts import FACT_NAMES, Facts                          # noqa: E402
from tests.households import HOUSEHOLDS                          # noqa: E402


def _needed_facts(facts: Facts, schemes: dict) -> list[str]:
    """Facts some still-undecided scheme is waiting on. Anything else is a wasted breath."""
    need: list[str] = []
    for res in evaluate(facts, schemes):
        if res.status == "uncertain":
            for m in res.missing:
                if m not in need:
                    need.append(m)
    return need


def pick_benefit(facts: Facts, schemes: dict, rng) -> str | None:
    return next_question(facts, schemes)[0]


def pick_coverage(facts: Facts, schemes: dict, rng) -> str | None:
    """Unblock the most schemes, treating a ₹1,000 pension as equal to a ₹5 lakh cover."""
    counts: dict[str, int] = {}
    for res in evaluate(facts, schemes):
        if res.status == "uncertain":
            for m in res.missing:
                counts[m] = counts.get(m, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else None


def pick_fixed(facts: Facts, schemes: dict, rng) -> str | None:
    need = _needed_facts(facts, schemes)
    for name in sorted(FACT_NAMES):
        if name in need:
            return name
    return None


def pick_random(facts: Facts, schemes: dict, rng) -> str | None:
    need = _needed_facts(facts, schemes)
    return rng.choice(need) if need else None


STRATEGIES = {
    "benefit": pick_benefit,
    "coverage": pick_coverage,
    "fixed": pick_fixed,
    "random": pick_random,
}


def secured(facts: Facts, schemes: dict, truth_ids: set[str]) -> float:
    """Rupees of the household's true entitlement already established as eligible.

    Only schemes that are genuinely theirs count. If a strategy were ever to reach
    'eligible' on a scheme that is not in the ground truth, it earns nothing for it -
    that would be a wrong answer, not progress.
    """
    total = 0.0
    for res in evaluate(facts, schemes):
        if res.status == "eligible" and res.id in truth_ids:
            total += _benefit_weight(res.scheme)
    return total


def run(household: dict, schemes: dict, pick, rng) -> dict:
    """One interview. Returns questions asked and the benefit curve."""
    truth = set(household["expect_eligible"])
    facts = Facts()
    answers = household["facts"]
    curve = [secured(facts, schemes, truth)]
    asked: list[str] = []
    while True:
        fact = pick(facts, schemes, rng)
        if fact is None or fact in asked:
            break
        asked.append(fact)
        facts = facts.with_(fact, answers.get(fact))
        curve.append(secured(facts, schemes, truth))
    final = summarise(evaluate(facts, schemes))
    return {
        "asked": asked,
        "curve": curve,
        "reached": sorted(r.id for r in final["eligible"]),
        "correct": sorted(r.id for r in final["eligible"]) == sorted(truth),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="schemes/schemes.yaml")
    ap.add_argument("--seeds", type=int, default=20, help="shuffles averaged for `random`")
    ap.add_argument("--horizon", type=int, default=8, help="questions in the curve table")
    args = ap.parse_args()

    schemes = load_schemes(args.schemes)
    truth_total = {}
    for h in HOUSEHOLDS:
        full = Facts(**h["facts"])
        truth_total[h["id"]] = secured(full, schemes, set(h["expect_eligible"]))

    print(f"\n{len(HOUSEHOLDS)} households, {len(schemes['schemes'])} schemes, "
          f"no model involved\n")

    summary = {}
    for name, pick in STRATEGIES.items():
        seeds = range(args.seeds) if name == "random" else [0]
        n_questions, curves, wrong = [], [], []
        for seed in seeds:
            rng = random.Random(seed)
            for h in HOUSEHOLDS:
                r = run(h, schemes, pick, rng)
                n_questions.append(len(r["asked"]))
                total = truth_total[h["id"]] or 1.0
                # pad the curve to the horizon: once the interview ends, the fraction
                # secured stops changing
                c = r["curve"] + [r["curve"][-1]] * (args.horizon + 1 - len(r["curve"]))
                curves.append([v / total for v in c[:args.horizon + 1]])
                if not r["correct"]:
                    wrong.append((h["id"], r["reached"]))
        mean_curve = [sum(c[k] for c in curves) / len(curves)
                      for k in range(args.horizon + 1)]
        summary[name] = {
            "questions": sum(n_questions) / len(n_questions),
            "curve": mean_curve,
            "wrong": wrong,
        }

    print("Questions asked before no remaining question can change any outcome")
    print("  (mean over households; every strategy reaches the same final answer)\n")
    for name, s in summary.items():
        bar = "#" * round(s["questions"] * 2)
        print(f"  {name:9s} {s['questions']:5.1f}  {bar}")

    print(f"\n\nFraction of the household's true entitlement already secured after k "
          f"questions")
    print("  (mean over households, in rupees of annualised benefit)\n")
    head = "  k          " + "".join(f"{k:>7d}" for k in range(1, args.horizon + 1))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, s in summary.items():
        row = "".join(f"{v * 100:6.0f}%" for v in s["curve"][1:args.horizon + 1])
        print(f"  {name:9s}  {row}")

    bad = [(n, s["wrong"]) for n, s in summary.items() if s["wrong"]]
    print()
    if bad:
        print("!! a strategy ended on the wrong answer:")
        for n, w in bad:
            print(f"   {n}: {w[:3]}")
    else:
        print("Every strategy ends on the hand-derived answer for every household.")
        print("The ordering changes what is known when, never what is concluded.")


if __name__ == "__main__":
    main()
