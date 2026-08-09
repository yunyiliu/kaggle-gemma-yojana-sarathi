"""Measure the perception step against hand-labelled utterances.

Reports two numbers that matter in opposite directions:

  recall     of the facts a person actually stated, how many were captured
  invention  of the facts they did not state, how many were filled in anyway

Invention is the one to watch. A missed fact costs a follow-up question; an invented one
silently decides an entitlement on something nobody said. The engine is built so silence
becomes a question rather than a denial, and that only holds if silence reaches it.
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")
from src.model import ollama_chat, available          # noqa: E402
from src.perceive import perceive                      # noqa: E402
from tests.perception_cases import CASES, DERIVED_SUFFIX  # noqa: E402


def close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(1.0, 0.02 * abs(float(b)))
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat each case, to see run-to-run stability")
    args = ap.parse_args()

    if not available():
        raise SystemExit("Ollama is not reachable; start it with `ollama serve`")

    hit = miss = invented = 0
    rows = []
    t0 = time.time()
    for case in CASES:
        got_all = []
        for _ in range(args.runs):
            facts, warnings, _raw = perceive(
                case["utterance"], ollama_chat,
                question=case.get("question"), expect_fact=case.get("expect_fact"))
            got_all.append(facts.known())
        got = got_all[0]

        wrong, missing, extra = [], [], []
        for k, v in case["expect"].items():
            if k in got and close(got[k], v):
                hit += 1
            elif k in got:
                wrong.append(f"{k}={got[k]!r} (wanted {v!r})")
                miss += 1
            else:
                missing.append(k)
                miss += 1
        for k in case["forbid"]:
            if k.endswith(DERIVED_SUFFIX):
                continue
            if k in got:
                extra.append(f"{k}={got[k]!r}")
                invented += 1

        stable = all(g == got_all[0] for g in got_all) if args.runs > 1 else None
        rows.append((case["id"], missing, wrong, extra, stable))

    total_expected = sum(len(c["expect"]) for c in CASES)
    total_forbidden = sum(len([k for k in c["forbid"]
                               if not k.endswith(DERIVED_SUFFIX)]) for c in CASES)
    print(f"\n{len(CASES)} cases, {args.runs} run(s), {time.time()-t0:.0f}s\n")
    for cid, missing, wrong, extra, stable in rows:
        flag = "ok  " if not (missing or wrong or extra) else "FAIL"
        note = []
        if missing:
            note.append("missed " + ",".join(missing))
        if wrong:
            note.append("wrong " + "; ".join(wrong))
        if extra:
            note.append("INVENTED " + "; ".join(extra))
        if stable is False:
            note.append("unstable across runs")
        print(f"  [{flag}] {cid:32s} {'  |  '.join(note)}")

    print(f"\nrecall    {hit}/{total_expected} = {hit/max(total_expected,1):.1%}"
          f"   (a miss costs one extra question)")
    print(f"invention {invented}/{total_forbidden} = {invented/max(total_forbidden,1):.1%}"
          f"   (an invention decides an entitlement nobody stated)")


if __name__ == "__main__":
    main()
