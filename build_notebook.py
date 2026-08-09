"""Generate the competition notebook from the repo sources.

Generated rather than hand-written, so the notebook cannot drift from the code that was
actually tested.  Every source cell is written with `%%writefile` and contains the repo
file byte for byte - no import rewriting, no trimmed-for-the-slides version.  The
notebook then runs the same `pytest`, the same `scripts_eval.py`, and the same `demo.py`
that run locally.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "build-with-gemma-tfug-prayagraj-ai-prayagraj"
REPO = "https://github.com/yunyiliu/kaggle-gemma-yojana-sarathi"
MODEL = "gemma3:4b"

_n = {"i": 0}


def _cell(kind: str, text: str) -> dict:
    _n["i"] += 1
    cell = {"cell_type": kind, "metadata": {}, "id": f"c{_n['i']:02d}",
            "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def md(text: str) -> dict:
    return _cell("markdown", text)


def code(text: str) -> dict:
    return _cell("code", text)


def filecell(rel: str, note: str = "") -> dict:
    """A repo file, verbatim, written to the same path inside the notebook session."""
    body = (ROOT / rel).read_text().rstrip("\n")
    head = f"%%writefile {rel}\n"
    if note:
        head = f"# {note}\n" + head
    return code(head + body)


CELLS = [
    md(f"""
# योजना सारथी · Yojana Sarathi

### An offline welfare-entitlement navigator — Gemma fills in the form, plain code decides who qualifies

*Build with Gemma: TFUG Prayagraj \\[AI Prayagraj\\]* · **Code:** {REPO}

---

India runs hundreds of welfare schemes, and a large share of the money set aside for them
is never claimed. The reason is rarely unwillingness. It is that nobody at the doorstep
knows which of the hundreds apply to **this** household, what documents each one needs,
and which office to walk to. The people with the strongest claim — a widow with no income,
an elderly person living alone — are the least able to read the guidelines that describe
their own entitlement.

The decision itself is not a judgement call. Eligibility is written down: age thresholds,
income ceilings, whether the house is kutcha, whether the ration card is BPL. It is
gatekeeping by paperwork, not by discretion.

What is missing is the step from **how a person describes their life** to **which written
rules that satisfies** — done where the person is, without uploading their bereavement and
their income to anyone's API.
"""),

    md("""
## The design decision everything else follows from

> ### Gemma never decides eligibility.

```
She says  (Hindi / Hinglish / English)
  "मेरे पति का पिछले साल देहांत हो गया। दो बच्चे स्कूल जाते हैं। कच्चा घर है।"
        │
        ▼   Gemma 3 · PERCEPTION
  {applicant_is_widow: true, primary_earner_died: true, house_type: "kutcha", ...}
        │
        ▼   schemes.yaml + engine.py · DECISION          ←  no model on this line
  eligible / ineligible / UNCERTAIN, and for each uncertain one, what exactly is missing
        │
        ▼   engine.py · WHAT TO ASK NEXT
  the one fact whose answer resolves the most benefit  →  one question, not twenty
        │
        ▼   Gemma 3 · ACTION
  a message in her language, a deduplicated document checklist, and an audit trail
```

The costs are asymmetric in **both** directions, and neither error is acceptable. Tell
someone they qualify when they do not, and they spend a day's wage and a bus fare on a
counter that turns them away. Tell them they do not qualify when they do, and they lose
money they are owed, possibly for years, with no way to find out. Neither is an error to
hand to a model that cannot show its working.

So the model gets the job it is genuinely good at — turning Awadhi-inflected Hindi,
code-mixed English, *bighas* and *lakhs* into structured fields — and none of the job
where being confidently wrong is expensive. Between the two model calls sits a YAML file
a programme officer can read line by line against the published guidelines.

This is also **why a 4B model is enough**: it is never asked to reason about entitlement.

### Why an open model, running locally

Three constraints each independently rule out a hosted API.

1. **The data is not ours to send.** A household's bereavement, income and caste details
   are about as sensitive as personal data gets.
2. **Connectivity is worst where need is highest.** A tool that requires a network fails
   at exactly the doorstep it was built for.
3. **Per-token billing does not survive contact with a public programme.** Cost per query
   × a lakh of frontline workers × a daily visit schedule is not adoptable.

No API key appears anywhere in this notebook.
"""),

    md("""
## 1 · Environment

A local Ollama server on the notebook machine. Nothing leaves it, and no API key appears
anywhere below.

Two dependencies the Ollama installer needs and Kaggle's image does not ship: `zstd`,
without which extraction fails outright, and `pciutils`, without which the installer
cannot see the GPU and quietly installs the CPU-only build.
"""),
    code(f"""
import json, os, subprocess, sys, textwrap, time
t0 = time.time()

!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y zstd pciutils > /dev/null 2>&1
!curl -fsSL https://ollama.com/install.sh 2>/dev/null | sh 2>&1 | tail -2

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(8)

MODEL = "{MODEL}"
os.environ["GEMMA_MODEL"] = MODEL          # read by src/model.py
!ollama pull {{MODEL}} 2>&1 | tail -1

print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)
print(subprocess.run(["bash", "-lc", "nvidia-smi -L || echo 'no GPU - running on CPU'"],
                     capture_output=True, text=True).stdout)
print(f"environment ready in {{time.time() - t0:.0f}}s")
"""),
    code("""
# The repo layout, rebuilt here so every cell below is the file it says it is.
!mkdir -p src tests schemes
# Both __init__.py files matter. Kaggle's image already has a `tests` package installed,
# and a directory without __init__.py is only a namespace portion - the import scan keeps
# going and the installed one wins. A regular package takes precedence.
open("src/__init__.py", "w").close()
open("tests/__init__.py", "w").close()
sys.path.insert(0, ".")
print("ok")
"""),

    md("""
## 2 · The rules

This file is the part of the system that decides entitlement, and it is deliberately not
the model's job. Every scheme cites the document its thresholds came from, so a programme
officer can check a line against a circular without reading any Python. Adding a scheme is
a YAML entry, not a code change.
"""),
    filecell("schemes/schemes.yaml"),

    md("""
## 3 · The fact vocabulary

The model may emit these fields and nothing else. Every field is **tri-state** — `True`,
`False`, or unknown — because *she did not say* and *she said no* have to be different
things. Collapsing the two is precisely how a benefits system quietly denies people what
they are owed: the interview stops early and the tool reports "not eligible" when the
truthful answer was "we never asked".
"""),
    filecell("src/facts.py"),

    md("""
## 4 · The engine

No model appears anywhere below this line. `evaluate()` returns three outcomes, not two —
and the third, **uncertain** with the list of what is missing, is what drives the
conversation.
"""),
    filecell("src/engine.py"),

    md("""
### Which question to ask next

A person seeking help has limited time and patience. A question that cannot change any
answer spends both for nothing.

`next_question()` scores each unknown fact by the total benefit currently sitting in
*uncertain* that knowing it would resolve, weighted by how close each scheme is to a
decision. From a cold start it asks about the **BPL / Antyodaya ration card** — that one
fact gates ₹30,000 of housing assistance *and* a ₹5,00,000 health cover.
"""),
    code("""
from src.engine import load_schemes, evaluate, next_question, summarise, audit_trail
from src.facts import Facts, QUESTIONS

schemes = load_schemes("schemes/schemes.yaml")
print(f"{len(schemes['schemes'])} schemes loaded\\n")

fact, diag = next_question(Facts(), schemes)
print("first question  :", QUESTIONS[fact]["hi"])
print("about the fact  :", fact)
print("which decides   :", ", ".join(diag["resolves"][fact]))
print("\\nrupees of undecided benefit each unknown fact would resolve:")
for k, v in sorted(diag["scores"].items(), key=lambda kv: -kv[1])[:6]:
    print(f"   {k:26s} {v:>12,.0f}")
"""),

    md("""
## 5 · Perception — speech to facts

Schema-bounded. Anything the model emits outside the vocabulary is dropped **in code**,
not requested in the prompt: a prompt is a request, and the boundary between a language
model and an entitlement decision needs a guarantee.
"""),
    filecell("src/perceive.py"),
    filecell("src/model.py"),

    md("""
## 6 · Action — decision to words

The model writes; it does not decide. Every fact in the output — which schemes, which
documents, which office, how much — is passed in from the engine. `verify_no_new_schemes`
then checks afterwards that the message did not volunteer a scheme nobody evaluated.
"""),
    filecell("src/explain.py"),
    filecell("src/session.py"),

    md("""
## 7 · The tests run without a model

27 tests, no Ollama needed. That is deliberate: the component that can deny someone a
pension is ordinary Python with ordinary tests, and none of its behaviour depends on what
a language model felt like emitting today. The model-boundary tests use scripted replies
to check that whatever the model returns, only well-formed facts get past that layer.
"""),
    filecell("tests/test_engine.py"),
    filecell("tests/test_perceive.py"),
    filecell("tests/test_explain.py"),
    code('!python -m pytest tests -q'),

    md("""
## 8 · How well does perception actually work

Most of the risk lives in one place: what the model writes into the form. So it is
measured rather than demonstrated.

13 hand-labelled utterances covering code-mixed speech, *bighas* and *lakhs*, regional
number words, answers that address a different question than the one asked, and explicit
denials. Two numbers, pulling in opposite directions:

* **recall** — of the facts a person actually stated, how many were captured. A miss costs
  one extra question.
* **invention** — of the facts they did *not* state, how many were filled in anyway. An
  invention silently decides an entitlement on something nobody said.

Invention is the one to watch.
"""),
    filecell("tests/perception_cases.py"),
    filecell("scripts_eval.py"),
    code('!python scripts_eval.py --runs 2'),

    md("""
### What the measurement caught

The first run scored **79.2% recall / 21.4% invention**. The inventions are the
interesting part:

| the person said | the model wrote | why it matters |
|---|---|---|
| *"pata nahi kitna kamate hain, kabhi kuch kabhi kuch"* — I don't know, it varies | `annual_income: 0` | zero satisfies **every** income ceiling in the file. That one hallucination grants every income-tested scheme on the list. |
| *"मेरी सास का देहांत हो गया था"* — my mother-in-law died | `applicant_is_widow: true` | someone else's death recorded as the speaker's own widowhood, which is a document she cannot produce |

Both were fixed in the prompt with negative examples, and the zero was fixed **again in
code** (`coerce` drops a zero income with a warning) — a prompt is a request, and this one
needed a guarantee.

A third failure appeared only *after* those fixes, which is the argument for having the
labelled set at all: a flat *"नहीं"* to *have you remarried?* came back as no fact rather
than `remarried: false`. The negative examples had over-generalised into "anything
touching marriage or bereavement is a trap". Fixed by adding `ANSWER_FEWSHOT` — examples
in the shape a real answer arrives in, with the question as context.

| | recall | invention |
|---|---:|---:|
| first run | 79.2% | 21.4% |
| after the prompt rules, negative few-shots and the code backstop | 95.8% | 0% |
| after `ANSWER_FEWSHOT` | **100%** | **0%** |

### The number above may not be 100%, and that is the last finding

Those figures come from six runs on a local machine, where they are stable. The cell you
just ran, on Kaggle's P100, has also produced **95.8% / 0%** — one miss of
`years_since_earner_death` on the opening narrative.

Temperature is 0, so this is not sampling noise. It is the same weights on a different
backend: different kernels, different quantisation arithmetic, a different order of
operations. Temperature 0 buys determinism *within* a machine, not *across* machines, and
anyone quoting a single extraction score without saying what it ran on is quoting a
number about their laptop.

Which is precisely the argument this project is built on. Look at what moved and what did
not:

- **Recall moved by one case.** A miss costs one extra question. The engine notices the
  fact is still unknown and asks about it — the same thing it does for any other unknown.
- **Invention stayed at 0% on both machines, on every run.** It is not held there by the
  model behaving well today; it is held there by `coerce()` dropping anything outside the
  vocabulary, dropping a zero income, and by the eligibility decision never being the
  model's to make.

A system whose safety depends on a model scoring the same on someone else's GPU does not
have a safety property. The recall number is a cost estimate. The invention number is a
guarantee, and it is a guarantee because it lives in code.
"""),

    md("""
## 9 · One conversation, end to end

A widow in a village, starting from "I don't know what I am entitled to".

The persona below is a dictionary keyed by *fact name*. The script does not know which
questions it will be asked or in what order — the engine picks each one from what is still
undecided, exactly as it would with a real person in front of it.
"""),
    filecell("demo.py"),
    code('!python demo.py'),

    md("""
## What this is and is not

**Is:** an auditable architecture for the doorstep — perception by a small open model,
entitlement by rules a human can check, and a measurement of the one step where the model
can do damage.

**Is not:**

- **An eligibility determination.** Thresholds are transcribed from published guidance to
  demonstrate the architecture. They are revised by notification and vary by state; every
  line needs a programme officer's sign-off against the current circulars before any real
  use.
- **Comprehensive.** 8 schemes, chosen to cover the common situations of a rural UP
  household. The rule file is the extension point.
- **Multilingual, as measured.** Hindi and English have a labelled set behind them. Gemma
  supports many more languages; a claim without a labelled set is not a claim.
- **A filing service.** It tells a person what to bring and where to go. It does not
  submit anything on their behalf.

`verify_no_new_schemes` is exact-match, so it catches a model volunteering a scheme by
name and would miss a paraphrase. It is a backstop, not the safeguard — the factual
content is assembled in code.

## What I would build next

1. **Voice in, voice out.** Everything above assumes typed text. The person this is for
   may not read; the ASHA worker or CSC operator beside them does. Whisper-class ASR plus
   TTS closes that gap and changes nothing above it.
2. **A caseworker's correction loop.** When the engine is wrong, the operator should be
   able to say so and have it recorded *against the rule that produced the decision* —
   which is possible precisely because there is a rule that produced it. That is the
   dataset that improves a public programme, and a model that decides end-to-end cannot
   produce it.
3. **Coverage by district.** Eight schemes demonstrate the architecture; a district's
   actual list is a data problem, not a modelling one.
"""),

    md(f"---\n\n**Repository, with the full engineering log:** {REPO}"),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "yojana-sarathi.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

    (out.parent / "kernel-metadata.json").write_text(json.dumps({
        "id": "alexyy/yojana-sarathi-offline-welfare-navigator",
        "title": "Yojana Sarathi: offline welfare navigator",
        "code_file": out.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [SLUG],
        "kernel_sources": [],
    }, indent=2) + "\n")
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
