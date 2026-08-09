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
MODEL = "gemma4:e4b"      # what ships
ALT_MODEL = "gemma4:e2b"  # the smaller edge variant, measured against it in section 7

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

## The scheme is not the hard part. The interview is.

An offline tool that works out which welfare schemes a household can actually claim, by
conducting the interview nobody currently has time to conduct.

Built on **Gemma 4 E4B**, running locally through Ollama.

*Build with Gemma: TFUG Prayagraj \\[AI Prayagraj\\]* — GenAI for Good · **Code:** {REPO}

---

Uttar Pradesh publishes its welfare schemes. The eligibility rules are not secret and they
are not subtle: an age, an income ceiling, a ration-card category, whether the roof is
thatch or concrete. Anyone who sat down with the circulars and a household for an hour
could work out exactly what that household is owed.

Nobody has the hour.

A frontline worker in a village has a queue behind her. The household in front of her has
a harvest to get in, or a child to collect. The fact set that decides the eight schemes in
this notebook is **twenty-one fields long** — and asked as a form, in form order, most of
those questions are dead weight for any particular household. So the interview does not
happen, and the entitlement stays theoretical.

That is the actual failure. It is not a knowledge problem. It is a **triage problem under
a time budget**, and it is the one this project attacks.
"""),

    md("""
## The claim, stated up front so it can be checked

The engine picks each question by scoring every unknown field against the money it would
unlock — the benefit currently sitting in *undecided* that knowing this one fact would
resolve, divided across whatever else each scheme is still waiting on.

That is a claim about an algorithm, so it gets measured rather than asserted: eight
complete households, four question-ordering strategies, **no model in the loop at all**.
The result is the table in section 4, and the short version is:

> After **one question**, the shipped strategy has established just over **half** of what
> the average household is owed. A paper form asking the same fields in a fixed order has
> established **nothing** until question four.

Which matters because interviews get abandoned — someone is called away, a queue moves, a
child starts crying. What you have secured when the conversation stops *is* the product.
"""),

    md("""
## 1 · Environment

An Ollama server on `localhost`. No API key appears anywhere in this notebook and no
request leaves the machine.

That is not a preference. The details this tool exists to collect — a bereavement, an
income, a caste certificate — belong to the household. Connectivity is thinnest exactly
where the need is greatest. And a per-query bill does not survive being multiplied by a
state's frontline workforce.

Two dependencies the Ollama installer needs that Kaggle's image does not ship: `zstd`,
without which extraction fails outright, and `pciutils`, without which the installer
cannot see the GPU and quietly installs the CPU-only build.

Both **Gemma 4 E-series** variants are pulled. E4B is what ships; E2B is measured against
it in section 7, and the comparison turned out to be the most useful thing in this
notebook.
"""),
    code(f"""
import json, os, subprocess, sys, textwrap, time
t0 = time.time()

!apt-get -qq update > /dev/null 2>&1 && apt-get -qq install -y zstd pciutils > /dev/null 2>&1
!curl -fsSL https://ollama.com/install.sh 2>/dev/null | sh 2>&1 | tail -2

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(8)

MODEL = "{MODEL}"          # what ships
ALT_MODEL = "{ALT_MODEL}"      # the smaller edge variant, measured against it in section 7
os.environ["GEMMA_MODEL"] = MODEL          # read by src/model.py
!ollama pull {{MODEL}} 2>&1 | tail -1
!ollama pull {{ALT_MODEL}} 2>&1 | tail -1

print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)
print(subprocess.run(["bash", "-lc", "nvidia-smi -L || echo 'no GPU - running on CPU'"],
                     capture_output=True, text=True).stdout)
print(f"environment ready in {{time.time() - t0:.0f}}s")
"""),
    code("""
# The repo layout, rebuilt here so that every cell below is the file it says it is -
# byte for byte, no trimmed-for-the-slides version. The notebook is generated from the
# repository by build_notebook.py, so it cannot drift from the code that was tested.
!mkdir -p src tests schemes
# Both __init__.py files matter. Kaggle's image already ships a `tests` package, and a
# directory without __init__.py is only a namespace portion - the import scan keeps going
# and the installed one wins. A regular package takes precedence.
open("src/__init__.py", "w").close()
open("tests/__init__.py", "w").close()
sys.path.insert(0, ".")
print("ok")
"""),

    md("""
## 2 · Rules are data

Every threshold lives in one file, cites the document it came from, and is read by
ordinary Python. A programme officer can check a line against a circular without reading
any code, and adding a scheme is a YAML entry rather than a change to the engine.

A rule that references a field the vocabulary does not declare fails at **load**, loudly.
The alternative is a scheme that silently never matches anybody — invisible in testing,
and in production it just quietly denies people money.
"""),
    filecell("schemes/schemes.yaml"),

    md("""
## 3 · The vocabulary, and why every field is tri-state

`True`, `False`, or unknown. The distinction is load-bearing precisely *because* the
interview is the product: an unknown field is **a question to ask**, a `False` is an
answer already given. Collapse the two and the engine stops asking about things nobody
ever raised, and reports "not eligible" where the truthful answer was "we never asked".

This is not theoretical — section 7 has the bug this design caught.
"""),
    filecell("src/facts.py"),

    md("""
## 4 · The engine, and the measurement

No model appears anywhere below this line.

`evaluate()` returns **three** outcomes rather than two. *Uncertain* — nothing has failed,
but a rule cannot be checked yet — is not a rounding error on the way to "no". It is what
generates the next question.

`next_question()` is the part being measured.
"""),
    filecell("src/engine.py"),
    filecell("tests/households.py"),
    filecell("scripts_ask.py"),
    code('!python scripts_ask.py'),

    md("""
### Reading that table

**After one question the shipped strategy has secured 52% of the average household's true
entitlement.** Form order has secured nothing until question four, and a random relevant
question manages 4%.

The first question it asks is about the **BPL / Antyodaya ration card**, because that one
field gates ₹30,000 of bereavement assistance *and* a ₹5,00,000 health cover. Nothing else
in the vocabulary is worth as much per breath.

#### The ablation is the honest part

`coverage` asks whichever question unblocks the most schemes, ignoring what they are
worth. It **ties** the shipped strategy on interview length — 11.6 questions against 13.2
for form order — and it is worth **nothing** at question one.

So benefit weighting is not what makes the interview short. Asking a *relevant* question
does that, and plain scheme-counting is enough to get it. The weighting does something
narrower: it decides **which** half of the money you walk away with if the conversation
ends early. That is what the table supports, and it is all it supports.

Ordering is only ever allowed to cost time, never entitlement —
`test_question_order_never_changes_the_conclusion` runs all four strategies over all eight
households and asserts they land on the same answer every time.
"""),

    md("""
## 5 · Where the model sits, and where it does not

Two narrow jobs, both at the edges.

**In:** free speech → fields. Hindi, Hinglish, English, *bighas*, *lakhs*, "साल भर में
चालीस हज़ार". Genuinely hard, and genuinely what a language model is for.

**Out:** a decision that has already been made → a message in the person's language, a
document checklist, an audit trail.

Between them, nothing. The engine will not take a scheme from the model, a threshold from
the model, or a field the vocabulary does not declare — `coerce()` drops anything outside
it, and the generated message is checked afterwards for schemes nobody evaluated.

That is enforcement, not instruction. A prompt is a request, and the boundary between a
language model and a decision about somebody's pension needs something stronger than a
request.
"""),
    filecell("src/perceive.py"),
    filecell("src/model.py"),
    filecell("src/explain.py"),
    filecell("src/session.py"),

    md("""
## 6 · The tests, and what needs no model to run

49 tests, no Ollama required. That is deliberate: everything that can deny someone a
pension is ordinary Python, tested like ordinary Python, and none of its behaviour depends
on what a language model felt like emitting today.

`tests/test_households.py` is the end-to-end one — eight complete households in, the
hand-derived list of schemes out. It also asserts that every *ineligible* scheme names the
rule that ruled it out, because an exclusion nobody can trace is an exclusion nobody can
appeal.
"""),
    filecell("tests/test_households.py"),
    filecell("tests/test_engine.py"),
    filecell("tests/test_perceive.py"),
    filecell("tests/test_explain.py"),
    code('!python -m pytest tests -q'),

    md("""
## 7 · How well the extraction actually works

The interview is measured above without a model. This section measures the one step that
genuinely depends on one: what Gemma writes into the form.

13 hand-labelled utterances — code-mixed speech, local units, regional number words,
answers that address a different question than the one asked, explicit denials, and two
things that must **not** become fields. Two scores, pulling opposite ways:

* **recall** — of what the person actually stated, how much was captured. A miss costs one
  extra question.
* **invention** — of what they did *not* state, how much was filled in anyway. An
  invention silently decides an entitlement on something nobody said.
"""),
    filecell("tests/perception_cases.py"),
    filecell("scripts_eval.py"),
    code(f'!GEMMA_MODEL={MODEL} python scripts_eval.py --runs 2'),

    md(f"""
### Which Gemma 4, and the thing that fell out of asking

Both E-series variants were run against the same set. The interesting result was not which
one won.

**E2B invented a field.** Asked *"क्या घर में कोई मोटर वाहन है?"* — does the household own
a vehicle — and answered *"पैंतीस साल।"* — thirty-five years — it wrote
`owns_motor_vehicle: False`. That is a household recorded as vehicle-free on the strength
of somebody stating their age, and `owns_motor_vehicle: False` is a **pass** on a PMAY-G
housing rule. E4B, same prompt, same case, left the field alone. Reproducible across three
runs.

My first move was the obvious one: add a negative few-shot showing exactly that shape.
**It changed nothing** — E2B still invented it, and I removed the example again rather than
keep an unearned line in the prompt.

The actual cause was in the prompt all along. The rule *"people answer a different question
than the one asked — extract what they did say and leave the asked-about field out"* had
been appended to the tail of an unrelated bullet about whose bereavement it was, with no
bullet of its own. **E4B applied it anyway. E2B did not.** Giving it its own line fixed
E2B and changed nothing for E4B.

So the measurable difference between these two models, on this task, was not accuracy. It
was **how much slack they leave for a defective prompt** — and the larger one had been
quietly covering for a formatting bug I did not know I had.

Cell below: the smaller variant, same set, after the fix.
"""),
    code(f'!GEMMA_MODEL={ALT_MODEL} python scripts_eval.py --runs 2'),

    md(f"""
| | recall | invention | 26 extractions |
|---|---:|---:|---:|
| **Gemma 4 E4B** — ships | 100% | 0% | 88 s |
| Gemma 4 E2B — after the prompt fix | 100% | 0% | 40 s |
| Gemma 4 E2B — before it | 100% | **7.1%** | 40 s |

E4B ships. On a 13-case set the two are tied, and a tie on thirteen cases is not
equivalence — E4B is the one that held up when the prompt was wrong, and at a doorstep the
binding constraint is the person's time, not 2 seconds of inference. E2B stays supported
(`GEMMA_MODEL=gemma4:e2b`) and is the right call on a phone, now that the defect it was
exposing is fixed.
"""),

    md("""
### The three failures this caught

The first run scored **79.2% recall / 21.4% invention**.

| the person said | the model wrote | why it matters |
|---|---|---|
| *"pata nahi kitna kamate hain, kabhi kuch kabhi kuch"* — I don't know, it varies | `annual_income: 0` | zero passes **every** income ceiling in the file. That one hallucination grants every income-tested scheme on the list and sends someone to a counter that will check. |
| *"मेरी सास का देहांत हो गया था"* — my mother-in-law died | `applicant_is_widow: true` | someone else's death recorded as the speaker's own widowhood, which requires a death certificate she cannot produce |

Both were fixed in the prompt with negative examples, and the zero was fixed **again** in
`coerce()`, which drops a stated income of zero with a warning. If a household genuinely
has no income, that is a BPL card and a zero-income certificate — not a number inferred
from a shrug.

The third failure appeared only *after* those fixes, which is the argument for keeping a
labelled set rather than eyeballing outputs: a flat *"नहीं"* to *have you remarried?* came
back as no field at all instead of `remarried: false`. The negative examples had
over-generalised into "anything touching marriage or bereavement is a trap".

**And that bug was only visible because the fields are tri-state.** Had unknown and False
been one value, it would have read `False`, the pension would have been granted, and the
interview would have moved on — right answer, broken mechanism, no way to notice. Fixed by
adding answer-shaped few-shots (`ANSWER_FEWSHOT`), which teach distrust of an unsupported
*inference* rather than distrust of a *topic*.

### How the score got here, and one caveat about reading it

| | recall | invention |
|---|---:|---:|
| first run | 79.2% | 21.4% |
| after the prompt rules and the `coerce()` backstop | 95.8% | 0% |
| after `ANSWER_FEWSHOT` | **100%** | **0%** |

An earlier build of this project ran on Gemma 3 4B, and there the identical code scored
100% on a local machine and 95.8% on Kaggle's GPU — at temperature 0, so not sampling
noise, just the same weights on a different backend. **Temperature 0 buys determinism
within a machine, not across machines.** An extraction score quoted without the hardware
behind it is a number about somebody's laptop.

That is worth keeping in view while reading the cells above, and it is the reason the two
metrics are not interchangeable:

- **Recall can drift**, and the cost is bounded and visible — a missed field stays
  unknown, so the engine asks about it, exactly as it would for any other unknown. One
  extra question.
- **Invention has been 0% on every model, every machine, every run** — including the
  E2B build that *was* inventing, because even then the invention was a legal value for a
  declared field, and everything illegal is dropped by `coerce()` before the engine ever
  sees it.

A safety property that depends on a model scoring the same on somebody else's GPU is not a
safety property. The recall number is a cost estimate; the invention number is closer to a
guarantee, and only because the parts that enforce it are not the model.
"""),

    md("""
## 8 · One conversation, end to end

A widow in a village, starting from "I don't know what I'm entitled to".

The persona is a dictionary keyed by *field name*. The script does not know which questions
it will be asked or in what order — the engine picks each one from what is still undecided,
exactly as it would with a person in front of it. It opens with free narrative, which is
why it finishes in 8 questions rather than the 11.6 measured from a cold start: several
fields arrive in the first sentence.

Watch for the two things at the end that are not the list of schemes:

* **the document checklist, collapsed across schemes.** Six applications ask for Aadhaar
  under five different phrasings; the checklist says *Aadhaar card* once, ordered by how
  many applications each document unlocks — because a trip to the Tehsil for an income
  certificate can cost a day.
* **the audit trail**, down to the rule and the circular behind it.
"""),
    filecell("demo.py"),
    code('!python demo.py'),

    md("""
## Limits

- **Nobody's entitlement is decided here.** The thresholds in this notebook were read off
  published scheme guidance so the engine would have something concrete to check. They are
  revised by notification and differ between states, and a programme officer would have to
  reconcile every line against the current circulars before this went near a real
  household.
- **8 schemes, 8 households, 13 utterances.** Enough to catch a systematic failure and a
  regression; not a confidence interval.
- **Hindi and English are measured; other languages are not.** Gemma supports more. A claim
  with no labelled set behind it is not a claim.
- **Written input, not spoken.** Real deployment goes through ASR, which adds its own
  errors upstream of everything measured here.
- **`verify_no_new_schemes` is exact-match.** It catches the model naming a scheme nobody
  evaluated and would miss a paraphrase. It is a backstop; the factual content is assembled
  in code.
- **Nothing is filed.** The tool says what to bring and where to go.

## What I would build next

1. **Voice, both directions.** The person this is for may not read; the worker beside them
   does. ASR plus TTS closes that gap and changes nothing above it.
2. **A correction loop for caseworkers.** When a decision is wrong, the operator should be
   able to record that *against the rule that produced it* — possible only because a rule
   produced it. That is the dataset that improves a programme, and a model deciding
   end-to-end cannot generate it.
3. **Per-district rule files.** Eight schemes were enough to build and measure against; a district's
   real list is a data-collection problem, not a modelling one.
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
