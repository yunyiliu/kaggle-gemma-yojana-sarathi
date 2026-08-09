# योजना सारथी · Yojana Sarathi

**An offline welfare-entitlement navigator, where Gemma fills in the form and plain code
decides who qualifies.**

Built for *Build with Gemma: TFUG Prayagraj [AI Prayagraj]*.

---

## The problem

India runs hundreds of welfare schemes. A large share of the money set aside for them is
never claimed, and the reason is rarely that people are unwilling. It is that nobody at
the doorstep knows which of the hundreds apply to *this* household, what documents each
needs, and which office to walk to. The people with the strongest claim — a widow with no
income, an elderly person living alone — are the least likely to be able to read the
guidelines that describe their own entitlement.

The decision is not really a judgement call. Eligibility is written down: age thresholds,
income ceilings, whether the house is kutcha, whether the ration card is BPL. It is
gatekeeping by paperwork, not by discretion.

What is missing is a way to get from *how a person describes their life* to *which written
rules that satisfies* — and to do it where the person is, without uploading their
bereavement and income to anyone's API.

## What this does

```
She says (Hindi / Hinglish / English)
  "मेरे पति का पिछले साल देहांत हो गया। दो बेटियाँ स्कूल जाती हैं। कच्चा घर है।"
        │
        ▼   Gemma 4  ·  PERCEPTION
  {applicant_is_widow: true, primary_earner_died: true, years_since_earner_death: 1,
   has_school_age_child: true, child_in_school: true, house_type: "kutcha", ...}
        │
        ▼   schemes.yaml + engine.py  ·  DECISION   (no model here)
  eligible / ineligible / UNCERTAIN, and for each uncertain one, exactly what is missing
        │
        ▼   engine.py  ·  WHAT TO ASK NEXT
  the single fact whose answer resolves the most benefit  →  one question, not twenty
        │
        ▼   Gemma 4  ·  ACTION
  a message in her language, a document checklist deduplicated across schemes,
  and an audit trail a caseworker can check line by line
```

After **8 questions**, the walkthrough in `demo.py` takes a widow in a village from
"I don't know what I'm entitled to" to six schemes worth about **₹6.75 lakh plus ₹1,000 a
month**, with the reason for every inclusion and exclusion written down.

## The design decision everything else follows from

**Gemma never decides eligibility.**

The costs here are asymmetric in both directions and neither is acceptable. Tell someone
they qualify when they do not, and they spend a day's wage and a bus fare on a counter
that turns them away. Tell them they do not qualify when they do, and they lose money
they are owed, possibly for years, with no way to find out. Neither error is one to hand
to a model that cannot show its working.

So the model is given the job it is genuinely good at — turning Awadhi-inflected Hindi,
code-mixed English, bighas and *lakhs* into structured fields — and none of the job where
being confidently wrong is expensive. Between the two model calls sits `schemes/schemes.yaml`,
which a programme officer can read line by line against the published guidelines.

This is also why a 4B model is enough. It is never asked to reason about entitlement.

## Three things that are not obvious until you build it

**1. "She didn't say" and "she said no" cannot be the same value.**

Every fact is tri-state. A missing fact becomes a *question*; only an explicit denial
becomes `False`. Collapsing the two is precisely how a benefits system quietly denies
people things — the interview stops early and the tool reports "not eligible" when the
truthful answer was "we never asked".

The evaluation caught the model doing this: a flat "नहीं" to *have you remarried?* came
back as no fact at all rather than `remarried: false`. Fixed by showing it answer-shaped
examples (`ANSWER_FEWSHOT` in `src/perceive.py`).

**2. The engine chooses the question, not the model.**

After each answer, `next_question()` recomputes what is still undecided and names the one
fact whose value would resolve the most benefit — weighted by the money at stake and by
how close each scheme is to a decision. Starting from nothing it asks about the BPL card
first, because that single fact gates ₹30,000 *and* a ₹5,00,000 health cover.

A person seeking help has limited time and patience. A question that cannot change any
answer spends both for nothing.

**3. An unstated zero is more dangerous than a wrong number.**

Asked "how much do you earn", a person said *"pata nahi, kabhi kuch kabhi kuch"* — I don't
know, it varies. The model returned `annual_income: 0`. Zero satisfies every income
ceiling in the file, so that one hallucination would have granted every income-tested
scheme on the list. Fixed in the prompt, and then again in code, because a prompt is a
request and this needed a guarantee.

## How well does the perception step actually work

Most of the risk lives in one place: what the model writes into the form. So it is
measured rather than demonstrated. `tests/perception_cases.py` holds 13 hand-labelled
utterances covering code-mixed speech, *bighas* and *lakhs*, answers that address a
different question than the one asked, and explicit denials.

```
$ python scripts_eval.py --runs 2

recall    24/24 = 100.0%    a miss costs one extra question
invention  0/14 =   0.0%    an invention decides an entitlement nobody stated
```

Both numbers matter, in opposite directions, and invention is the one to watch. Before
the fixes above it read **79.2% recall / 21.4% invention** — the two inventions being the
zero income and a mother-in-law's death recorded as the speaker's own widowhood.

Those figures are from six runs on a local machine, where they are stable. The same code
on Kaggle's P100 scores **95.8% / 0%**, missing `years_since_earner_death` once. Temperature
is 0, so that is not sampling noise — it is the same weights on a different backend, and it
means temperature 0 buys determinism within a machine, not across machines.

What moved and what did not is the whole point. Recall moved by one case, and a miss costs
one extra question: the engine sees the fact is still unknown and asks. Invention stayed at
0% on both machines on every run — not because the model behaved, but because `coerce()`
drops anything outside the vocabulary and drops a zero income, and because entitlement was
never the model's call. The recall number is a cost estimate; the invention number is a
guarantee, and it is one only because it lives in code.

## Why an open model, running locally

Three constraints each independently rule out a hosted API:

1. **The data is not ours to send.** A household's bereavement, income, and caste details
   are about as sensitive as personal data gets.
2. **Connectivity is worst where need is highest.** A tool that requires a network fails
   at exactly the doorstep it was built for.
3. **Per-token billing does not survive contact with a public programme.** Cost per
   query × a lakh of frontline workers × a daily visit schedule is not adoptable.

The default backend is a local Ollama server. No API key appears anywhere in this repo.

## Running it

```bash
ollama serve &            # in another terminal
ollama pull gemma3:4b

python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q     # 27 tests, no model needed
.venv/bin/python demo.py                # the full walkthrough
.venv/bin/python scripts_eval.py --runs 2   # perception measurement
```

The test suite runs without a model. That is deliberate: the component that can deny
someone a pension is ordinary Python with ordinary tests, and none of its behaviour
depends on what a language model felt like emitting today.

## Layout

| Path | What it is |
|---|---|
| `schemes/schemes.yaml` | The rules. Every scheme cites the document it came from. |
| `src/facts.py` | The closed vocabulary the model may emit, and how to ask about each field. |
| `src/engine.py` | Eligibility, and the choice of what to ask next. **No model.** |
| `src/perceive.py` | Speech → facts. Schema-bounded; anything outside the vocabulary is dropped. |
| `src/explain.py` | Decision → words. Content assembled in code, wording generated, output verified. |
| `src/session.py` | The loop. |
| `scripts_eval.py` | Perception measurement. |

## Limits, stated plainly

- **Prototype, not an eligibility determination.** Thresholds are transcribed from
  published guidance to demonstrate an auditable architecture. They are revised by
  notification and vary by state, and every line needs a programme officer's sign-off
  against the current circulars before any real use.
- **8 schemes**, chosen to cover the common situations of a rural UP household. The rule
  file is the extension point; adding a scheme is a YAML entry, not a code change.
- **Hindi and English measured; other languages untested.** Gemma supports many more, but
  a claim without a labelled set behind it is not a claim.
- **The `verify_no_new_schemes` check is exact-match**, so it catches a model volunteering
  a scheme by name and would miss a paraphrase. It is a backstop, not the safeguard — the
  factual content is assembled in code.
- **No live form submission.** The tool tells a person what to bring and where to go; it
  does not file anything on their behalf.
