# योजना सारथी · Yojana Sarathi

**The scheme is not the hard part. The interview is.**

An offline tool that works out which welfare schemes a household can actually claim, by
conducting the interview that nobody currently has time to conduct.

Built for *Build with Gemma: TFUG Prayagraj [AI Prayagraj]*.

---

## The gap this sits in

Uttar Pradesh publishes its welfare schemes. The eligibility rules are not secret, and
they are not subtle: an age, an income ceiling, a ration-card category, whether the roof
is thatch or concrete. Anyone who sat down with the circulars and a household for an hour
could work out precisely what that household is owed.

Nobody has the hour.

A frontline worker visiting a village has a queue behind her. The household in front of
her has a harvest to get in, or a child to collect. The full fact set that decides these
eight schemes is twenty-one fields long — and asked as a form, in form order, most of
those questions are dead weight for any particular household. So the interview does not
happen, and the entitlement stays theoretical.

That is the actual failure, and it is not a knowledge problem. It is a **triage problem
under a time budget**, and it is the one this project attacks.

## The result that matters

`next_question()` picks what to ask by scoring each unknown field against the money it
would unlock — the benefit currently sitting in *undecided* that knowing this one fact
would resolve, divided across whatever else each scheme is still waiting on.

That is a claim about an algorithm, so it is measured, on eight complete households, with
no model in the loop at all:

```
$ python scripts_ask.py

Fraction of the household's true entitlement already secured after k questions

  k                1      2      3      4      5      6      7      8
  -------------------------------------------------------------------
  benefit        52%    52%    52%    52%    59%    68%    69%    70%
  coverage        0%    52%    52%    52%    52%    64%    69%    69%
  fixed           0%     0%     0%    52%    62%    62%    62%    64%
  random          4%     7%    10%    13%    15%    19%    22%    27%
```

**After a single question, the shipped strategy has established just over half of what the
average household is owed.** A paper form, asking the same fields in a fixed order, has
established nothing until question four.

This matters because interviews get abandoned. Someone is called away, a queue moves, a
child starts crying. What you have secured when the conversation stops is the whole
product.

### The ablation is the honest part

`coverage` asks whichever question unblocks the most schemes, ignoring what they are
worth. It ties the shipped strategy on interview *length* — 11.6 questions to a final
answer, against 13.2 for form order — and it is worth nothing at question one.

So the benefit weighting is not what makes the interview short. Asking a *relevant*
question does that, and counting schemes is enough. The weighting does something else and
narrower: it decides **which** half of the money you walk away with if the conversation
ends early. That is the claim the table supports, and it is the only claim it supports.

And ordering is only allowed to touch the person's time. `test_question_order_never_changes_the_conclusion`
runs all four strategies over all eight households and asserts they land on the same
answer every time.

## Rules are data

Every threshold lives in `schemes/schemes.yaml`, cites the document it came from, and is
read by ordinary Python:

```yaml
- id: nfbs
  name_en: National Family Benefit Scheme
  benefit_en: ₹30,000 one-time
  source: NSAP Programme Guidelines, Ministry of Rural Development
  rules:
    - {fact: primary_earner_died, op: eq, value: true}
    - {fact: years_since_earner_death, op: lte, value: 3}
    - {fact: deceased_age_at_death, op: between, value: [18, 60]}
    - {fact: bpl, op: eq, value: true}
```

Adding a scheme is a YAML entry. A rule that references a field the vocabulary does not
declare fails at load, loudly — because the alternative is a scheme that silently never
matches anybody, which is invisible in testing and denies people money in production.

`evaluate()` returns three outcomes rather than two. **Uncertain** — nothing has failed,
but a rule cannot be checked yet — is not a rounding error on the way to "no". It is the
thing that generates the next question, and a tool that collapses it into "no" is a tool
that closes the interview early and reports that someone does not qualify when the honest
answer was that nobody asked.

## Why the fields are tri-state

Every field is `True`, `False`, or unknown, and the distinction is load-bearing precisely
*because* the interview is the product: an unknown field is a question to ask, a `False`
is an answer already given. Collapse them and the engine stops asking about things nobody
ever raised.

This is not theoretical. The evaluation caught the model returning nothing at all for a
flat *"नहीं"* to *have you remarried?* instead of `remarried: false`. Had unknown and
False been the same value, that bug would have been invisible: the field would have read
`False`, the pension would have been granted, the interview would have moved on. Right
answer, broken mechanism, no way to notice.

## Where the model sits, and where it does not

Two narrow jobs, both at the edges:

**In:** free speech → fields. Hindi, Hinglish, English, *bighas*, *lakhs*, "साल भर में
चालीस हज़ार". This is genuinely hard and genuinely what a language model is for.

**Out:** a decision already made → a message in the person's language, plus a document
checklist and an audit trail.

Between them, nothing. The engine does not accept a scheme from the model, a threshold
from the model, or a field the vocabulary does not declare — `coerce()` drops anything
outside it. The output is checked afterwards for schemes nobody evaluated.

This is enforcement, not instruction. A prompt is a request; the boundary between a
language model and a decision about somebody's pension needs something stronger than a
polite request.

### An unstated zero is worse than a wrong number

Asked what the household earns, a person said *"pata nahi, kabhi kuch kabhi kuch"* — I
don't know, it varies. The model wrote `annual_income: 0`.

Zero passes every income ceiling in the file. That one hallucination would have granted
every income-tested scheme on the list and sent someone to a counter that will check. It
is fixed in the prompt, and then again in `coerce()`, which drops a stated income of zero
with a warning. If a household genuinely has no income, that is a BPL card and a
zero-income certificate — not a number inferred from a shrug.

## How well the extraction actually works

`tests/perception_cases.py` holds 13 hand-labelled utterances: code-mixed speech, local
units, regional number words, answers that address a different question than the one
asked, explicit denials, and two things that must *not* become fields.

```
$ python scripts_eval.py --runs 2

recall    24/24 = 100.0%    a miss costs one extra question
invention  0/14 =   0.0%    an invention decides an entitlement nobody stated
```

Two numbers, pulling opposite ways. Before the fixes above it read **79.2% / 21.4%** — the
inventions being the zero income and a mother-in-law's death recorded as the speaker's own
widowhood.

### The same code scores differently on different hardware

Those figures are six stable runs on a local machine. On Kaggle's P100 the same code
scores **95.8% / 0%**, missing `years_since_earner_death` once. Temperature is 0, so that
is not sampling noise — it is the same weights on a different backend.

What moved and what did not is the useful part. Recall moved by one case, and the cost is
bounded and visible: the field stays unknown, so the engine asks about it. Invention was
0% on both machines on every run — not because the model behaved, but because it is
enforced in code.

A safety property that depends on a model scoring the same on somebody else's GPU is not a
safety property. The recall number is a cost estimate; the invention number is a guarantee,
and it is one only because it does not depend on the model.

## What comes out at the end

Beyond the list of schemes:

**A document checklist collapsed across schemes.** Six applications ask for Aadhaar under
five different phrasings; the checklist says *Aadhaar card* once, and orders the list by
how many applications each document unlocks, because a trip to the Tehsil for an income
certificate can cost a day.

**An audit trail per decision**, down to the rule and the circular it came from:

```
Pradhan Mantri Awaas Yojana - Gramin [pmay_g] -> ELIGIBLE
  source: PMAY-G Framework for Implementation, Ministry of Rural Development
  [PASS] residence='rural' ok (eq rural)
  [PASS] house_type='kutcha' ok (in ['none', 'kutcha'])
  [PASS] owns_motor_vehicle=False ok (eq False)
  [PASS] monthly_income=3000.0 ok (lte 15000)
```

An exclusion nobody can trace is an exclusion nobody can appeal, so every ineligible
scheme names the rule that ruled it out, and a test asserts it.

## Running locally, and why

The backend is an Ollama server on `localhost`. There is no API key in this repository
and no request leaves the machine.

That is not a preference. A household's bereavement, income and caste are the details this
tool exists to collect, and they belong to the household. Connectivity is thinnest exactly
where the need is greatest. And a per-query bill does not survive being multiplied by a
state's frontline workforce.

```bash
ollama serve &
ollama pull gemma3:4b

python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q        # 49 tests, no model needed
.venv/bin/python scripts_ask.py            # the interview measurement, no model needed
.venv/bin/python scripts_eval.py --runs 2  # the extraction measurement
.venv/bin/python demo.py                   # one conversation, end to end
```

Two of those four need no model at all. Everything that can deny someone a pension is in
that half.

## Layout

| Path | What it is |
|---|---|
| `schemes/schemes.yaml` | The rules, each citing its source document. |
| `src/facts.py` | The closed vocabulary, and how to ask about each field. |
| `src/engine.py` | Eligibility, and the choice of what to ask next. **No model.** |
| `src/perceive.py` | Speech → fields. Schema-bounded; anything else is dropped. |
| `src/explain.py` | Decision → words, checklist, audit trail. |
| `src/session.py` | The interview loop. |
| `scripts_ask.py` | Measures the interview. |
| `scripts_eval.py` | Measures the extraction. |
| `tests/households.py` | Eight complete households with hand-derived answers. |

## Limits

- **Nobody's entitlement is decided here.** The thresholds in this repository were read
  off published scheme guidance so the engine would have something concrete to check. They
  are revised by notification and differ between states, and a programme officer would have
  to reconcile every line against the current circulars before this went near a real
  household.
- **8 schemes and 8 households.** Enough to catch a systematic failure and a regression;
  not a confidence interval. The rule file is the extension point.
- **Hindi and English are measured. Other languages are not.** Gemma supports more; a
  claim with no labelled set behind it is not a claim.
- **Written input, not spoken.** Real deployment goes through ASR, which adds its own
  errors upstream of everything measured here.
- **`verify_no_new_schemes` is exact-match**, so it catches a model naming a scheme it
  invented and would miss a paraphrase. It is a backstop; the factual content is assembled
  in code.
- **Nothing is filed.** The tool says what to bring and where to go.

## What I would build next

1. **Voice, both directions.** The person this is for may not read. The worker beside them
   does. ASR plus TTS closes that gap and changes nothing above it.
2. **A correction loop for caseworkers.** When a decision is wrong, the operator should be
   able to record that *against the rule that produced it* — possible only because a rule
   produced it. That is the dataset that improves a programme, and a model deciding
   end-to-end cannot generate it.
3. **Per-district rule files.** Eight schemes were enough to build and measure against. A district's
   real list is a data-collection problem, not a modelling one.
