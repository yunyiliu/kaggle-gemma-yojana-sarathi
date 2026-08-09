# Yojana Sarathi

## The scheme is not the hard part. The interview is.

*Track: GenAI for Good*

---

Uttar Pradesh publishes its welfare schemes. The eligibility rules are not secret and they
are not subtle: an age, an income ceiling, a ration-card category, whether the roof is
thatch or concrete. Anyone who sat down with the circulars and a household for an hour
could work out exactly what that household is owed.

Nobody has the hour.

A frontline worker in a village has a queue behind her. The household in front of her has
a harvest to get in or a child to collect. The fact set deciding the eight schemes here is
twenty-one fields long, and asked as a form, in form order, most of those questions are
dead weight for any particular household. So the interview does not happen and the
entitlement stays theoretical.

That is a triage problem under a time budget, not a knowledge problem, and it is the one
this project attacks.

## Architecture

Gemma 4 E4B, local through Ollama, does two jobs at the edges of the system. **In:** free
Hindi / Hinglish / English speech into structured fields — *bighas*, *lakhs*, "साल भर में
चालीस हज़ार". **Out:** a decision already made, written back as a message in the person's
language, a document checklist, and an audit trail.

Between them sits `engine.py` plus `schemes.yaml`: eight schemes, each rule citing the
circular it came from. The engine will not accept a scheme, a threshold, or an undeclared
field from the model — `coerce()` drops anything outside the vocabulary, and the generated
message is checked afterwards for schemes nobody evaluated. That is enforcement rather
than instruction, because a prompt is a request and the boundary between a language model
and somebody's pension needs more than a request.

`evaluate()` returns three outcomes, not two. **Uncertain** — nothing failed, but a rule
cannot be checked yet — is what generates the next question. Every field is tri-state for
the same reason: an unknown field is a question to ask, a `False` is an answer already
given. Collapse them and the tool reports "not eligible" where the honest answer was "we
never asked".

## The result I would defend

`next_question()` scores each unknown field by the benefit sitting in *undecided* that
knowing it would resolve, split across whatever else each scheme still waits on. That is a
claim about an algorithm, so it is measured — eight complete households with hand-derived
answers, four ordering strategies, no model involved:

```
fraction of the household's true entitlement secured after k questions
  k              1     2     3     4     5     6     7     8
  benefit      52%   52%   52%   52%   59%   68%   69%   70%
  coverage      0%   52%   52%   52%   52%   64%   69%   69%
  fixed         0%    0%    0%   52%   62%   62%   62%   64%
  random        4%    7%   10%   13%   15%   19%   22%   27%
```

After **one question** the shipped strategy has established just over half of what the
average household is owed. Form order has established nothing until question four. This is
the right metric because interviews get abandoned — someone is called away, a queue moves,
a child starts crying — and what you hold when the conversation stops is the product.

**The ablation is the honest part.** `coverage` asks whatever unblocks the most schemes,
ignoring what they are worth. It *ties* on interview length (11.6 questions against 13.2)
and is worth nothing at question one. So benefit weighting is not what makes the interview
short — asking a relevant question does that, and plain scheme-counting gets you there.
The weighting decides *which* half of the money you leave with if the interview ends early.
That is all the table supports.

Ordering may cost time, never entitlement: a test runs all four strategies over all eight
households and asserts identical conclusions.

## Which Gemma 4 — and what fell out of asking

Both E-series variants were run against the same labelled set. The interesting result was
not which won.

**E2B invented a field.** Asked *"क्या घर में कोई मोटर वाहन है?"* and answered *"पैंतीस
साल।"* — thirty-five years — it wrote `owns_motor_vehicle: False`. A household recorded as
vehicle-free because somebody stated their age, and that `False` is a **pass** on a PMAY-G
housing rule. E4B, same prompt, same case, left it alone. Reproducible across three runs.

My first move was a negative few-shot of exactly that shape. **It changed nothing**, so I
removed it rather than keep an unearned line in the prompt.

The cause was in the prompt all along. The rule *"people answer a different question than
the one asked — extract what they did say and leave the asked-about field out"* had been
appended to the tail of an unrelated bullet about whose bereavement it was, with no bullet
of its own. **E4B applied it anyway. E2B did not.** Giving it its own line fixed E2B and
changed nothing for E4B.

The measurable difference between these two models on this task was not accuracy. It was
how much slack they leave for a defective prompt — and the larger one had been quietly
covering for a formatting bug I did not know I had.

| | recall | invention | 26 extractions |
|---|---:|---:|---:|
| **Gemma 4 E4B** — ships | 100% | 0% | 88 s |
| Gemma 4 E2B — after the fix | 100% | 0% | 40 s |
| Gemma 4 E2B — before | 100% | **7.1%** | 40 s |

E4B ships: a tie on thirteen cases is not equivalence, and E4B is the one that held up when
the prompt was wrong. E2B stays supported for phone-class hardware.

## Measuring extraction: two numbers, opposite directions

13 hand-labelled utterances — code-mixed speech, local units, regional number words,
answers addressing a different question than the one asked, explicit denials. **Recall**:
what was stated and captured; a miss costs one extra question. **Invention**: what was
*not* stated and filled in anyway; an invention silently decides an entitlement nobody
raised.

Invention is the one to watch, and it caught real damage. Asked about income, a person said
*"pata nahi, kabhi kuch kabhi kuch"*. The model wrote `annual_income: 0`. Zero passes every
income ceiling in the file — that single hallucination would have granted every
income-tested scheme and sent someone to a counter that will check. Fixed in the prompt,
then again in `coerce()`, which drops a stated zero income with a warning.

A third failure appeared only *after* those fixes, which is the argument for keeping a
labelled set rather than eyeballing outputs: a flat *"नहीं"* to *have you remarried?* came
back as no field instead of `remarried: false`. The negatives had over-generalised into
"anything touching marriage is a trap". **That bug was only visible because fields are
tri-state** — had unknown and False been one value it would have read `False`, the pension
would have been granted, and nothing would have looked wrong.

One caveat on every number here: an earlier Gemma 3 build scored 100% locally and 95.8% on
Kaggle's GPU at temperature 0. Temperature 0 buys determinism within a machine, not across
machines. Recall can drift and the cost is bounded — the field stays unknown, so the engine
asks. Invention is held at 0% by code, not by the model behaving.

## Why local, and what it is not

The details this tool collects — a bereavement, an income, a caste certificate — belong to
the household. Connectivity is thinnest where need is greatest. A per-query bill does not
survive multiplication by a state's frontline workforce. No API key exists in the
repository.

**Limits, plainly.** Nobody's entitlement is decided here: thresholds were read off
published guidance so the engine had something concrete to check, and would need a
programme officer's sign-off against current circulars. Eight schemes, eight households,
thirteen utterances — enough to catch a systematic failure, not a confidence interval.
Hindi and English are measured; other languages are not. Input is written, not spoken; real
deployment adds ASR errors upstream. Nothing is filed on anyone's behalf.

**Next:** voice both directions; a caseworker correction loop that records a wrong decision
*against the rule that produced it* — possible only because a rule produced it, and the
dataset a model deciding end-to-end cannot generate; and per-district rule files.

49 tests, no model required to run them. Everything that can deny someone a pension is in
that half.
