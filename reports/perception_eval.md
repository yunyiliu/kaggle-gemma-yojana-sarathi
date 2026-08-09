# Measuring the perception step

`src/perceive.py` is the only place where a language model's output becomes an input to an
entitlement decision. Everything downstream of it — the engine, the rules, the audit trail
— is ordinary Python with ordinary tests. So this is the one component whose behaviour has
to be measured rather than asserted.

## What is measured

`tests/perception_cases.py` holds 13 hand-labelled utterances. They were written to cover
what actually goes wrong at a counter, not what is easy to extract:

- free narrative in Hindi, where several facts arrive in one sentence
- code-mixed Hinglish, as people actually speak
- local units (*bigha*) and regional number words (*lakh*, *hazaar*)
- answers to a specific question, including bare "हाँ" / "नहीं"
- an answer that addresses a **different** question than the one asked
- an explicit "I don't know"
- a bereavement that is *not* the applicant's widowhood

Each case carries two labels:

| field | meaning |
|---|---|
| `expect` | facts the utterance genuinely determines |
| `forbid` | facts it does **not** determine, however reasonable a guess would be |

Two scores fall out, and they pull in opposite directions:

- **recall** — of the facts in `expect`, how many were captured. A miss costs one extra
  question, and nothing worse.
- **invention** — of the facts in `forbid`, how many were emitted anyway. An invention
  silently decides an entitlement on something nobody said.

A system optimised for recall alone will score well and be dangerous. Invention is the
number to watch.

```
$ python scripts_eval.py --runs 2
```

Temperature is 0 and `--runs 2` re-runs every case, so a disagreement between runs is
itself a finding.

## Results

| | recall | invention |
|---|---:|---:|
| first run | 79.2% | 21.4% |
| after prompt rules, negative few-shots, and the code backstop | 95.8% | 0% |
| after `ANSWER_FEWSHOT` | **100%** | **0%** |

Measured on a local machine (Apple silicon, Ollama), stable across six runs.

### The same code scores differently on different hardware

Run on Kaggle's Tesla P100, the final configuration scores **95.8% / 0%** — one miss of
`years_since_earner_death` in `widow_narrative_hi`, from *"मेरे पति का पिछले साल देहांत हो
गया"*. The `demo.py` run in the same session extracts it correctly from the same sentence.

Temperature is 0, so this is not sampling noise. It is the same weights on a different
backend: different kernels, different quantisation arithmetic, a different order of
floating-point operations. **Temperature 0 gives determinism within a machine, not across
machines.** An extraction score quoted without the hardware it ran on is a number about
someone's laptop.

What moved and what did not is the useful part:

- **Recall moved by one case.** The cost of a miss is bounded and visible: the fact stays
  unknown, so `next_question()` asks about it, exactly as it would for any other unknown.
  The conversation gets one question longer.
- **Invention was 0% on both machines, on every run.** That is not the model behaving well
  on the day. It is `coerce()` dropping every field outside the vocabulary, `coerce()`
  dropping a stated income of 0, and the eligibility decision never being the model's to
  make.

A safety property that depends on a model scoring the same on someone else's GPU is not a
safety property. The recall number is a cost estimate; the invention number is a guarantee,
and it is a guarantee only because it is enforced in code.

## The three failures, and what each one taught

### 1. `annual_income: 0` from "I don't know"

> *"Pata nahi kitna kamate hain, kabhi kuch kabhi kuch."* — I don't know how much they
> earn, sometimes this, sometimes that.

The model returned `annual_income: 0`.

Zero is not a small error. It satisfies **every** income ceiling in `schemes.yaml`, so that
single hallucination would have granted every income-tested scheme on the list — reported
to a person as an entitlement, sending them to a counter that will turn them away.

Fixed twice, deliberately:

1. in the prompt — an explicit rule that "I don't know" is not a number and is never zero,
   plus this exact utterance as a negative few-shot returning `{}`;
2. in `coerce()` — a stated income of 0 is dropped with a warning.

The second fix is the one that matters. A prompt is a request; the boundary between a
language model and a benefits decision needs a guarantee. If a household genuinely has no
income, that is `bpl: true` and a zero-income *certificate*, not a number the model
inferred from a shrug.

### 2. A mother-in-law's death recorded as the speaker's widowhood

> *"मेरी सास का देहांत हो गया था।"* — my mother-in-law had passed away.

The model returned `applicant_is_widow: true` and `primary_earner_died: true`.

Neither follows. A death in the family is not the applicant's widowhood, and not
necessarily the household's earner. Downstream this routes someone to a widow pension that
requires a death certificate naming her husband — a document she cannot produce.

Fixed with an explicit rule about *whose* death is being described, plus the utterance as a
negative few-shot.

### 3. The over-correction: "नहीं" to *have you remarried?* returned nothing

This one only appeared **after** the two fixes above, which is the entire argument for
keeping a labelled set rather than eyeballing outputs.

> Q: *क्या आपने दोबारा विवाह किया है?* — have you remarried?
> A: *"नहीं, दोबारा शादी नहीं की।"* — no, I have not remarried.

The model returned `{}` instead of `remarried: false`.

The negative examples had over-generalised into "anything touching marriage or bereavement
is a trap". And because a missing fact becomes a *question*, the interview would have
looped: asking about remarriage, being told no, and asking again.

This is where the tri-state design earns its cost. Had unknown and False been the same
value, the bug would have been invisible — the fact would have read `False`, the pension
would have been granted, and the interview would have moved on. Correct outcome, broken
mechanism, and no way to tell.

Fixed by adding `ANSWER_FEWSHOT` in `src/perceive.py`: few-shot examples in the *shape* a
real answer arrives in — a question, then a short reply — rather than only free narrative.
The negative examples were teaching the model to distrust a topic; the answer-shaped
examples taught it to distrust an *unsupported inference*, which is the actual rule.

## What this measurement does not cover

- **13 cases is small.** It is enough to catch a systematic failure and to notice a
  regression; it is not a confidence interval.
- **Hindi and English only.** Gemma supports many more languages, but a claim without a
  labelled set behind it is not a claim.
- **Written, not spoken.** Real input will arrive through ASR, which will add its own
  errors upstream of everything measured here.
- **The cases are mine.** They were written by someone who knows what the extractor does.
  Utterances collected from actual doorstep interviews would be harder, and would be the
  right next step before any deployment.
