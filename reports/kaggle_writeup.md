### 💡 Inspiration

Uttar Pradesh publishes its welfare schemes. The eligibility rules are not secret and they
are not subtle: an age, an income ceiling, a ration-card category, whether the roof is
thatch or concrete. Anyone who sat down with the circulars and a household for an hour
could work out exactly what that household is owed.

**Nobody has the hour.**

An ASHA or a CSC operator has a queue behind her. The household in front of her has a
harvest to get in or a child to collect. The fact set deciding the eight schemes here is
twenty-one fields long, and asked as a form, in form order, most of those questions are
dead weight for any particular household. So the interview never happens and the money
stays unclaimed — not because anybody was refused, but because nobody was asked.

That is a **triage problem under a time budget**, not a knowledge problem. So the thing I
built is not a chatbot that knows about schemes. It is the interview.

### 🛠️ How we built it

**Model: Gemma 4 E4B**, local through Ollama. No API key exists in the repository and no
request leaves the machine — the details this tool collects (a bereavement, an income, a
caste certificate) belong to the household, connectivity is thinnest where need is
greatest, and a per-query bill does not survive multiplication by a state's frontline
workforce.

**Prompt engineering, not RAG and not fine-tuning.** There is nothing to retrieve: the
rules are eight YAML entries I can hold in a prompt-free rule engine, and RAG over them
would only add a failure mode. Fine-tuning in one day, on thirteen labelled utterances,
would fit noise.

**Frameworks: deliberately almost none.** Ollama's HTTP API via `urllib`, PyYAML, pytest.
No LangChain, no agent framework. Everything between the two model calls is ordinary
Python, because the component that can deny someone a pension should be readable by a
programme officer and testable without a GPU.

**Architecture — Gemma at the edges, never in the middle:**

- **In:** free Hindi / Hinglish / English speech → 21 declared fields. *bighas*, *lakhs*,
  "साल भर में चालीस हज़ार".
- **Middle:** `engine.py` + `schemes.yaml`. Eight schemes, every rule citing the circular
  it came from. **No model on this line.**
- **Out:** the decision written back as a message in her language, a document checklist
  deduplicated across schemes, and an audit trail.

`coerce()` drops anything outside the declared vocabulary, so a hallucinated field cannot
reach the engine. The generated message is checked afterwards for schemes nobody
evaluated. That is enforcement, not instruction — a prompt is a request, and the boundary
between a language model and somebody's pension needs more than a request.

Two design choices carry most of the weight. `evaluate()` returns **three** outcomes, not
two: *uncertain* — nothing failed, but a rule can't be checked yet — is what generates the
next question. And every field is **tri-state**, because an unknown field is a question to
ask and a `False` is an answer already given. Collapse them and the tool reports "not
eligible" where the honest answer was "we never asked".

**The result I would defend.** `next_question()` scores each unknown field by the benefit
sitting in *undecided* that knowing it would resolve. That is a claim about an algorithm,
so it is measured — eight households with hand-derived answers, four strategies, no model:

```
fraction of the household's true entitlement secured after k questions
  k              1     2     3     4     5     6     7     8
  benefit      52%   52%   52%   52%   59%   68%   69%   70%
  coverage      0%   52%   52%   52%   52%   64%   69%   69%
  fixed         0%    0%    0%   52%   62%   62%   62%   64%
  random        4%    7%   10%   13%   15%   19%   22%   27%
```

After **one question**, 52% of what the average household is owed is established. A paper
form has established nothing until question four. This is the right metric because
interviews get abandoned, and what you hold when the conversation stops *is* the product.

**The ablation is the honest part.** `coverage` asks whatever unblocks the most schemes,
ignoring value. It *ties* on length (11.6 questions vs 13.2) and is worth nothing at
question one. So benefit weighting is not what makes the interview short — asking a
relevant question does that. It decides *which* half of the money you leave with if the
interview ends early. That is all the table supports.

### 🚀 The Prototype

**Kaggle Notebook (runnable demo):**
https://www.kaggle.com/code/alexyy/yojana-sarathi-offline-welfare-navigator

**GitHub:** https://github.com/yunyiliu/kaggle-gemma-yojana-sarathi

The notebook installs Ollama, pulls both Gemma 4 E-series variants, and runs the real
suite: 49 tests, the interview measurement, the extraction measurement on **both** models,
and one full conversation — a widow in a village going from "I don't know what I'm
entitled to" to six schemes worth ≈₹6.75 lakh plus ₹1,000/month in **seven questions**,
with the rule behind every inclusion and exclusion printed.

### 🧗 Challenges we ran into

**The hardest part was choosing between the two Gemma 4 variants — and what that turned
up.**

E2B **invented a field**. Asked *"क्या घर में कोई मोटर वाहन है?"* — does the household own a
vehicle — and answered *"पैंतीस साल।"* — thirty-five years — it wrote
`owns_motor_vehicle: False`. A household recorded as vehicle-free because somebody stated
their age, and that `False` is a **pass** on a PMAY-G housing rule. E4B, same prompt, same
case, left it alone. Reproducible across three runs.

My first move was the obvious one: a negative few-shot of exactly that shape. **It changed
nothing.** So I removed it rather than keep an unearned line in the prompt.

The cause was in the prompt all along. The rule *"people answer a different question than
the one asked — extract what they did say and leave the asked-about field out"* had been
appended to the tail of an unrelated bullet about whose bereavement it was, with no bullet
of its own. **E4B applied it anyway. E2B did not.** Giving it its own line fixed E2B and
changed nothing for E4B.

So the measurable difference between these two models here was not accuracy. It was **how
much slack they leave for a defective prompt** — the larger one had been quietly covering
for a formatting bug I didn't know I had.

| | recall | invention |
|---|---:|---:|
| **Gemma 4 E4B** — ships | 100% | 0% |
| Gemma 4 E2B — after the fix | 100% | 0% |
| Gemma 4 E2B — before | 100% | **7.1%** |

E4B ships: a tie on thirteen cases is not equivalence, and E4B held up when the prompt was
wrong.

**Two earlier failures, both caught by measurement rather than by looking.** Asked about
income, someone said *"pata nahi, kabhi kuch kabhi kuch"* and the model wrote
`annual_income: 0` — zero passes every income ceiling in the file, so that single
hallucination would have granted every income-tested scheme and sent her to a counter that
will check. Fixed in the prompt, then again in `coerce()`. And a flat *"नहीं"* to *have you
remarried?* came back as no field instead of `remarried: false` — visible **only** because
fields are tri-state; had unknown and False been one value it would have read `False`, the
pension would have been granted, and nothing would have looked wrong.

**Limits, plainly.** Nobody's entitlement is decided here — thresholds were read off
published guidance so the engine had something concrete to check, and would need a
programme officer's sign-off against current circulars. Eight schemes, eight households,
thirteen utterances: enough to catch a systematic failure, not a confidence interval.
Hindi and English are measured; other languages are not. Input is written, not spoken.
Nothing is filed on anyone's behalf.

One caveat on every number: an earlier Gemma 3 build scored 100% locally and 95.8% on
Kaggle's GPU at temperature 0 — same weights, different backend. Both Gemma 4 variants
held at 100% / 0% on both machines. Recall can drift and the cost is bounded (the field
stays unknown, so the engine asks). Invention is held at 0% by code, not by the model
behaving.

**49 tests, none of which need a model to run.** Everything that can deny someone a
pension is in that half.
