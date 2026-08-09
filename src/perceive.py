"""Perception: free speech in, structured facts out.

This is the one job the model is given, and it is the job it is actually good at -
turning "my husband passed away last year, two children in school, we have two bighas"
into fields a rule engine can read.  It is asked for nothing else.  It does not see the
eligibility rules, it is not told which schemes exist, and it never states an outcome.

Two properties are enforced in code rather than requested in the prompt, because a
prompt is a request and this needs a guarantee:

* **Closed vocabulary.**  Anything the model emits outside `Facts` is dropped, with a
  warning.  A hallucinated field cannot reach the engine.
* **Silence is not denial.**  A field the person did not mention stays `None`.  The
  model is told explicitly not to guess, and any value it invents for an unmentioned
  field would still have to survive the schema, but the important half of this is that
  the engine treats `None` as "ask", never as "no".

Local units are converted here rather than in the prompt: 1 bigha is not a fixed
quantity nationally, and the conversion used is stated in code where it can be
corrected, instead of buried in a model's arithmetic.
"""
from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any

from .facts import Facts

# 1 bigha varies by region; in western UP it is commonly taken as ~0.25 ha.  Stated here
# so a programme officer can change one number rather than re-engineer a prompt.
BIGHA_TO_HECTARE = 0.25

_ALLOWED = {f.name for f in fields(Facts)}

SYSTEM_PROMPT = """You extract facts from what a person says about their household.

You are filling in a form. You are NOT deciding anything, you are NOT recommending
anything, and you must NOT mention any government scheme.

Return ONLY a JSON object. Use these keys and no others:

applicant_age (int), applicant_is_widow (bool), remarried (bool),
residence ("rural"|"urban"), monthly_income (number, rupees),
annual_income (number, rupees), bpl (bool),
house_type ("none"|"kutcha"|"semi_pucca"|"pucca"), owns_motor_vehicle (bool),
has_girl_child (bool), girl_children_count (int),
has_school_age_child (bool), child_in_school (bool),
owns_farmland (bool), farmland_bighas (number), farmland_hectares (number),
is_income_tax_payer (bool), holds_government_post (bool),
primary_earner_died (bool), years_since_earner_death (number),
deceased_age_at_death (int)

RULES:
- Include a key ONLY if the person actually said it or it follows necessarily.
  "My husband died" gives applicant_is_widow true AND primary_earner_died true if he
  earned. It does NOT give you their income.
- NOT MENTIONED and SAID NO are different, and this distinction matters more than any
  other rule here.
    not mentioned  -> leave the key out
    said no        -> set the key to false. "No vehicle" IS information. Record it.
  Leaving a key out means "we still have to ask". Setting false means "we asked and the
  answer was no". Collapsing the two either wastes the person's time re-asking or
  silently denies them something.
- Land in bighas goes in farmland_bighas. Do not convert it yourself.
- Rupees: "20 hazaar" = 20000. "2 lakh" = 200000.
- If you are told what was asked and the answer does not address it AT ALL, do not fill
  that field. A denial does address it - "no", "nahi", "koi nahi" set it to false.
  This exception is only for answers about something else entirely. An answer that
  addresses the question AND adds more still fills in everything it addresses.
- "I don't know" / "pata nahi" / "पता नहीं" / "kabhi kuch kabhi kuch" is NOT a number and
  NOT a no. Leave the key out. Never write 0 for an amount the person did not state -
  zero income passes every income limit there is.
- Whose event it was matters. A death in the family is only applicant_is_widow if it was
  the speaker's husband, and only primary_earner_died if that person earned. A
  mother-in-law, parent, or sibling dying gives you neither. People answer a different question than the one asked, especially when tired or
  hard of hearing. Extract whatever they did say and leave the asked-about field out.
  Returning {} is a correct answer.

Output the JSON object and nothing else."""

# The negative examples carry more weight than the positive ones. Both inventions the
# evaluation caught - an amount conjured from "I don't know", and the speaker's
# widowhood conjured from someone else's death - are shown here returning {}.
FEWSHOT: list[tuple[str, dict]] = [
    (
        "मेरे पति का पिछले साल देहांत हो गया। दो बच्चे स्कूल जाते हैं। गाँव में रहते हैं।",
        {"applicant_is_widow": True, "primary_earner_died": True,
         "years_since_earner_death": 1, "has_school_age_child": True,
         "child_in_school": True, "residence": "rural"},
    ),
    (
        "I am 65. I live alone in the village. Kutcha house. No income now.",
        {"applicant_age": 65, "residence": "rural", "house_type": "kutcha"},
    ),
    (
        "Pata nahi kitna kamate hain, kabhi kuch kabhi kuch.",
        {},
    ),
    (
        "मेरी सास का देहांत हो गया था।",
        {},
    ),
    (
        "अड़तालीस की हूँ, और दो बेटियाँ हैं।",
        {"applicant_age": 48, "has_girl_child": True, "girl_children_count": 2},
    ),
]


# Shown in the same shape a real answer arrives in, because the bare-utterance examples
# above do not teach the model what to do with a one-word reply. Without this, a flat
# "no" to the remarriage question came back as {} - the negative examples above had
# taught it that anything touching marriage or bereavement was a trap.
ANSWER_FEWSHOT: list[tuple[str, str, str, dict]] = [
    ("क्या आपने दोबारा विवाह किया है?", "remarried", "नहीं।", {"remarried": False}),
    ("क्या आपके पास बीपीएल या अंत्योदय राशन कार्ड है?", "bpl", "हाँ, है।", {"bpl": True}),
    ("क्या घर में कोई मोटर वाहन है?", "owns_motor_vehicle", "अभी तक तो कुछ समझ नहीं आया।", {}),
]


def _answer_turn(question: str, expect_fact: str | None, utterance: str) -> str:
    hint = f'The person was asked: "{question}"'
    if expect_fact:
        hint += (f'\nIf their answer addresses it, it sets "{expect_fact}". '
                 f'If it does not address it, leave "{expect_fact}" out entirely.')
    return f"{hint}\n\nThey answered: {utterance}"


def build_prompt(utterance: str, question: str | None = None,
                 expect_fact: str | None = None) -> list[dict[str, str]]:
    """Build the extraction prompt.

    When the utterance is an answer to a question we asked, the question goes in too.
    Without it the model is reading "yes" or "forty thousand" with no idea what was
    asked, and it guesses - in testing it read "barely forty thousand for the year" as a
    monthly figure and produced an annual income twelve times too high, which flipped
    three schemes from eligible to ineligible. Bare numbers and bare yes/no need the
    question to mean anything.
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, out in FEWSHOT:
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    if question:
        for q, ef, a, out in ANSWER_FEWSHOT:
            msgs.append({"role": "user", "content": _answer_turn(q, ef, a)})
            msgs.append({"role": "assistant",
                         "content": json.dumps(out, ensure_ascii=False)})
        msgs.append({"role": "user",
                     "content": _answer_turn(question, expect_fact, utterance)})
    else:
        msgs.append({"role": "user", "content": utterance})
    return msgs


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of whatever the model returned."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model output")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in model output")


def coerce(raw: dict[str, Any]) -> tuple[Facts, list[str]]:
    """Map raw model output onto the closed vocabulary, dropping anything else."""
    warnings: list[str] = []
    clean: dict[str, Any] = {}

    if "farmland_bighas" in raw and raw["farmland_bighas"] is not None:
        try:
            ha = float(raw.pop("farmland_bighas")) * BIGHA_TO_HECTARE
            clean["farmland_hectares"] = round(ha, 3)
            clean.setdefault("owns_farmland", True)
        except (TypeError, ValueError):
            warnings.append("farmland_bighas was not a number")

    for k, v in raw.items():
        if v is None:
            continue
        if k not in _ALLOWED:
            warnings.append(f"dropped unknown field {k!r}")
            continue
        clean.setdefault(k, v)

    # Backstop for the one invention with an asymmetric cost. Zero income satisfies every
    # income ceiling in the rule file, so a zero the person did not actually state would
    # grant them everything. The prompt now handles this, but a prompt is a request; this
    # is the guarantee. A genuinely zero-income household reaches the same place by a
    # different route - the field stays unknown and the engine asks.
    for _f in ("annual_income", "monthly_income"):
        if _f in clean and float(clean[_f]) == 0.0:
            warnings.append(f"dropped {_f}=0: an unstated zero would pass every "
                            f"income ceiling; treating it as unknown so the engine asks")
            clean.pop(_f)

    # a stated hectare figure implies ownership; the engine should not have to infer it
    if clean.get("farmland_hectares") and "owns_farmland" not in clean:
        clean["owns_farmland"] = True
    # Rules read both the monthly and the annual figure, so one is derived from the
    # other. This is arithmetic, not inference - but it does mean a misread period
    # propagates, which is why the question context above matters as much as it does.
    if "annual_income" in clean and "monthly_income" not in clean:
        clean["monthly_income"] = round(float(clean["annual_income"]) / 12, 2)
    elif "monthly_income" in clean and "annual_income" not in clean:
        clean["annual_income"] = round(float(clean["monthly_income"]) * 12, 2)

    typed: dict[str, Any] = {}
    for f in fields(Facts):
        if f.name not in clean:
            continue
        v = clean[f.name]
        try:
            if "bool" in str(f.type):
                typed[f.name] = bool(v)
            elif "int" in str(f.type):
                typed[f.name] = int(float(v))
            elif "float" in str(f.type):
                typed[f.name] = float(v)
            else:
                typed[f.name] = v
        except (TypeError, ValueError):
            warnings.append(f"dropped {f.name}={v!r}: wrong type")
    return Facts(**typed), warnings


def perceive(utterance: str, chat_fn, question: str | None = None,
             expect_fact: str | None = None) -> tuple[Facts, list[str], str]:
    """Run one perception pass.  `chat_fn(messages) -> str` keeps the model swappable."""
    reply = chat_fn(build_prompt(utterance, question, expect_fact))
    try:
        raw = _extract_json(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        return Facts(), [f"could not parse model output: {exc}"], reply
    facts, warnings = coerce(raw)
    return facts, warnings, reply
