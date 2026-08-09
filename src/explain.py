"""Explanation: turn a decision the engine has already made into words.

The model writes; it does not decide.  Everything factual in the output - which schemes,
which documents, which office, how much - is passed in from the engine and the YAML.
The model's job is to say it in the person's language, in an order that makes sense to
someone who is about to spend a day travelling to a counter.

Why this matters beyond politeness: the failure mode of a chat assistant here is a
fluent sentence that adds a scheme nobody checked, or softens "you need the death
certificate" into "you may want to bring some documents".  Both send a person on a
wasted trip.  So the sections are assembled in code and only the wording is generated,
and `verify_no_new_schemes` checks the output afterwards.
"""
from __future__ import annotations

from .engine import SchemeResult

SYSTEM_PROMPT = """You write short, plain messages for someone with limited literacy who
is about to travel to a government office.

You will be given a decision that has ALREADY been made. Your job is only to say it
clearly in the requested language.

RULES:
- Do NOT add any scheme that is not in the list you are given.
- Do NOT change any amount, document name, or office name.
- Do NOT say "you will definitely get" - say what they are entitled to apply for.
- Short sentences. No bullet symbols, no markdown, no English jargon in a Hindi message.
- Lead with what to do first."""


def render_brief(eligible: list[SchemeResult], uncertain: list[SchemeResult],
                 lang: str = "hi") -> str:
    """The factual brief handed to the model.  Everything here comes from the engine."""
    lines = []
    if eligible:
        lines.append("ENTITLED TO APPLY:")
        for r in eligible:
            s = r.scheme
            name = s["name_hi"] if lang == "hi" and "name_hi" in s else s["name_en"]
            lines.append(f"- {name} ({s['name_en']}): {s['benefit_en']}")
            lines.append(f"  where: {s.get('where', 'local office')}")
            lines.append(f"  documents: {'; '.join(s['documents'])}")
    if uncertain:
        lines.append("")
        lines.append("MAY ALSO QUALIFY, needs one more detail confirmed:")
        for r in uncertain:
            s = r.scheme
            name = s["name_hi"] if lang == "hi" and "name_hi" in s else s["name_en"]
            lines.append(f"- {name}: {s['benefit_en']}")
    if not eligible and not uncertain:
        lines.append("NOTHING MATCHED on the information given.")
    return "\n".join(lines)


def build_prompt(eligible: list[SchemeResult], uncertain: list[SchemeResult],
                 lang: str = "hi") -> list[dict[str, str]]:
    language = {"hi": "Hindi (Devanagari script)", "en": "English"}.get(lang, lang)
    brief = render_brief(eligible, uncertain, lang)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Write the message in {language}.\n\n{brief}\n\n"
            f"Write at most 120 words."},
    ]


def verify_no_new_schemes(text: str, allowed: list[SchemeResult],
                          all_schemes: dict) -> list[str]:
    """Check the generated message did not introduce a scheme nobody evaluated.

    A cheap, exact check: any scheme name in the rule file that appears in the message
    but is not in the allowed set is flagged.  It will not catch a paraphrase, and it is
    not meant to be the only safeguard - the factual content is assembled in code - but
    it does catch the specific failure of the model volunteering a scheme it happens to
    know about.
    """
    ok = set()
    for r in allowed:
        ok.add(r.scheme["name_en"].lower())
        if "name_hi" in r.scheme:
            ok.add(r.scheme["name_hi"])
    problems = []
    low = text.lower()
    for s in all_schemes["schemes"]:
        for key in (s["name_en"].lower(), s.get("name_hi", "")):
            if not key:
                continue
            present = key in (low if key == key.lower() else text)
            if present and key not in ok:
                problems.append(f"message mentions {s['name_en']}, which was not offered")
                break
    return problems


# Schemes name the same document differently - "Aadhaar card", "Aadhaar card of the
# applicant", "Aadhaar and bank passbook of the student". Grouping on the literal string
# produces a list telling someone to fetch their Aadhaar four times, which defeats the
# only purpose the checklist has. Each entry below maps to one physical thing a person
# has to obtain and carry.
DOC_CLASSES: list[tuple[str, tuple[str, ...]]] = [
    ("Aadhaar card", ("aadhaar",)),
    ("Bank account passbook", ("passbook", "bank account")),
    ("Ration card (BPL / Antyodaya)", ("ration card",)),
    ("Income certificate (from the Tehsil)", ("income certificate",)),
    ("Death certificate", ("death certificate",)),
    ("Caste certificate", ("caste certificate",)),
    ("MGNREGA job card", ("job card",)),
    ("Birth certificate of the child", ("birth certificate",)),
    ("School enrolment certificate", ("school enrolment", "enrolment certificate")),
    ("Passport photograph", ("photograph", "photo")),
    ("Land record (khatauni / khasra)", ("land record", "khatauni", "khasra")),
    ("Consent for Aadhaar seeding", ("seeding",)),
]


def normalise_document(raw: str) -> str:
    """Map a scheme's wording onto the physical document it refers to."""
    low = raw.lower()
    for label, keys in DOC_CLASSES:
        if any(k in low for k in keys):
            return label
    return raw


def document_checklist(eligible: list[SchemeResult]) -> list[tuple[str, list[str]]]:
    """Documents grouped so a person collects each one once, not once per scheme.

    A trip to a Tehsil for an income certificate can cost a day. The list is ordered by
    how many applications a document unlocks, so the most load-bearing errand is first.
    """
    need: dict[str, set[str]] = {}
    for r in eligible:
        for d in r.scheme["documents"]:
            need.setdefault(normalise_document(d), set()).add(r.scheme["name_en"])
    return sorted(((d, sorted(v)) for d, v in need.items()),
                  key=lambda kv: (-len(kv[1]), kv[0]))


def explain(eligible: list[SchemeResult], uncertain: list[SchemeResult],
            all_schemes: dict, chat_fn, lang: str = "hi") -> tuple[str, list[str]]:
    text = chat_fn(build_prompt(eligible, uncertain, lang))
    return text.strip(), verify_no_new_schemes(text, eligible + uncertain, all_schemes)
