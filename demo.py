"""End-to-end walkthrough of one conversation.

Answers are matched to the questions the engine actually asks, which is the whole point:
the script does not know in advance what will be asked, because the engine decides that
from what is still undecided.
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, ".")
from src.engine import audit_trail, load_schemes                     # noqa: E402
from src.explain import document_checklist                            # noqa: E402
from src.model import ollama_chat                                     # noqa: E402
from src.session import Session                                       # noqa: E402

# What this person would say if asked. A real deployment asks out loud; this stands in
# for the microphone so the walkthrough is reproducible.
PERSONA = {
    "bpl": "हाँ, अंत्योदय कार्ड है।",
    "applicant_age": "पैंतीस साल की हूँ।",
    "remarried": "नहीं, दोबारा शादी नहीं की।",
    "annual_income": "साल भर में मुश्किल से चालीस हज़ार।",
    "owns_farmland": "नहीं, अपनी ज़मीन नहीं है।",
    "owns_motor_vehicle": "नहीं, कोई गाड़ी नहीं है।",
    "monthly_income": "महीने में तीन-चार हज़ार।",
    "house_type": "कच्चा घर है।",
    "has_girl_child": "हाँ, दोनों बेटियाँ हैं।",
    "girl_children_count": "दो बेटियाँ।",
    "deceased_age_at_death": "चालीस के आसपास थे।",
    "years_since_earner_death": "पिछले साल की बात है।",
    "is_income_tax_payer": "नहीं, टैक्स नहीं भरते।",
    "holds_government_post": "नहीं, कोई सरकारी नौकरी नहीं।",
    "has_school_age_child": "हाँ, दोनों स्कूल जाती हैं।",
    "child_in_school": "हाँ, स्कूल जाती हैं।",
    "residence": "गाँव में रहते हैं।",
    "farmland_hectares": "ज़मीन नहीं है।",
    "primary_earner_died": "हाँ, वही कमाते थे।",
    "applicant_is_widow": "हाँ, विधवा हूँ।",
}

OPENING = ("मेरे पति का पिछले साल देहांत हो गया। दो बेटियाँ स्कूल जाती हैं। "
           "गाँव में रहते हैं, कच्चा घर है।")


def main() -> None:
    schemes = load_schemes("schemes/schemes.yaml")
    s = Session(schemes=schemes, chat_fn=ollama_chat, lang="hi", max_questions=8)
    t0 = time.time()

    print("SHE SAYS:\n  " + OPENING + "\n")
    s.hear(OPENING)
    print("  extracted:", s.facts.known(), "\n")

    while True:
        fact, question, resolves = s.ask()
        if fact is None:
            print("-- engine: no remaining question can change any outcome\n")
            break
        answer = PERSONA.get(fact, "पता नहीं।")
        print(f"ASK  [{fact}]  {question}")
        print(f"     (asked because it decides: {', '.join(resolves)})")
        print(f"SHE  {answer}")
        s.hear(answer, question=question, expect_fact=fact, resolves=resolves)
        print()

    r = s.result()
    n_q = len(s.turns) - 1
    print(f"=== after {n_q} questions, {time.time()-t0:.0f}s ===\n")
    print("ENTITLED TO APPLY:")
    for x in r["eligible"]:
        print(f"  - {x.scheme['name_en']}: {x.scheme['benefit_en']}")
    print("\nSTILL UNRESOLVED:")
    for x in r["uncertain"]:
        print(f"  - {x.scheme['name_en']} (needs {', '.join(x.missing)})")
    print("\nRULED OUT (with the rule that ruled it out):")
    for x in r["ineligible"]:
        print(f"  - {x.scheme['name_en']}: {x.failed[0].describe()}")

    print("\n=== DOCUMENTS TO COLLECT (each one once) ===")
    for doc, used_by in document_checklist(r["eligible"]):
        print(f"  {doc}  ->  {', '.join(used_by)}")

    print("\n=== AUDIT TRAIL for one decision ===")
    if r["eligible"]:
        print(audit_trail(r["eligible"][0]))

    print("\n=== MESSAGE TO THE PERSON (Gemma writes, engine decided) ===")
    text, problems = s.message()
    print(text)
    print("\nverification:", "no invented schemes" if not problems else problems)
    if s.warnings:
        print("perception warnings:", s.warnings)


if __name__ == "__main__":
    main()
