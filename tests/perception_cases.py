"""Hand-labelled utterances for measuring the perception step.

Written to cover what actually goes wrong at a counter rather than what is easy to
extract: code-mixed Hindi and English, local units, regional number words, answers that
address a different question than the one asked, and denials - which must come back as
`False`, not as silence.

`expect` lists only the fields the utterance genuinely determines. A field the speaker
did not establish is listed in `forbid`: emitting it is a hallucination even if the
guess would be reasonable, because downstream it becomes an entitlement decision nobody
asked about.
"""
from __future__ import annotations

CASES: list[dict] = [
    # ---- free narrative, Hindi
    dict(
        id="widow_narrative_hi",
        utterance="मेरे पति का पिछले साल देहांत हो गया। दो बच्चे स्कूल जाते हैं। "
                  "गाँव में रहते हैं, कच्चा घर है।",
        expect={"applicant_is_widow": True, "primary_earner_died": True,
                "years_since_earner_death": 1, "residence": "rural",
                "house_type": "kutcha", "has_school_age_child": True,
                "child_in_school": True},
        forbid=["annual_income", "bpl", "applicant_age"],
    ),
    dict(
        id="elderly_narrative_hi",
        utterance="मैं सत्तर साल का हूँ, अकेले रहता हूँ गाँव में। कोई कमाई नहीं है अब।",
        expect={"applicant_age": 70, "residence": "rural"},
        forbid=["applicant_is_widow", "bpl"],
    ),
    # ---- code-mixed, as people actually speak
    dict(
        id="hinglish_farmer",
        utterance="Do bigha zameen hai, gaon mein rehte hain. Ration card BPL wala hai.",
        expect={"owns_farmland": True, "farmland_hectares": 0.5,
                "residence": "rural", "bpl": True},
        forbid=["annual_income", "applicant_age"],
    ),
    dict(
        id="english_urban",
        utterance="I live in Prayagraj city, pucca house, I earn about 18000 a month.",
        expect={"residence": "urban", "house_type": "pucca", "monthly_income": 18000},
        forbid=["bpl", "owns_farmland"],
    ),
    # ---- regional number words
    dict(
        id="lakh_income",
        utterance="Saal bhar mein do lakh ke aas paas kama lete hain.",
        expect={"annual_income": 200000},
        forbid=["monthly_income_stated"],   # derived, not stated - see note below
    ),
    dict(
        id="hazaar_income_hi",
        utterance="महीने में बीस हज़ार मिल जाते हैं।",
        expect={"monthly_income": 20000},
        forbid=["annual_income_stated"],
    ),
    # ---- answers to a specific question
    dict(
        id="answer_yes_bpl",
        question="क्या आपके पास बीपीएल या अंत्योदय राशन कार्ड है?",
        expect_fact="bpl",
        utterance="हाँ, अंत्योदय कार्ड है।",
        expect={"bpl": True},
        forbid=[],
    ),
    dict(
        id="answer_no_vehicle",
        question="क्या घर में कोई मोटर वाहन है?",
        expect_fact="owns_motor_vehicle",
        utterance="नहीं, कोई गाड़ी नहीं है।",
        expect={"owns_motor_vehicle": False},
        forbid=[],
    ),
    dict(
        id="answer_no_remarriage",
        question="क्या आपने दोबारा विवाह किया है?",
        expect_fact="remarried",
        utterance="नहीं, दोबारा शादी नहीं की।",
        expect={"remarried": False},
        forbid=[],
    ),
    # ---- the answer addresses something else entirely
    dict(
        id="mismatched_answer",
        question="क्या घर में कोई मोटर वाहन है?",
        expect_fact="owns_motor_vehicle",
        utterance="पैंतीस साल।",
        expect={},
        forbid=["owns_motor_vehicle"],
        why="People answer the previous question, or the one they expected. Guessing "
            "here writes a fact nobody stated into an entitlement decision.",
    ),
    dict(
        id="answer_with_extra",
        question="आपकी उम्र क्या है?",
        expect_fact="applicant_age",
        utterance="अड़तालीस की हूँ, और दो बेटियाँ हैं।",
        expect={"applicant_age": 48, "has_girl_child": True, "girl_children_count": 2},
        forbid=[],
    ),
    # ---- things that must not become facts
    dict(
        id="hedged_income",
        utterance="Pata nahi kitna kamate hain, kabhi kuch kabhi kuch.",
        expect={},
        forbid=["annual_income", "monthly_income"],
        why="An explicit 'I don't know' is not a number. Filling one in here silently "
            "decides eligibility on a value the person never gave.",
    ),
    dict(
        id="bereavement_without_earner_claim",
        utterance="मेरी सास का देहांत हो गया था।",
        expect={},
        forbid=["applicant_is_widow", "primary_earner_died"],
        why="A death in the family is not the applicant's widowhood, and not "
            "necessarily the household's earner.",
    ),
]

# Derived fields are computed in code from a stated one, so a case that states a monthly
# figure will legitimately also carry the annual figure. `forbid` entries ending in
# `_stated` mark that distinction for the reader; the scorer ignores them.
DERIVED_SUFFIX = "_stated"
