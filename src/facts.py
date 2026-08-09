"""The fact vocabulary the model is allowed to emit.

The perception step turns free speech into these fields and nothing else.  Two reasons
the vocabulary is closed rather than open:

* A rule in schemes.yaml may only reference a name declared here, so a typo in either
  file is a startup error rather than a silently unmatched scheme.
* The model cannot invent a fact that no rule reads.  It fills in a form; it does not
  get to introduce new grounds for entitlement.

Every field is tri-state - True, False, or unknown (None) - because "she did not say"
and "she said no" have to be different things.  Treating silence as False is how a
system quietly denies people benefits they qualify for.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

Residence = Literal["rural", "urban"]
HouseType = Literal["none", "kutcha", "semi_pucca", "pucca"]


@dataclass
class Facts:
    """A household situation, as far as it is known."""

    # who is asking
    applicant_age: int | None = None
    applicant_is_widow: bool | None = None
    remarried: bool | None = None
    residence: Residence | None = None

    # household
    monthly_income: float | None = None
    annual_income: float | None = None
    bpl: bool | None = None
    house_type: HouseType | None = None
    owns_motor_vehicle: bool | None = None

    # children
    has_girl_child: bool | None = None
    girl_children_count: int | None = None
    has_school_age_child: bool | None = None
    child_in_school: bool | None = None

    # land and work
    owns_farmland: bool | None = None
    farmland_hectares: float | None = None
    is_income_tax_payer: bool | None = None
    holds_government_post: bool | None = None

    # bereavement
    primary_earner_died: bool | None = None
    years_since_earner_death: float | None = None
    deceased_age_at_death: int | None = None

    def known(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) is not None}

    def unknown(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is None]

    def with_(self, name: str, value: Any) -> "Facts":
        if name not in FACT_NAMES:
            raise KeyError(f"unknown fact: {name}")
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d[name] = value
        return Facts(**d)


FACT_NAMES = {f.name for f in fields(Facts)}

# What to ask a person, in their language, to establish each fact.  Kept next to the
# vocabulary so a new fact cannot be added without deciding how to ask about it.
QUESTIONS: dict[str, dict[str, str]] = {
    "applicant_age": {"en": "How old are you?", "hi": "आपकी उम्र क्या है?"},
    "applicant_is_widow": {"en": "Has your husband passed away?",
                           "hi": "क्या आपके पति का देहांत हो चुका है?"},
    "remarried": {"en": "Have you remarried since?", "hi": "क्या आपने दोबारा विवाह किया है?"},
    "residence": {"en": "Do you live in a village or a town?",
                  "hi": "आप गाँव में रहती हैं या शहर में?"},
    "monthly_income": {"en": "Roughly how much does the household earn in a month?",
                       "hi": "घर में महीने भर में लगभग कितनी आमदनी होती है?"},
    "annual_income": {"en": "Roughly how much does the household earn in a year?",
                      "hi": "घर की सालाना आमदनी लगभग कितनी है?"},
    "bpl": {"en": "Do you have a BPL or Antyodaya ration card?",
            "hi": "क्या आपके पास बीपीएल या अंत्योदय राशन कार्ड है?"},
    "house_type": {"en": "Is your house pucca, semi-pucca, or kutcha?",
                   "hi": "आपका घर पक्का है, अर्ध-पक्का है या कच्चा?"},
    "owns_motor_vehicle": {"en": "Does the household own a motorised vehicle?",
                           "hi": "क्या घर में कोई मोटर वाहन है?"},
    "has_girl_child": {"en": "Do you have a daughter?", "hi": "क्या आपकी कोई बेटी है?"},
    "girl_children_count": {"en": "How many daughters do you have?",
                            "hi": "आपकी कितनी बेटियाँ हैं?"},
    "has_school_age_child": {"en": "Do you have a child of school age?",
                             "hi": "क्या आपका कोई बच्चा स्कूल जाने की उम्र का है?"},
    "child_in_school": {"en": "Is the child currently enrolled in school?",
                        "hi": "क्या बच्चा अभी स्कूल में पढ़ रहा है?"},
    "owns_farmland": {"en": "Is there farmland in the family's name?",
                      "hi": "क्या परिवार के नाम पर खेती की ज़मीन है?"},
    "farmland_hectares": {"en": "How much land, in bighas or hectares?",
                          "hi": "कितनी ज़मीन है, बीघा या हेक्टेयर में?"},
    "is_income_tax_payer": {"en": "Does anyone in the household pay income tax?",
                            "hi": "क्या घर में कोई आयकर भरता है?"},
    "holds_government_post": {"en": "Does anyone hold a government post or pension?",
                              "hi": "क्या घर में कोई सरकारी पद पर है या पेंशन पाता है?"},
    "primary_earner_died": {"en": "Has the main earner of the household died?",
                            "hi": "क्या घर के मुख्य कमाने वाले का देहांत हुआ है?"},
    "years_since_earner_death": {"en": "How long ago did that happen?",
                                 "hi": "यह कितने समय पहले हुआ था?"},
    "deceased_age_at_death": {"en": "How old were they when they passed away?",
                              "hi": "देहांत के समय उनकी उम्र क्या थी?"},
}
