"""Complete households, used to measure the interview rather than the extractor.

Each entry is a full fact set - what would be true if every question had been asked and
answered honestly - plus the schemes that ought to come out of it, worked out by hand
against schemes/schemes.yaml.

Two different things are measured with these:

* `tests/test_households.py` checks the engine reaches the hand-derived answer. That is
  end-to-end correctness of the part that decides entitlement, and it needs no model.
* `scripts_ask.py` uses each household as a truthful oracle: it answers whatever the
  engine asks, so the *order* the engine asks in can be measured against alternatives.

The households are deliberately not all easy. Two of them qualify for almost nothing,
because a tool that is only measured on people with a strong claim will look good and
waste the time of everyone else.
"""
from __future__ import annotations

HOUSEHOLDS: list[dict] = [
    dict(
        id="widow_rural_bpl",
        note="The walkthrough case: widowed last year, two daughters, kutcha house.",
        facts=dict(
            applicant_age=35, applicant_is_widow=True, remarried=False,
            residence="rural", monthly_income=3000, annual_income=36000, bpl=True,
            house_type="kutcha", owns_motor_vehicle=False,
            has_girl_child=True, girl_children_count=2,
            has_school_age_child=True, child_in_school=True,
            owns_farmland=False, farmland_hectares=0.0,
            is_income_tax_payer=False, holds_government_post=False,
            primary_earner_died=True, years_since_earner_death=1,
            deceased_age_at_death=40,
        ),
        expect_eligible=["pmay_g", "nfbs", "widow_pension", "kanya_sumangala",
                         "ayushman", "scholarship_pre_matric"],
    ),
    dict(
        id="elderly_alone_rural",
        note="70, lives alone in a village, no surviving earner, no children at home.",
        facts=dict(
            applicant_age=70, applicant_is_widow=True, remarried=False,
            residence="rural", monthly_income=1500, annual_income=18000, bpl=True,
            house_type="kutcha", owns_motor_vehicle=False,
            has_girl_child=False, girl_children_count=0,
            has_school_age_child=False, child_in_school=False,
            owns_farmland=False, farmland_hectares=0.0,
            is_income_tax_payer=False, holds_government_post=False,
            primary_earner_died=True, years_since_earner_death=9,
            deceased_age_at_death=72,
        ),
        expect_eligible=["pmay_g", "widow_pension", "old_age_pension", "ayushman"],
    ),
    dict(
        id="smallholder_family",
        note="Two bighas, pucca house, no bereavement. Land schemes, not welfare ones.",
        facts=dict(
            applicant_age=44, applicant_is_widow=False, remarried=False,
            residence="rural", monthly_income=9000, annual_income=108000, bpl=False,
            house_type="pucca", owns_motor_vehicle=True,
            has_girl_child=True, girl_children_count=1,
            has_school_age_child=True, child_in_school=True,
            owns_farmland=True, farmland_hectares=0.5,
            is_income_tax_payer=False, holds_government_post=False,
            primary_earner_died=False, years_since_earner_death=0,
            deceased_age_at_death=0,
        ),
        expect_eligible=["pm_kisan", "kanya_sumangala", "scholarship_pre_matric"],
    ),
    dict(
        id="urban_salaried",
        note="Qualifies for almost nothing. The tool has to say so quickly.",
        facts=dict(
            applicant_age=38, applicant_is_widow=False, remarried=False,
            residence="urban", monthly_income=45000, annual_income=540000, bpl=False,
            house_type="pucca", owns_motor_vehicle=True,
            has_girl_child=False, girl_children_count=0,
            has_school_age_child=False, child_in_school=False,
            owns_farmland=False, farmland_hectares=0.0,
            is_income_tax_payer=True, holds_government_post=False,
            primary_earner_died=False, years_since_earner_death=0,
            deceased_age_at_death=0,
        ),
        expect_eligible=[],
    ),
    dict(
        id="widow_remarried",
        note="Widowed, then remarried. The widow pension is correctly ruled out, and "
             "getting that wrong in the generous direction sends her to a counter that "
             "will check.",
        facts=dict(
            applicant_age=41, applicant_is_widow=True, remarried=True,
            residence="rural", monthly_income=4000, annual_income=48000, bpl=True,
            house_type="semi_pucca", owns_motor_vehicle=False,
            has_girl_child=True, girl_children_count=1,
            has_school_age_child=True, child_in_school=True,
            owns_farmland=False, farmland_hectares=0.0,
            is_income_tax_payer=False, holds_government_post=False,
            primary_earner_died=True, years_since_earner_death=6,
            deceased_age_at_death=48,
        ),
        expect_eligible=["kanya_sumangala", "ayushman", "scholarship_pre_matric"],
    ),
    dict(
        id="bereaved_recent_urban",
        note="Urban, earner died this year within the NFBS age window, poor. Housing is "
             "out because PMAY-G is rural.",
        facts=dict(
            applicant_age=33, applicant_is_widow=True, remarried=False,
            residence="urban", monthly_income=2500, annual_income=30000, bpl=True,
            house_type="kutcha", owns_motor_vehicle=False,
            has_girl_child=False, girl_children_count=0,
            has_school_age_child=False, child_in_school=False,
            owns_farmland=False, farmland_hectares=0.0,
            is_income_tax_payer=False, holds_government_post=False,
            primary_earner_died=True, years_since_earner_death=0.5,
            deceased_age_at_death=37,
        ),
        expect_eligible=["nfbs", "widow_pension", "ayushman"],
    ),
    dict(
        id="government_post_farmer",
        note="Has land but holds a government post, which excludes PM-KISAN. An "
             "exclusion that is easy to miss and expensive to get wrong.",
        facts=dict(
            applicant_age=50, applicant_is_widow=False, remarried=False,
            residence="rural", monthly_income=25000, annual_income=300000, bpl=False,
            house_type="pucca", owns_motor_vehicle=True,
            has_girl_child=False, girl_children_count=0,
            has_school_age_child=True, child_in_school=True,
            owns_farmland=True, farmland_hectares=1.2,
            is_income_tax_payer=False, holds_government_post=True,
            primary_earner_died=False, years_since_earner_death=0,
            deceased_age_at_death=0,
        ),
        expect_eligible=[],
    ),
    dict(
        id="landless_labourer",
        note="Landless, kutcha house, children in school, no bereavement.",
        facts=dict(
            applicant_age=29, applicant_is_widow=False, remarried=False,
            residence="rural", monthly_income=5000, annual_income=60000, bpl=True,
            house_type="kutcha", owns_motor_vehicle=False,
            has_girl_child=True, girl_children_count=2,
            has_school_age_child=True, child_in_school=True,
            owns_farmland=False, farmland_hectares=0.0,
            is_income_tax_payer=False, holds_government_post=False,
            primary_earner_died=False, years_since_earner_death=0,
            deceased_age_at_death=0,
        ),
        expect_eligible=["pmay_g", "kanya_sumangala", "ayushman",
                         "scholarship_pre_matric"],
    ),
]
