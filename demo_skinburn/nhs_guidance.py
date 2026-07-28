"""
Burn first-aid guidance, paraphrased (not copied verbatim) from NHS sources,
for use as grounding context for the local LLM in treatment_recommender.py.

Sources:
  1. NHS - Burns and scalds: Treatment
     https://www.nhs.uk/conditions/burns-and-scalds/treatment/
     (page last reviewed 31 March 2026)
  2. NHS inform (NHS 24) - Burns and scalds
     https://www.nhsinform.scot/illnesses-and-conditions/injuries/skin-injuries/burns-and-scalds/
     (last updated 23 July 2026)
  3. University Hospitals Plymouth NHS Trust - Burns and Scalds patient
     information leaflet (ref A-537/NB/ED/Burns and Scalds v2)
     https://www.plymouthhospitals.nhs.uk/display-pil/pil-burns-and-scalds-6065/
     (issued February 2025, review due February 2027)

IMPORTANT: sources 1 and 2 do NOT classify burns by "degree" (1st/2nd/3rd) --
they base emergency escalation (999/A&E) on things a photo can't reliably
capture: size, depth, exact location, and cause. That escalation logic is
kept universal (EMERGENCY_RED_FLAGS / WHEN_TO_SEEK_111 below) rather than
being made to depend on the predicted degree.

Source 3, however, *does* describe the three depths that "degree_1/2/3" are
conventionally shorthand for (superficial / partial-thickness / full-
thickness), including how they look, feel, and heal differently -- that's
what BURN_PROFILES below is built from, so the generated recommendation
actually differs in substance (not just tone) by predicted degree, while
still always surfacing the universal red flags regardless of the prediction.
"""

SOURCES = [
    {
        "name": "NHS - Burns and scalds: Treatment",
        "url": "https://www.nhs.uk/conditions/burns-and-scalds/treatment/",
        "reviewed": "31 March 2026",
    },
    {
        "name": "NHS inform (NHS 24) - Burns and scalds",
        "url": "https://www.nhsinform.scot/illnesses-and-conditions/injuries/skin-injuries/burns-and-scalds/",
        "reviewed": "23 July 2026",
    },
    {
        "name": "University Hospitals Plymouth NHS Trust - Burns and Scalds patient information leaflet",
        "url": "https://www.plymouthhospitals.nhs.uk/display-pil/pil-burns-and-scalds-6065/",
        "reviewed": "February 2025 (review due February 2027)",
    },
]

# Kept for backwards compatibility / simple citation strings elsewhere.
SOURCE_NAME = SOURCES[0]["name"]
SOURCE_URL = SOURCES[0]["url"]
LAST_REVIEWED = SOURCES[0]["reviewed"]


def sources_citation() -> str:
    return "; ".join(f"{s['name']} ({s['url']}, {s['reviewed']})" for s in SOURCES)


# ---------------------------------------------------------------------------
# Universal guidance -- applies no matter what the classifier predicted.
# ---------------------------------------------------------------------------

IMMEDIATE_FIRST_AID = [
    "Remove any clothing or jewellery near the burn, but never pull away anything "
    "that is stuck to the burnt skin.",
    "Cool the burn under cool (not ice-cold) running water for 15-30 minutes, or "
    "until it feels less painful -- use cool bottled water if no tap is available.",
    "Keep the person warm while cooling the burn (a blanket over unaffected areas), "
    "especially for a large area, a young child, or an elderly person -- prolonged "
    "cooling of a large burn can cause hypothermia.",
    "Once the burn has cooled, loosely lay (don't wrap) cling film, or a clean "
    "plastic bag for a hand, over it to protect it until it can be properly assessed.",
]

EMERGENCY_RED_FLAGS = [
    "The burn is very large or appears deep.",
    "The burn is on the face, genitals, or bottom.",
    "The burn was caused by a chemical, acid, or electricity.",
    "The skin looks white, brown, black, or leathery (charred).",
    "There is little or no pain in the burnt area despite it looking serious -- "
    "this can happen when nerve endings are destroyed, and does NOT mean the "
    "burn is minor.",
]

WHEN_TO_SEEK_111 = [
    "You're not sure how serious the burn is.",
    "The burn is on a child under 5 years old.",
]

HOME_CARE_DONTS_UNIVERSAL = [
    "Don't put creams, oils, butter, or other greasy substances on a fresh burn.",
    "Don't use ice or iced water to cool it.",
    "Don't cover it with plasters or other sticky dressings.",
    "Don't burst any blisters that form -- a burst blister is more likely to get infected.",
]

INFECTION_WARNING_SIGNS = [
    "Increasing redness or warmth around the burn.",
    "Increasing pain.",
    "Oozing or discharge from the wound.",
]

CLASSIFIER_CAVEAT = (
    "This prediction comes from an experimental, unvalidated image-classification "
    "model (not a certified medical device) trained on a small public dataset. "
    "It cannot see the burn's real size, exact location, or cause -- all of which "
    "matter alongside depth for judging how urgent the situation is. Always "
    "personally check the emergency red-flag criteria yourself, regardless of "
    "what the model predicted."
)

# ---------------------------------------------------------------------------
# Depth-specific profiles -- this is what actually varies by predicted degree.
# "degree_1/2/3" are conventional shorthand for superficial / partial-thickness
# / full-thickness burns (source 3).
# ---------------------------------------------------------------------------

BURN_PROFILES = {
    "degree_1": {
        "clinical_name": "superficial burn",
        "description": (
            "Affects only the top layer of skin. Typically red and painful, but "
            "usually doesn't blister or scar -- similar to mild sunburn."
        ),
        "home_care_dos": [
            "Take paracetamol or ibuprofen for pain if needed.",
            "Once it's healing, ask a pharmacist about a plain, unperfumed "
            "emollient/moisturiser if the skin becomes dry or itchy.",
            "Protect the area from direct sun while it heals and for a while after.",
        ],
        "home_care_donts": HOME_CARE_DONTS_UNIVERSAL,
        "recovery_outlook": (
            "Usually heals in around 14 days with little to no scarring, "
            "provided none of the emergency red flags apply."
        ),
        "escalation_note": (
            "Get it checked by NHS 111 or a GP if it isn't improving after a few "
            "days, looks infected, or you're at all unsure."
        ),
    },
    "degree_2": {
        "clinical_name": "partial-thickness burn",
        "description": (
            "Deeper damage than a superficial burn. Usually forms painful "
            "blisters, though some of the deeper skin layer (the dermis) is "
            "typically still intact."
        ),
        "home_care_dos": [
            "Take paracetamol or ibuprofen for pain if needed.",
            "Leave blisters intact and keep the area clean and covered.",
            "Watch closely for signs of infection: "
            + "; ".join(s.rstrip(".").lower() for s in INFECTION_WARNING_SIGNS) + ".",
        ],
        "home_care_donts": HOME_CARE_DONTS_UNIVERSAL,
        "recovery_outlook": (
            "Can heal well, sometimes without scarring, if it isn't too deep or "
            "widespread -- but often benefits from a professional dressing check "
            "rather than being left to heal completely untreated."
        ),
        "escalation_note": (
            "It's sensible to get this checked by NHS 111 or a GP, especially if "
            "it's more than a couple of centimetres across, on a hand, joint, or "
            "sensitive area, or shows any infection warning signs above."
        ),
    },
    "degree_3": {
        "clinical_name": "full-thickness burn",
        "description": (
            "Damage extends through all layers of skin. There may be surprisingly "
            "little pain in the burnt area itself, because the nerve endings have "
            "been destroyed -- low pain does NOT mean this is minor, it can mean "
            "the opposite."
        ),
        "home_care_dos": [
            "Do the immediate first aid steps above, then treat this as a prompt "
            "to get urgent professional care -- this is not primarily a home-care situation.",
        ],
        "home_care_donts": HOME_CARE_DONTS_UNIVERSAL,
        "recovery_outlook": (
            "Often needs specialist assessment and sometimes skin-graft surgery; "
            "healing can take months or longer and usually leaves visible scarring."
        ),
        "escalation_note": (
            "Treat this prediction as a signal to seek urgent medical care now "
            "(NHS 111, a GP, or 999/A&E if any red flag applies) rather than "
            "relying on home first aid alone."
        ),
    },
}

HOSPITAL_CARE_OVERVIEW = [
    "A clinician will assess the size and depth of the burn.",
    "The wound will be cleaned and a suitable dressing applied (not all burns need one).",
    "Pain relief, and sometimes antibiotics or a tetanus check, may be given.",
    "Dressings are typically reviewed at 24 hours, then 48 hours, then every 3-5 "
    "days until healed.",
    "Fluids given through a vein (IV), and/or surgery such as a skin graft, for "
    "larger or deeper burns.",
]
