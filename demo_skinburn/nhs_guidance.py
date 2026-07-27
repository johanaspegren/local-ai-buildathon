"""
Burn first-aid guidance, paraphrased (not copied verbatim) from the NHS
"Burns and scalds" patient page, for use as grounding context for the local
LLM in treatment_recommender.py.

Source: NHS - Burns and scalds: Treatment
        https://www.nhs.uk/conditions/burns-and-scalds/treatment/
        (page last reviewed 31 March 2026)

IMPORTANT: the NHS page does NOT classify burns by "degree" (1st/2nd/3rd).
It bases emergency escalation on things a photo can't reliably capture:
size, depth, exact location, and cause. DEGREE_NOTES below is our own
rough, non-clinical mapping to give the LLM *some* context for the
classifier's output -- it is deliberately paired with CLASSIFIER_CAVEAT and
the full EMERGENCY_RED_FLAGS list every time it's used, rather than being
treated as authoritative on its own.
"""

SOURCE_NAME = "NHS - Burns and scalds: Treatment"
SOURCE_URL = "https://www.nhs.uk/conditions/burns-and-scalds/treatment/"
LAST_REVIEWED = "31 March 2026"

IMMEDIATE_FIRST_AID = [
    "Cool the burn under cool (not ice-cold) running water for 15-30 minutes, "
    "or until it feels less painful. Use cool bottled water if no tap is available.",
    "Carefully remove nearby clothing or jewellery, but never pull away anything "
    "that is stuck to the burn itself.",
    "Once the burn has cooled, loosely lay (don't wrap) cling film or a clean "
    "plastic bag over it to protect it.",
]

EMERGENCY_RED_FLAGS = [
    "The burn is very large or appears deep.",
    "The burn is on the face, genitals, or bottom.",
    "The burn was caused by a chemical, acid, or electricity.",
]

WHEN_TO_SEEK_111 = [
    "You're not sure how serious the burn is.",
    "The burn is on a child under 5 years old.",
]

HOME_CARE_DOS = [
    "Take an over-the-counter painkiller such as paracetamol or ibuprofen for discomfort.",
    "Once it's healing, ask a pharmacist about a plain emollient if the skin becomes dry or itchy.",
]

HOME_CARE_DONTS = [
    "Don't put creams, oils, or butter on a fresh burn.",
    "Don't cover it with plasters or other sticky dressings.",
    "Don't burst any blisters that form.",
]

HOSPITAL_CARE_OVERVIEW = [
    "Pain relief and/or antibiotics.",
    "Professional wound cleaning and dressing.",
    "Fluids given through a vein (IV), if needed.",
    "Surgery to repair or reconstruct the skin, in serious cases.",
]

CLASSIFIER_CAVEAT = (
    "This prediction comes from an experimental, unvalidated image-classification "
    "model (not a certified medical device) trained on a small public dataset. "
    "It cannot see the burn's real size, depth, exact location, or cause -- all "
    "of which matter more than this label for judging how urgent the situation "
    "is. Always personally check the emergency red-flag criteria yourself, "
    "regardless of what the model predicted."
)

DEGREE_NOTES = {
    "degree_1": (
        "This label is sometimes used for the mildest, most superficial burns. "
        "Usually manageable with home first aid, PROVIDED none of the emergency "
        "red flags apply."
    ),
    "degree_2": (
        "This label is sometimes used for a moderate, partial-thickness burn, "
        "which may blister. Alongside first aid, it's sensible to get it "
        "checked by NHS 111 or a GP, especially if it's more than a couple of "
        "centimetres across."
    ),
    "degree_3": (
        "This label is sometimes used for the most severe, full-thickness "
        "burns. Treat this prediction as a prompt to seek urgent professional "
        "medical care, not just home first aid."
    ),
}
