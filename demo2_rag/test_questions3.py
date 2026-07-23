"""
DEMO 3 test questions -- same 8-question, 3-trap pattern as test_questions.py,
but sourced from data/who_pph_preeclampsia_excerpt.pdf (WHO "Recommendations
on maternal health," 2nd ed., 2023 -- Part B, Section 2.1 Haemorrhage and
2.2.1 Hypertensive Disorders/pre-eclampsia & eclampsia).

Trap boundaries were found by actually running naive_fixed_size_chunks() on
the extracted excerpt at chunk_size=500 and inspecting the output (see
chunking.py) -- not guessed.
"""

QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is the first-line uterotonic drug recommended for the treatment of postpartum haemorrhage (PPH)?",
        "expected_keywords": ["intravenous oxytocin"],
        "trap": False,
    },
    {
        "id": "Q2",
        "question": "If IV oxytocin is unavailable or bleeding doesn't respond to it, what fixed-dose combination drug does WHO recommend for treating PPH?",
        "expected_keywords": ["oxytocin-ergometrine"],
        "trap": True,
        "note": "The compound drug name 'oxytocin-ergometrine' is split mid-word across a naive chunk boundary ('...oxytoc' | 'in-ergometrine...'), so it never appears intact in any one chunk.",
    },
    {
        "id": "Q3",
        "question": "What dose of sublingual misoprostol is used when treating PPH?",
        "expected_keywords": ["800"],
        "trap": False,
    },
    {
        "id": "Q4",
        "question": "Under what condition is tranexamic acid recommended for the treatment of PPH?",
        "expected_keywords": ["uterotonics fail"],
        "trap": True,
        "note": "'uterotonics' is split mid-word across a chunk boundary ('...other u' | 'terotonics fail...'), so the phrase 'uterotonics fail' never appears intact in any one chunk.",
    },
    {
        "id": "Q5",
        "question": "List the temporizing measures recommended for PPH due to uterine atony when uterotonics have failed or are unavailable.",
        "expected_keywords": [
            "balloon tamponade",
            "bimanual uterine compression",
            "external aortic compression",
            "non-pneumatic anti-shock garments",
        ],
        "trap": True,
        "note": "These 4 measures span 3 separate naive chunks; top-k retrieval only surfaces 2 of the 3 chunks, silently dropping at least one measure.",
    },
    {
        "id": "Q6",
        "question": "What dose of low-dose aspirin is recommended for preventing pre-eclampsia in high-risk women, and by when should it be started?",
        "expected_keywords": ["75 mg", "20 weeks"],
        "trap": False,
    },
    {
        "id": "Q7",
        "question": "What drug is preferred over other anticonvulsants for the prevention and treatment of eclampsia?",
        "expected_keywords": ["magnesium sulfate"],
        "trap": False,
    },
    {
        "id": "Q8",
        "question": "In women with severe pre-eclampsia at term, what delivery policy does WHO recommend?",
        "expected_keywords": ["early delivery"],
        "trap": False,
    },
]
