"""
Test questions for the naive-vs-curated batch demo.

8 questions over the WHO PPH + pre-eclampsia excerpt
(documents/who_pph_preeclampsia_excerpt.pdf and its curated Q&A twin
documents/qa_who_pph_preeclampsia_excerpt.json). Two of them are traps:
their expected keyword is a WHO term that this codebase's per-page
500-char/no-overlap chunker actually splits, verified by running the
chunker and inspecting where the boundaries land. Naive should fail
those two; curation preserves the intact term in a single Q&A pair so
retrieval finds it every time.

If you change the chunker (concatenate pages, add overlap, switch to a
different chunk size), re-verify the traps - a "trap" is only a trap
against a specific chunking implementation. To re-verify, run
main.py --show-chunks and grep the output for the expected term.
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
        "question": "What specific device does WHO recommend as a temporizing measure for PPH due to uterine atony when uterotonics have failed?",
        "expected_keywords": ["intrauterine balloon tamponade"],
        "trap": True,
        "note": "'intrauterine balloon tamponade' is the WHO device name and never appears intact in any single 500-char chunk of the per-page-chunked PDF (the compound gets split at every occurrence). 'balloon tamponade' alone would survive; the full clinical term is what breaks. Curated Q&A #7 has it intact.",
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
        "trap": False,
    },
    {
        "id": "Q5",
        "question": "What technique is recommended alongside additional oxytocin if the placenta is not expelled spontaneously?",
        "expected_keywords": ["controlled cord traction"],
        "trap": True,
        "note": "'controlled cord traction' is split at a page-4 chunk boundary ('...c' | 'ontrolled cord traction...'), so the intact phrase never appears in any retrieved chunk. Curated Q&A #9 has it intact.",
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
