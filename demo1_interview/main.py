"""
Demo 1: Multilingual patient interview -> structured journal note.

Pipeline (each step is one clear, separate function):
  1. Patient text (any language) -> English translation
  2. English text -> structured JSON (symptoms, onset, etc.)
  3. Structured JSON -> human-readable draft note (plain Python, no LLM)

Run: python main.py
"""

import json
import ollama

MODEL = "gemma4:e2b"

# Example patient statement in Swahili - swap this for your own test input.
SAMPLE_INPUTS = [
    "Nina maumivu ya kichwa tangu jana. Pia nina homa na ninahisi kizunguzungu ninaposimama.",
]


def translate_to_english(text: str) -> str:
    """Step 1: translate the patient's own words into English.

    This is kept as its own call (rather than folded into extraction) so
    the clinician can always see the translation on its own, separate
    from anything the model infers in the next step.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical translator. Translate the patient's "
                    "message into clear, simple English. Output only the "
                    "translation, with no extra commentary."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return response["message"]["content"].strip()


def extract_structured_info(english_text: str) -> dict:
    """Step 2: turn free text into a fixed set of structured fields.

    The prompt explicitly forbids diagnosis - the model's job is to
    organise what the patient said, not to decide what's wrong with them.
    We pass format="json" so Ollama constrains the output to valid JSON.
    """
    system_prompt = """You are a clinical note-taking assistant.
Extract information from the patient's statement into the JSON fields below.
Use null for any field the patient did not mention. Do not guess or diagnose.

Fields:
- chief_complaint: string, the main reason for the visit
- symptoms: list of strings
- onset: string or null, when symptoms started
- severity: string or null
- medications: list of strings
- allergies: list of strings
- red_flags: list of strings, anything urgent the patient mentioned
- missing_information: list of strings, important clinical details the
  patient did NOT mention (e.g. temperature, vomiting, injury)

Respond with JSON only."""

    response = ollama.chat(
        model=MODEL,
        format="json",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": english_text},
        ],
    )
    return json.loads(response["message"]["content"])


def generate_draft_note(info: dict) -> str:
    """Step 3: turn structured JSON into a readable draft note.

    This step is plain Python, not another LLM call - once the data is
    structured, formatting it is deterministic, and a template is
    simpler and more reliable than asking a model to do it again.
    """
    lines = []
    lines.append(f"Presenting complaint: {info.get('chief_complaint') or 'Not stated'}")

    symptoms = info.get("symptoms") or []
    if symptoms:
        symptom_line = ", ".join(symptoms)
        if info.get("onset"):
            symptom_line += f" (onset: {info['onset']})"
        lines.append(f"Symptoms: {symptom_line}")

    if info.get("severity"):
        lines.append(f"Severity: {info['severity']}")

    if info.get("medications"):
        lines.append(f"Current medications: {', '.join(info['medications'])}")

    if info.get("allergies"):
        lines.append(f"Allergies: {', '.join(info['allergies'])}")

    if info.get("red_flags"):
        lines.append(f"Red flags noted: {', '.join(info['red_flags'])}")

    if info.get("missing_information"):
        lines.append("Information still required: " + ", ".join(info["missing_information"]))

    return "\n".join(lines)


def run_pipeline(original_text: str) -> None:
    print("=" * 60)
    print("ORIGINAL PATIENT STATEMENT")
    print(original_text)

    english = translate_to_english(original_text)
    print("\nENGLISH TRANSLATION")
    print(english)

    info = extract_structured_info(english)
    print("\nSTRUCTURED JOURNAL (JSON)")
    print(json.dumps(info, indent=2))

    note = generate_draft_note(info)
    print("\nDRAFT NOTE FOR CLINICIAN REVIEW")
    print(note)
    print("=" * 60)


if __name__ == "__main__":
    for text in SAMPLE_INPUTS:
        run_pipeline(text)
