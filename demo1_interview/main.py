"""
Demo 1: Multilingual patient interview -> structured journal note.

Pipeline (each step is one clear, separate function):
  0. Spoken audio (any language) -> text (speech-to-text, optional - skip
     this step if you already have typed text)
  1. Patient text (any language) -> English translation
  2. English text -> structured JSON (symptoms, onset, etc.)
  3. Structured JSON -> human-readable draft note (plain Python, no LLM)

Run: python main.py
"""

import json
import ollama
from faster_whisper import WhisperModel

MODEL = "gemma4:e2b"

# faster-whisper is a lighter, offline-friendly alternative to the original
# OpenAI Whisper - no torch required, which matters on a Raspberry Pi.
# "small" is a reasonable accuracy/speed tradeoff for a laptop; on a Pi 5
# you may want to try "base" or "tiny" first and see what speed you get.
WHISPER_MODEL_SIZE = "small"

SAMPLE_AUDIO_PATH = "audio/sample_swahili.wav"

# Example patient statement in Swahili, as typed text - useful for testing
# steps 1-3 without needing an audio file at all.
SAMPLE_INPUTS = [
    "Nina maumivu ya kichwa tangu jana. Pia nina homa na ninahisi kizunguzungu ninaposimama.",
]

# The whisper model is loaded once and reused, since loading it is slow
# and running it several times per session is normal for this demo.
_whisper_model = None


def transcribe_audio(audio_path: str, language: str | None = None) -> str:
    """Step 0: turn spoken audio into text, in whatever language it's in.

    By default we don't tell Whisper which language to expect - it detects
    it - which is exactly the multilingual behaviour this demo wants. The
    translation to English still happens in the next step, so the
    patient's original words are always available too.

    Pass e.g. language="sw" to force a language instead of auto-detecting.
    This is mainly useful for troubleshooting: a synthetic/robotic test
    voice (like the espeak-ng sample in audio/) can confuse auto-detection
    even though a real speaker's voice detects reliably.
    """
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE)

    segments, info = _whisper_model.transcribe(audio_path, language=language)
    print(f"(detected language: {info.language}, confidence {info.language_probability:.2f})")
    return " ".join(segment.text.strip() for segment in segments)


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


def _as_text(value) -> str:
    """format="json" constrains valid JSON, but not the exact shape - a
    field documented as a string can still come back as a list. Coercing
    defensively here is simpler than trying to prompt this away entirely.
    """
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def generate_draft_note(info: dict) -> str:
    """Step 3: turn structured JSON into a readable draft note.

    This step is plain Python, not another LLM call - once the data is
    structured, formatting it is deterministic, and a template is
    simpler and more reliable than asking a model to do it again.
    """
    lines = []
    complaint = info.get("chief_complaint")
    lines.append(f"Presenting complaint: {_as_text(complaint) if complaint else 'Not stated'}")

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


def run_pipeline_from_audio(audio_path: str, language: str | None = None) -> None:
    """Same pipeline as run_pipeline, starting from an audio file instead
    of typed text."""
    original_text = transcribe_audio(audio_path, language=language)
    run_pipeline(original_text)


if __name__ == "__main__":
    # Starting from audio (speech-to-text is step 0). To test steps 1-3
    # on their own without any audio, use run_pipeline(text) with one of
    # the strings in SAMPLE_INPUTS instead.
    #
    # language="sw" is forced here because the bundled sample audio is a
    # synthetic (robotic) text-to-speech voice that confuses Whisper's
    # language auto-detection. With a real recording of a real speaker,
    # drop the language argument and let it auto-detect.
    run_pipeline_from_audio(SAMPLE_AUDIO_PATH, language="sw")
