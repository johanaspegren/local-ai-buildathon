"""
Turns a predicted burn-degree label into a first-aid/treatment recommendation
by grounding a small local LLM (served by Ollama) in paraphrased NHS
guidance (see nhs_guidance.py). Nothing leaves the device: Ollama runs
entirely locally, no cloud API key involved.

The recommendation is meant to genuinely differ by predicted degree, not
just in tone: nhs_guidance.BURN_PROFILES carries depth-specific description,
home care, recovery outlook, and escalation advice per degree (paraphrased
from a real NHS Trust patient leaflet, see nhs_guidance.py's module
docstring), while a smaller set of universal items (first aid, the
emergency-999 red flags, and when to call NHS 111) are always included in
full regardless of the prediction, because that's how NHS's own guidance
actually works -- those triggers are based on size/depth/location/cause, not
a "degree" label.

NOT a diagnostic or clinical decision-support tool. See CLASSIFIER_CAVEAT
in nhs_guidance.py and README.md.

Setup (on the Raspberry Pi):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2:1b
    ollama serve   # usually already running as a systemd service after install

Only needs the Python standard library (urllib) -- no extra pip dependency.
"""

import json
import textwrap
from urllib import request as urllib_request
from urllib.error import URLError

import nhs_guidance as nhs

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a cautious first-aid information assistant embedded in a hobby
    Raspberry Pi project. You ONLY use the NHS guidance text you are given
    in the user message -- never invent medical facts, dosages, or claims
    beyond it.

    Always structure your reply with these exact section headings:
    1. What this burn type usually means
    2. Immediate first aid
    3. Seek urgent medical help now if
    4. Home care and recovery outlook
    5. Important note

    Rules:
    - Section 1 must use the depth-specific description and recovery outlook
      given for the predicted degree -- do not write the same thing for every
      degree. A full-thickness prediction should read as clearly more urgent
      than a superficial one; do not soften it.
    - Section 3 must always include ALL of the emergency red-flag criteria
      provided, regardless of the predicted degree -- the predicted degree
      is unreliable and cannot see the burn's real size, exact location, or
      cause.
    - Section 4 must use the depth-specific home care/recovery text given,
      not generic advice.
    - Section 5 must state this is not a medical diagnosis, the prediction
      comes from an experimental, unvalidated image classifier, and the
      person should contact NHS 111 or a healthcare professional if they
      have any doubt.
    - Keep the whole answer under 220 words, plain language, no invented
      statistics or claims not present in the supplied guidance.
""")


def _bullets(items) -> str:
    return "\n".join(f"  - {item}" for item in items)


def _get_profile(predicted_label: str) -> dict:
    return nhs.BURN_PROFILES.get(predicted_label, nhs.BURN_PROFILES["degree_2"])


def _build_user_prompt(predicted_label: str, confidence: float) -> str:
    profile = _get_profile(predicted_label)
    return textwrap.dedent(f"""\
        Predicted classification: {predicted_label} (model confidence: {confidence:.0%})
        Conventionally shorthand for: {profile['clinical_name']}

        Classifier caveat to weave into your answer:
        {nhs.CLASSIFIER_CAVEAT}

        NHS guidance to base your answer on (sources: {nhs.sources_citation()}):

        What this burn type usually means:
        {profile['description']}

        Immediate first aid (universal, do this regardless of degree):
        {_bullets(nhs.IMMEDIATE_FIRST_AID)}

        Emergency red flags (call 999 / go to A&E) -- universal, always include ALL of these:
        {_bullets(nhs.EMERGENCY_RED_FLAGS)}

        When to call NHS 111 instead (universal):
        {_bullets(nhs.WHEN_TO_SEEK_111)}

        Home care - do (specific to this predicted degree):
        {_bullets(profile['home_care_dos'])}

        Home care - don't:
        {_bullets(profile['home_care_donts'])}

        Recovery outlook for this predicted degree:
        {profile['recovery_outlook']}

        When to get THIS predicted degree checked professionally:
        {profile['escalation_note']}

        Possible professional/hospital treatment, if needed:
        {_bullets(nhs.HOSPITAL_CARE_OVERVIEW)}

        Write the structured reply now, and make sure section 1 and section 4
        actually reflect this specific predicted degree rather than generic
        burn advice.
    """)


def _fallback_text(predicted_label: str, confidence: float, error: Exception) -> str:
    """Deterministic, LLM-free output used if Ollama can't be reached -- the
    tool should never fail to show basic safety info just because the local
    LLM service is down. Unlike the LLM path this can't paraphrase, but it
    still genuinely differs by predicted degree since it's built from
    BURN_PROFILES rather than a single shared block of text."""
    profile = _get_profile(predicted_label)
    lines = [
        f"[Local LLM unavailable ({error}) -- showing structured NHS guidance instead]",
        "",
        nhs.CLASSIFIER_CAVEAT,
        "",
        f"1. What this burn type usually means ({profile['clinical_name']})",
        profile["description"],
        f"Recovery outlook: {profile['recovery_outlook']}",
        "",
        "2. Immediate first aid",
        _bullets(nhs.IMMEDIATE_FIRST_AID),
        "",
        "3. Seek urgent medical help now if the burn:",
        _bullets(nhs.EMERGENCY_RED_FLAGS),
        "Also call NHS 111 if:",
        _bullets(nhs.WHEN_TO_SEEK_111),
        "",
        "4. Home care and recovery, specific to this predicted degree",
        "Do:",
        _bullets(profile["home_care_dos"]),
        "Don't:",
        _bullets(profile["home_care_donts"]),
        f"When to get it checked: {profile['escalation_note']}",
        "",
        "5. Important note",
        "This is not a medical diagnosis. The predicted label comes from an "
        "experimental, unvalidated image classifier. If in doubt, contact "
        "NHS 111 or a healthcare professional.",
        "",
        f"Sources: {nhs.sources_citation()}.",
    ]
    return "\n".join(lines)


def get_recommendation(predicted_label: str, confidence: float,
                        model: str = DEFAULT_MODEL, host: str = DEFAULT_OLLAMA_HOST,
                        timeout: float = 60.0) -> str:
    """Returns a treatment recommendation string, generated by the local
    Ollama model grounded in NHS guidance. Falls back to structured NHS
    guidance (no LLM) if Ollama isn't reachable or returns something unusable.
    """
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": _build_user_prompt(predicted_label, confidence),
        "stream": False,
        "options": {"temperature": 0.2},
    }
    req = urllib_request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "").strip()
        if not text:
            raise ValueError("empty response from Ollama")
        return text
    except (URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as e:
        return _fallback_text(predicted_label, confidence, e)
