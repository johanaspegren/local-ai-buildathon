"""
Second, independent pathway for burn assessment: instead of the TFLite
classifier + NHS-grounded LLM pipeline in infer_pi.py, this script shows the
photo directly to a local multimodal medical LLM (MedGemma, served by
Ollama) and asks it -- unaided -- to classify the burn AND recommend a
treatment, with no other input: no TFLite prediction, no NHS guidance text,
no system prompt beyond the bare task. The point is to see what the model
can do purely from its own training, so this script deliberately does NOT
reuse nhs_guidance.py/treatment_recommender.py.

MedGemma (https://ollama.com/library/medgemma) is a Gemma 3 variant whose
vision encoder is pretrained on de-identified medical images, including
dermatology images specifically -- unlike a general-purpose vision model,
so it's a meaningful comparison point against pathway 1.

Setup (on the Raspberry Pi):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull medgemma:4b   # ~3.3 GB -- see README.md for a hardware caveat
                              # on a 4GB Pi, this is tight; test before relying on it

Usage:
    python3 infer_medgemma.py path/to/photo.jpg
    python3 infer_medgemma.py path/to/photo.jpg --model medgemma:4b --host http://localhost:11434

Only needs the Python standard library (urllib, base64) -- no extra pip
dependency, and no TensorFlow/TFLite runtime at all.

NOT a diagnostic or clinical decision-support tool. This pathway has even
fewer safety rails than pathway 1: there is no forced red-flag checklist and
no human-authored NHS grounding, only whatever MedGemma itself outputs, so
treat its answer with more scepticism, not less.
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import URLError

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "medgemma:4b"

PROMPT = (
    "You are shown a photo of a burn injury. "
    "1) Classify the burn's severity/degree (e.g. first/second/third degree, "
    "or superficial/partial-thickness/full-thickness -- say which system you're using). "
    "2) Recommend an appropriate treatment procedure for this specific burn. "
    "Be concise and clearly state your confidence/uncertainty."
)


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def get_classification_and_recommendation(image_path: Path, model: str = DEFAULT_MODEL,
                                           host: str = DEFAULT_OLLAMA_HOST,
                                           timeout: float = 120.0) -> str:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [encode_image(image_path)],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    req = urllib_request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body.get("response", "").strip()
    if not text:
        raise ValueError("empty response from Ollama")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the burn photo")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help=f"Ollama model tag to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST,
                         help=f"Ollama API base URL (default: {DEFAULT_OLLAMA_HOST})")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    print(f"Sending {image_path.name} to {args.model} via Ollama at {args.host} ...")
    print("(No classifier output, no NHS guidance, no other context is being given to the model.)\n")

    try:
        result = get_classification_and_recommendation(image_path, model=args.model, host=args.host)
    except (URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as e:
        sys.exit(
            f"Could not get a response from Ollama ({e}).\n"
            f"Make sure Ollama is running and the model is pulled:\n"
            f"  ollama pull {args.model}"
        )

    print("=" * 60)
    print(f"MedGemma raw output ({args.model}, unaided -- image only)")
    print("=" * 60)
    print(result)
    print()
    print(
        "Reminder: this is an experimental, unvalidated model output with no "
        "human-authored safety checklist behind it (unlike infer_pi.py's "
        "NHS-grounded pathway). Not a medical diagnosis -- contact NHS 111 or "
        "a healthcare professional for any real burn."
    )


if __name__ == "__main__":
    main()
