"""
Demo 3: Medical image description.

Pipeline (two steps, same pattern as demo 1):
  1. Image -> structured observations (one vision-capable Ollama call)
  2. Structured JSON -> human-readable summary (plain Python, no LLM)

The model's job is limited on purpose: describe what's visible, flag
image-quality problems, and say what cannot be determined from a photo
alone. It does not diagnose - see extract_observations() below.

Run: python main.py
"""

import json
import ollama

# gemma1.5 (MedGemma) is vision-capable and is the target model for the
# Raspberry Pi. On a dev machine where it isn't pulled, swap in any other
# vision-capable model you have (e.g. "qwen3-vl:2b") to test the pipeline.
MODEL = "qwen3-vl:2b"

IMAGE_PATH = "images/sample_ct_image.png"


def extract_observations(image_path: str) -> dict:
    """Step 1: describe the image into a fixed set of structured fields.

    The system prompt explicitly limits the model to observation and
    quality assessment - no diagnosis, no guessing at what can't be seen.
    format="json" constrains the output to valid JSON.
    """
    system_prompt = """You are a medical image triage assistant.
Describe the image using the JSON fields below. Do not diagnose or guess
what condition is shown - only describe what is visible and what is not.

Fields:
- image_type: string, what kind of image this appears to be
- visible_observations: list of strings, neutral descriptions of what is visible
- image_quality: object with "adequate" (bool) and "issues" (list of strings,
  e.g. poor lighting, out of focus, no scale reference)
- cannot_determine: list of strings, clinically relevant things a photo
  alone cannot show (e.g. depth, temperature, tenderness, duration)
- recommended_next_step: string, always defer to a trained clinician

Respond with JSON only."""

    # "format=json" is a strong hint, not a hard guarantee - some models
    # occasionally return malformed or truncated JSON. Retrying a couple
    # of times is a common, simple way to handle that instead of crashing
    # on the first bad response.
    attempts = 3
    for attempt in range(1, attempts + 1):
        response = ollama.chat(
            model=MODEL,
            format="json",
            # "Thinking" models (e.g. qwen3-vl) write a long internal
            # reasoning trace before the actual JSON answer, so this call
            # needs a generous token budget to reach that answer at all.
            options={"num_predict": 4096},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Describe this image.",
                    "images": [image_path],
                },
            ],
        )
        try:
            return json.loads(response["message"]["content"])
        except json.JSONDecodeError:
            if attempt == attempts:
                raise
            print(f"(got malformed JSON, retrying... attempt {attempt}/{attempts})")


def generate_summary(info: dict) -> str:
    """Step 2: turn the structured JSON into a readable summary.

    Plain Python again, same reasoning as demo 1: formatting known data
    doesn't need another model call.
    """
    lines = []
    lines.append(f"Image type: {info.get('image_type') or 'Not stated'}")

    observations = info.get("visible_observations") or []
    if observations:
        lines.append("Visible observations: " + ", ".join(observations))

    quality = info.get("image_quality") or {}
    adequate = quality.get("adequate")
    lines.append(f"Image quality adequate: {adequate}")
    if quality.get("issues"):
        lines.append("Quality issues: " + ", ".join(quality["issues"]))

    if info.get("cannot_determine"):
        lines.append("Cannot determine from this image: " + ", ".join(info["cannot_determine"]))

    lines.append(f"Recommended next step: {info.get('recommended_next_step') or 'Review by a trained clinician'}")

    return "\n".join(lines)


def run_pipeline(image_path: str) -> None:
    print("=" * 60)
    print(f"IMAGE: {image_path}")

    info = extract_observations(image_path)
    print("\nSTRUCTURED OBSERVATIONS (JSON)")
    print(json.dumps(info, indent=2))

    summary = generate_summary(info)
    print("\nSUMMARY FOR CLINICIAN REVIEW")
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline(IMAGE_PATH)
