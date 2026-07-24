"""
Demo 1: Multilingual patient interview -> structured journal note.

Pipeline (each step is one clear, separate function):
  0. Spoken audio (any language) -> text (speech-to-text, optional - skip
     this step if you already have typed text)
  1. Patient text (any language) -> English translation
  2. English text -> structured JSON (symptoms, onset, etc.)
  3. Structured JSON -> human-readable draft note (plain Python, no LLM)

The CLI is parameterized so the same code can be driven with different
inputs, languages, and Whisper model sizes without editing code:

  # Bundled sample (Swahili audio, language forced because clip is short)
  python main.py

  # Any audio file, letting Whisper auto-detect the language
  python main.py --audio path/to/clip.m4a

  # Skip STT entirely and feed typed text through steps 1-3
  python main.py --text "Nina maumivu ya kichwa tangu jana."

  # Trade STT speed for accuracy - "medium" catches the "naho makali"
  # bug that "small" produces on the bundled sample.
  python main.py --whisper-size medium

  # Side-by-side comparison of two Whisper sizes on the same audio.
  # The reproducible silent-STT-failure demo: watch "fever" appear (or
  # not) as the model grows.
  python main.py --compare small,medium
"""

import argparse
import json
import os
import time

import ollama
from faster_whisper import WhisperModel

# Two models, one per role.
#
# TRANSLATE_MODEL: a general (not medical-fine-tuned) multilingual model
# is what we actually want here. Empirically, a medical-tuned model
# given garbled or degraded input (bad STT, wrong language detected)
# tends to *confabulate plausibly-shaped medical content* rather than
# admit it doesn't understand - "Nina has ten years of language
# experience" instead of "I have pain from [unclear reference]". A
# general model fails loudly (obvious placeholders, literal
# transliterations); a medical model fails silently (invented but
# clinically-plausible scenarios). For a translation task, loud
# failure is safer.
#
# EXTRACT_MODEL: MedGemma earns its place here. The input is now clean
# English and the task is genuinely medical - taxonomies of symptoms,
# standard medication names, spotting red flags. This is what the
# medical fine-tuning is *for*.
#
# Override either from the CLI with --translate-model / --extract-model.
TRANSLATE_MODEL = "gemma4:e2b"
EXTRACT_MODEL = "medgemma1.5:latest"

# faster-whisper is a lighter, offline-friendly alternative to the original
# OpenAI Whisper - no torch required, which matters on a Raspberry Pi.
# "small" is a reasonable accuracy/speed tradeoff for a laptop; on a Pi 5
# you may want to try "base" or "tiny" first and see what speed you get.
# The bundled Swahili sample exposes an accuracy limit of "small" (it
# transcribes "na homa kali" as "naho makali", silently dropping "fever")
# that "medium" fixes - see --compare mode.
DEFAULT_WHISPER_SIZE = "small"

SAMPLE_AUDIO_PATH = "audio/I_have_a_headache_since_yesterday_I_also_have_a_fever_and_I_feel_dizzy_when_I_stand_up.m4a"

# Example patient statement in Swahili, as typed text - useful for testing
# steps 1-3 without needing an audio file at all.
SAMPLE_INPUT_TEXT = "Nina maumivu ya kichwa tangu jana. Pia nina homa na ninahisi kizunguzungu ninaposimama."

# Loading a Whisper model is slow. Cache one instance per requested size
# so a --compare run doesn't re-load the same weights, and so a repeat
# question in a longer session doesn't pay the load cost twice.
_whisper_models: dict[str, WhisperModel] = {}


def get_whisper_model(size: str) -> WhisperModel:
    if size not in _whisper_models:
        print(f"(loading Whisper model '{size}'...)")
        t0 = time.time()
        _whisper_models[size] = WhisperModel(size)
        print(f"(loaded in {time.time() - t0:.1f}s)")
    return _whisper_models[size]


def transcribe_audio(audio_path: str, language: str | None = None,
                     whisper_size: str = DEFAULT_WHISPER_SIZE) -> tuple[str, dict]:
    """Step 0: turn spoken audio into text, in whatever language it's in.

    By default we don't tell Whisper which language to expect - it detects
    it - which is exactly the multilingual behaviour this demo wants. The
    translation to English still happens in the next step, so the
    patient's original words are always available too.

    Pass e.g. language="sw" to force a language instead of auto-detecting.
    This is mainly useful for troubleshooting: short single-sentence clips
    (like the bundled sample) can confuse Whisper's auto-detection even
    when the speech itself is real and clear.

    Returns (transcript, timings) so callers can log or display how long
    STT took separately from the rest of the pipeline.
    """
    model = get_whisper_model(whisper_size)
    t0 = time.time()
    segments, info = model.transcribe(audio_path, language=language)
    # faster-whisper returns a generator; joining forces evaluation.
    text = " ".join(segment.text.strip() for segment in segments)
    elapsed = time.time() - t0
    print(f"(detected language: {info.language}, confidence {info.language_probability:.2f})")
    timings = {
        "stt_s": elapsed,
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "whisper_size": whisper_size,
    }
    return text, timings


def translate_to_english(text: str, model: str = TRANSLATE_MODEL) -> tuple[str, dict]:
    """Step 1: translate the patient's own words into English.

    This is kept as its own call (rather than folded into extraction) so
    the clinician can always see the translation on its own, separate
    from anything the model infers in the next step.

    Uses a general (non-medical-tuned) model by default - see the
    TRANSLATE_MODEL comment for the reasoning.
    """
    t0 = time.time()
    response = ollama.chat(
        model=model,
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
    return response["message"]["content"].strip(), {"translate_s": time.time() - t0}


def extract_structured_info(english_text: str, model: str = EXTRACT_MODEL) -> tuple[dict, dict]:
    """Step 2: turn free text into a fixed set of structured fields.

    The prompt explicitly forbids diagnosis - the model's job is to
    organise what the patient said, not to decide what's wrong with them.
    We pass format="json" so Ollama constrains the output to valid JSON.

    Uses a medical-tuned model by default (medgemma1.5) - see the
    EXTRACT_MODEL comment for the reasoning.
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

    t0 = time.time()
    response = ollama.chat(
        model=model,
        format="json",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": english_text},
        ],
    )
    return json.loads(response["message"]["content"]), {"extract_s": time.time() - t0}


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


def run_pipeline(original_text: str, label: str | None = None,
                 translate_model: str = TRANSLATE_MODEL,
                 extract_model: str = EXTRACT_MODEL) -> dict:
    """Run steps 1-3 against already-transcribed (or typed) text.

    Returns a dict with the intermediate results and per-step timings, so
    callers (in particular --compare mode) can display or log more than
    just the final printed output.
    """
    heading = "PATIENT STATEMENT" if not label else f"PATIENT STATEMENT [{label}]"
    print("=" * 60)
    print(heading)
    print(original_text)

    english, t_translate = translate_to_english(original_text, model=translate_model)
    print(f"\nENGLISH TRANSLATION (via {translate_model})")
    print(english)

    info, t_extract = extract_structured_info(english, model=extract_model)
    print(f"\nSTRUCTURED JOURNAL (JSON) (via {extract_model})")
    print(json.dumps(info, indent=2))

    note = generate_draft_note(info)
    print("\nDRAFT NOTE FOR CLINICIAN REVIEW")
    print(note)

    print("\nTIMING")
    print(f"  translate: {t_translate['translate_s']:.1f}s")
    print(f"  extract:   {t_extract['extract_s']:.1f}s")
    print("=" * 60)

    return {
        "original": original_text,
        "english": english,
        "info": info,
        "note": note,
        "timings": {**t_translate, **t_extract},
    }


def run_pipeline_from_audio(audio_path: str, language: str | None = None,
                            whisper_size: str = DEFAULT_WHISPER_SIZE,
                            label: str | None = None,
                            translate_model: str = TRANSLATE_MODEL,
                            extract_model: str = EXTRACT_MODEL) -> dict:
    """Same pipeline as run_pipeline, starting from an audio file instead
    of typed text.
    """
    transcript, t_stt = transcribe_audio(audio_path, language=language, whisper_size=whisper_size)
    print(f"\n(STT with '{whisper_size}' took {t_stt['stt_s']:.1f}s)")
    result = run_pipeline(transcript, label=label,
                          translate_model=translate_model, extract_model=extract_model)
    result["timings"] = {**t_stt, **result["timings"]}
    return result


def run_compare(audio_path: str, sizes: list[str], language: str | None,
                translate_model: str = TRANSLATE_MODEL,
                extract_model: str = EXTRACT_MODEL) -> None:
    """Run the same audio through the pipeline once per Whisper size and
    print a compact diff summary at the end.

    This is the reproducible "silent STT failure" demo: the bundled
    Swahili sample transcribes as "naho makali" (nonsense) under
    whisper-size 'small', losing the word 'homa' (fever) from every
    downstream step - translation, extraction, and the draft note all
    look fluent and confident with the symptom silently missing. Under
    'medium' the same audio transcribes correctly and 'fever' surfaces.
    Same pipeline, same audio, one model-size knob, radically different
    clinical safety.
    """
    print(f"\n### Whisper size comparison on {audio_path}: {sizes}\n")
    results = {}
    for size in sizes:
        print(f"\n>>> Whisper size: {size}\n")
        results[size] = run_pipeline_from_audio(
            audio_path, language=language, whisper_size=size, label=f"whisper={size}",
            translate_model=translate_model, extract_model=extract_model,
        )

    # Compact summary: transcription + extracted symptoms per size.
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for size in sizes:
        r = results[size]
        symptoms = r["info"].get("symptoms") or []
        print(f"\n[{size}] transcript: {r['original']}")
        print(f"[{size}] symptoms:   {', '.join(symptoms) if symptoms else '(none extracted)'}")
        print(f"[{size}] STT time:   {r['timings']['stt_s']:.1f}s")
    print("=" * 60)


def run_transcribe_only(audio_paths: list[str], sizes: list[str], language: str | None) -> None:
    """Run STT and stop - no translation, no extraction, no LLM at all.

    Prints one section per audio file with one transcript per requested
    model size, plus a summary table at the end. Point at multiple
    audios and multiple sizes to build a quick matrix - great for
    workshop prep when you're deciding which recording + model
    combination to use live.
    """
    print(f"\n### Transcribe-only: {len(audio_paths)} audio(s) x {len(sizes)} model(s)\n")
    # results[audio_path][size] = (transcript, timing_dict)
    results: dict[str, dict[str, tuple[str, dict]]] = {}
    for audio_path in audio_paths:
        results[audio_path] = {}
        print("=" * 60)
        print(f"AUDIO: {audio_path}")
        print("=" * 60)
        for size in sizes:
            print(f"\n>>> {size}")
            transcript, t = transcribe_audio(audio_path, language=language, whisper_size=size)
            print(f"    transcript: {transcript}")
            print(f"    stt time:   {t['stt_s']:.1f}s")
            results[audio_path][size] = (transcript, t)

    # Compact matrix summary at the end - easy to eyeball across audios.
    print("\n" + "=" * 60)
    print("SUMMARY MATRIX")
    print("=" * 60)
    for audio_path in audio_paths:
        print(f"\n[{audio_path}]")
        for size in sizes:
            transcript, t = results[audio_path][size]
            print(f"  {size:<40} ({t['stt_s']:>5.1f}s)  {transcript}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", nargs="+",
                        help="One or more audio files (default: bundled Swahili sample). Multiple "
                             "paths are only meaningful with --transcribe-only.")
    source.add_argument("--text", help="Skip STT entirely and feed this typed text through steps 1-3")

    parser.add_argument("--language",
                        help="Force a language code for Whisper (e.g. 'sw', 'en'). Default: auto-detect. "
                             "The bundled sample is short and Whisper's auto-detect is unreliable on short "
                             "clips, so the default run forces 'sw' when no --audio is given.")
    parser.add_argument("--whisper-size", default=DEFAULT_WHISPER_SIZE,
                        help=f"Whisper model size: tiny/base/small/medium/large (default: {DEFAULT_WHISPER_SIZE})")
    parser.add_argument("--compare",
                        help="Comma-separated Whisper sizes to compare on the same audio, e.g. 'small,medium'. "
                             "Runs the full pipeline once per size and prints a summary diff at the end.")
    parser.add_argument("--translate-model", default=TRANSLATE_MODEL,
                        help=f"Ollama model for the translate step (default: {TRANSLATE_MODEL}). "
                             "Try a general multilingual model here; medical-tuned models tend to "
                             "confabulate plausible clinical content on garbled input.")
    parser.add_argument("--extract-model", default=EXTRACT_MODEL,
                        help=f"Ollama model for the extract step (default: {EXTRACT_MODEL}). "
                             "The medical fine-tuning earns its place here - clean English input, "
                             "a genuinely medical task.")
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Run STT only - no translation, no extraction, no LLM at all. "
                             "Great for testing different Whisper models/paths against different "
                             "recordings without waiting on the rest of the pipeline. Combine with "
                             "--audio file1 file2 ... and --compare small,medium,path/to/custom "
                             "to build a full model x recording matrix.")
    return parser.parse_args()


def _is_bundled_sample(path: str) -> bool:
    """Path-equivalence check, so `--audio audio/foo.m4a`,
    `--audio ./audio/foo.m4a`, and `--audio $(pwd)/audio/foo.m4a` all
    resolve to the same "is this the bundled sample?" answer.
    os.path.samefile is the right tool but requires the file to exist;
    fall back to a normalised-path compare if it doesn't.
    """
    try:
        return os.path.samefile(path, SAMPLE_AUDIO_PATH)
    except OSError:
        return os.path.normpath(path) == os.path.normpath(SAMPLE_AUDIO_PATH)


def _resolve_language(audio: str, explicit_language: str | None) -> str | None:
    """If the user hasn't specified --language and we're using the bundled
    sample, force 'sw' - the clip is short enough that Whisper's language
    auto-detect is unreliable and often mis-detects Slovenian at low
    confidence. Any explicit --language wins; any non-bundled --audio
    falls back to auto-detect.
    """
    if explicit_language is not None:
        return explicit_language
    if _is_bundled_sample(audio):
        return "sw"
    return None


if __name__ == "__main__":
    args = parse_args()

    # --audio is nargs="+", so it's a list (or None). For the full-pipeline
    # modes we only expect one file; --transcribe-only accepts many.
    audio_list = args.audio or [SAMPLE_AUDIO_PATH]

    if args.text:
        run_pipeline(args.text,
                     translate_model=args.translate_model, extract_model=args.extract_model)
    elif args.transcribe_only:
        sizes = ([s.strip() for s in args.compare.split(",") if s.strip()]
                 if args.compare else [args.whisper_size])
        # Language is resolved per-audio (so the bundled sample still gets
        # its "force sw" default even when mixed with other files).
        # For multi-audio runs, an explicit --language applies to all of them.
        # Simplification: resolve once using the first audio - workshop-prep
        # tool, users can pass --language explicitly when it matters.
        language = _resolve_language(audio_list[0], args.language)
        run_transcribe_only(audio_list, sizes, language=language)
    elif args.compare:
        sizes = [s.strip() for s in args.compare.split(",") if s.strip()]
        if len(audio_list) > 1:
            print("Warning: --compare full pipeline uses only the first --audio; "
                  "use --transcribe-only for multi-audio matrices.")
        audio = audio_list[0]
        run_compare(audio, sizes, language=_resolve_language(audio, args.language),
                    translate_model=args.translate_model, extract_model=args.extract_model)
    else:
        if len(audio_list) > 1:
            print("Warning: full pipeline uses only the first --audio; "
                  "use --transcribe-only for multi-audio runs.")
        audio = audio_list[0]
        run_pipeline_from_audio(audio, language=_resolve_language(audio, args.language),
                                whisper_size=args.whisper_size,
                                translate_model=args.translate_model,
                                extract_model=args.extract_model)
