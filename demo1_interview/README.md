# Demo 1: Patient interview -> structured journal

A minimal, readable example of a four-step pipeline: transcribe, translate,
extract, format. This is the starting point for the buildathon - the goal
is to show *how* to build this, not to be a polished product.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
ollama pull gemma4:e2b   # if not already on this machine
```

`MODEL` in `main.py` is currently set to `gemma4:e2b`, since `gemma1.5`
(MedGemma) wasn't pullable on this machine. On the Raspberry Pi, where
`gemma1.5:latest` is available, change `MODEL` back to `"gemma1.5:latest"`
to use the medical-tuned model instead.

The first run downloads a `faster-whisper` speech-to-text model
(`WHISPER_MODEL_SIZE = "small"`) from Hugging Face - this needs internet
access once, then it's cached locally.

## Run

```bash
./.venv/bin/python3 main.py
```

## How it works

0. **`transcribe_audio`** - turns spoken audio into text using
   `faster-whisper`, a lighter offline-friendly alternative to the
   original OpenAI Whisper (no torch dependency, which matters on a
   Raspberry Pi). By default it auto-detects the spoken language - that's
   the multilingual part of this demo. Skip this step entirely and call
   `run_pipeline(text)` directly if you already have typed text.
1. **`translate_to_english`** - one Ollama call, one job: translate the
   patient's own words. Kept separate so the original and the translation
   can always be shown side by side (translation can lose nuance).
2. **`extract_structured_info`** - a second Ollama call that reads the
   English text and fills in a fixed set of fields (symptoms, onset,
   medications, etc.). Notice the system prompt explicitly says *do not
   diagnose* - the model organises what the patient said, it does not
   decide what's wrong with them. `format="json"` tells Ollama to
   constrain its output to valid JSON, so you don't need to worry about
   parsing prose.
3. **`generate_draft_note`** - plain Python, not an LLM call. Once the
   data is structured, turning it into a readable note is deterministic -
   a template is simpler and more reliable than asking a model to do it
   again. A good rule of thumb: only call the LLM for the part of the task
   that actually needs language understanding.

## Two real quirks you'll hit (and why they matter)

- **`format="json"` constrains valid JSON, not the exact shape.** In
  testing, `chief_complaint` (documented as a string) sometimes came back
  as a list instead. `_as_text()` coerces defensively rather than trying
  to prompt this away entirely - worth knowing generally when working with
  structured LLM output, not just here.
- **Whisper's language auto-detection can be unreliable on short clips,**
  even with real, clear speech. `audio/swahili-headache-fever.m4a` is a
  genuine recording, but auto-detect still guessed the wrong language at
  low confidence (a single short sentence doesn't give it much to work
  with) - so `language="sw"` is forced in `run_pipeline_from_audio` for
  the bundled sample. Try it on a longer recording with the `language`
  argument dropped to see auto-detect actually work.
- **`WHISPER_MODEL_SIZE = "small"` has real accuracy limits on Swahili.**
  In testing, it transcribed "na homa kali" (and severe fever) as "naho
  makali", which silently dropped "fever" from every step downstream -
  translation, extraction, and the final note all looked clean and
  confident, with nothing signaling that the input had been misheard.
  This is worth treating as a general lesson: a fluent, well-formatted
  answer is not the same as an accurate one, and STT errors here are
  invisible unless you have the original audio to check against. Trying
  a larger Whisper model size (e.g. `"medium"`) is one lever if accuracy
  matters more than speed for your use case.

## Things to try next (exercises)

- Record a longer, clearer Swahili sentence and run the pipeline with no
  `language` argument - see whether auto-detection does better with more
  speech to work with.
- Try `WHISPER_MODEL_SIZE = "medium"` on the same recording and compare
  the transcription against `"small"` - does "na homa kali" transcribe
  correctly this time?
- Add a second sample input in English text (skip audio, use
  `run_pipeline(text)` directly) and confirm the translation step is a
  no-op.
- Print how long each step takes (`time.time()` before/after each call).
  On a Raspberry Pi 5 this will be the first place you feel the model's
  real-world speed.
- Change one field in the system prompt's schema and see how it affects
  the JSON output.
- Try feeding it a sentence with no symptoms at all - what does
  `missing_information` look like?
