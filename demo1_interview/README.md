# Demo 1: Patient interview -> structured journal

A minimal, readable example of a three-step LLM pipeline: translate, extract,
format. This is the starting point for the buildathon - the goal is to show
*how* to build this, not to be a polished product.

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

## Run

```bash
./.venv/bin/python3 main.py
```

## How it works

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

## Things to try next (exercises)

- Add a second sample input in English and confirm the translation step
  is a no-op (or skip translation entirely when the text is already
  English - how would you detect that?).
- Print how long each step takes (`time.time()` before/after each call).
  On a Raspberry Pi 5 this will be the first place you feel the model's
  real-world speed.
- Change one field in the system prompt's schema and see how it affects
  the JSON output.
- Try feeding it a sentence with no symptoms at all - what does
  `missing_information` look like?
