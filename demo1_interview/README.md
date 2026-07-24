# Demo 1: Patient interview -> structured journal

A minimal, readable four-step pipeline: transcribe, translate, extract,
format. Same house-style as demo 2 - single file, one function per
step, honest comments about the quirks. Starting point for the
buildathon, not a polished product.

## Setup

```bash
cd demo1_interview
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -r requirements.txt
```

### Chat models: one per job

This demo uses **two** Ollama models, each on the task it's actually
good at:

- **`TRANSLATE_MODEL = "gemma4:e2b"`** - a general (not medical-tuned)
  multilingual model for step 1 (Swahili -> English).
- **`EXTRACT_MODEL = "medgemma1.5:latest"`** - the medical-tuned model
  for step 2 (English -> structured JSON).

Install both:

```bash
ollama pull gemma4:e2b
# and see demo 2's README for medgemma1.5 (Ollama pull or HF fallback)
```

Override either from the CLI with `--translate-model <name>` and
`--extract-model <name>`.

**Why the split?** See "Why translate and extract use different models"
below - it's a genuine safety difference, not just an architectural
preference.

### Whisper (speech-to-text)

`faster-whisper` fetches models from Hugging Face on first use and
caches them under `~/.cache/huggingface/hub/`. Sizes go from `tiny`
(~40MB) to `large` (~3GB); we default to `small` (~500MB) and use
`medium` (~1.5GB) for the accuracy-comparison mode. Switch sizes on
the CLI with `--whisper-size`.

**Auto-fetch on first run.** Just run the pipeline with the size you
want; the model downloads before the first transcription. Fine if
you're on a decent connection.

**Manual pre-fetch.** If you'd rather control when the download
happens (limited bandwidth, going offline soon, or don't want the
first live demo to stall on a download bar), pull the models you'll
need up front:

```bash
# One size
./.venv/bin/python3 -c "from faster_whisper import WhisperModel; WhisperModel('small')"

# Multiple sizes back-to-back (for --compare)
./.venv/bin/python3 -c "from faster_whisper import WhisperModel; [WhisperModel(s) for s in ['small', 'medium']]"
```

Each call triggers the download if the model isn't cached yet, then
exits. Subsequent runs of `main.py` with the same size start straight
from the cached copy.

### Optional: Swahili-specific STT fine-tune

For the STT-only comparison section below, we use a Whisper-medium
fine-tune trained on Common Voice Swahili
(`keystats/kiswahili_sahihi_asr`). It ships in `safetensors` format so
you need to convert it to CTranslate2 once, after which `faster-whisper`
loads it via a directory path exactly like a built-in size name.

```bash
# One-time: install torch (only needed for the conversion, not runtime)
./.venv/bin/pip install torch ctranslate2 transformers

# Convert to CT2, quantized to int8
./.venv/bin/ct2-transformers-converter \
  --model keystats/kiswahili_sahihi_asr \
  --output_dir models/kiswahili-sahihi-ct2 \
  --quantization int8

# Sanity check
./.venv/bin/python3 -c "from faster_whisper import WhisperModel; WhisperModel('models/kiswahili-sahihi-ct2')"
```

After that, pass the directory anywhere a Whisper size is expected:
`--whisper-size models/kiswahili-sahihi-ct2` or as an entry in
`--compare small,medium,models/kiswahili-sahihi-ct2`.

## Run

```bash
# Bundled sample: Swahili audio, language forced (short clip)
./.venv/bin/python3 main.py

# Any audio file, letting Whisper auto-detect the language
./.venv/bin/python3 main.py --audio ./audio/some-other-clip.m4a

# Skip STT entirely - feed typed text through steps 1-3 only
./.venv/bin/python3 main.py --text "Nina maumivu ya kichwa tangu jana."

# Trade STT speed for accuracy on the same audio
./.venv/bin/python3 main.py --whisper-size medium
```

Every run prints per-step timings (STT / translate / extract) so you
can feel where time actually goes on your hardware. On a Pi 5, Whisper
STT usually dominates.

## The "silent STT failure" comparison

The single most important lesson in this demo: a fluent, well-formatted
answer is not the same as an accurate one. When speech-to-text drops
a word - or transcribes it as the wrong word, or breaks a compound
into fragments - every downstream step still looks confident and
clean, with nothing signalling that the input was misheard.

Run the same audio through the full pipeline twice, once per Whisper
size:

```bash
./.venv/bin/python3 main.py --compare small,medium
```

The bundled sample is a Google-TTS recording of *"Nina maumivu ya
kichwa tangu jana. Pia nina homa na ninahisi kizunguzungu
ninaposimama."* — "I have had a headache since yesterday. I also have
a fever and I feel dizzy when I stand up."

What actually happens (reproducible, seen multiple times during
workshop prep):

- **small** produces a transcript with no punctuation and no spaces
  between several words (`homana`, `hisiki zungu zungu`, `posimama`).
  The downstream LLM then translates it into fluent English that
  drops some real symptoms (fever, dizziness-on-standing) *and*
  invents new ones (sore throat, chills — different fabrications on
  different runs, all confident-sounding).
- **medium** gets spacing mostly right but slips on `jana` (yesterday)
  → `kana`, and still breaks the `kizunguzungu`/`ninaposimama`
  compounds. Fewer fabrications but the onset info is often lost.

Same audio, same pipeline, one model-size knob. The summary at the
bottom of the compare run shows both transcripts and both symptom
lists side by side — watch what appears, disappears, or shape-shifts
as the model grows.

What we learn? Well we get two clinical notes
generated from the same audio. Neither is entirely right. Neither is
entirely wrong. Nothing in the output tells you which fields to trust.
STT quality is a patient-safety issue, not a nice-to-have."

**Bigger STT is better, but not sufficient.** The next section digs into
*why*, using `--transcribe-only` to isolate the STT layer from the
LLM's error-recovery ability.

## Comparing STT models in isolation (`--transcribe-only`)

For workshop prep, `--transcribe-only` skips the LLM pipeline entirely
and just runs STT on one or more audio files across one or more
Whisper models. Great for building a matrix quickly without waiting on
translation and extraction each time:

```bash
./.venv/bin/python3 main.py --transcribe-only \
  --audio ./audio/clip1.m4a ./audio/clip2.m4a \
  --language sw \
  --compare small,medium,models/kiswahili-sahihi-ct2
```

Prints a per-audio breakdown plus a compact summary matrix at the end.
No Ollama, no `<unused94>` reasoning traces, no schema drift - pure
STT signal.

### What we learned running this on the bundled Swahili sample

Ground truth Swahili:
*"Nina maumivu ya kichwa tangu jana. Pia nina homa na ninahisi
kizunguzungu ninaposimama."*

Ground truth English:
*"I have had a headache since yesterday. I also have a fever and I
feel dizzy when I stand up."*

Three-way `--transcribe-only` matrix on the Google-TTS recording of
that sentence, `--language sw`:

| Model                    | Compound`kizunguzungu`        | Compound`ninaposimama`   | Onset (`jana`)        | Punctuation      |
| ------------------------ | ------------------------------- | -------------------------- | ----------------------- | ---------------- |
| `small`                | broken (`hisiki zungu zungu`) | broken (`Nina posimama`) | wrong (`tangugana`)   | none             |
| `medium`               | broken (`hisiki zungu zungu`) | broken (`posi mama`)     | wrong (`tangu kana`)  | none             |
| `kiswahili-sahihi-ct2` | **intact**                | **intact**           | wrong (`tangu ghana`) | **intact** |

Two things worth flagging:

1. **`jana` breaks in all three.** Small → `tangugana`, medium → `kana`,
   Swahili fine-tune → `ghana`. Three totally different models,
   consistent failure - which strongly suggests the *audio* is the
   problem, not the models. Google TTS Swahili probably pronounces
   `jana` with a soft/aspirated first consonant that every Whisper
   variant hears the same way. A real Swahili speaker recording the
   same sentence would probably fix this instantly. **Model choice
   can't rescue you from input-modality artefacts.**
2. **The Swahili fine-tune is free on this hardware where we tested (Mac M3), around 5s vs medium's 6s -
   same underlying architecture (Whisper-medium), same RAM. Better
   output at no cost. On the Pi expect a similar ratio (but slower)

### The deeper finding: semantic-building-blocks vs surface fidelity

We took each of the three transcripts above and fed them to
**Google Translate** (a strong general translator, independent of our
pipeline) as a sanity-check on what downstream translation can
recover:

| Transcript from          | Google Translate output                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| `small`                | "I have a headache, a fever, and dizzy spells, and I feel unsteady on my feet."                    |
| `medium`               | "I have had a headache since yesterday. I also have a fever and feel dizzy; I feel really unwell." |
| `kiswahili-sahihi-ct2` | "I have had a headache since yesterday; I also have a fever and feel dizzy when I stand up."       |

The Swahili fine-tune's transcript survives translation almost
perfectly - even though it contains a wrong word (`ghana`) and an
inserted syllable (`nani`), the surrounding structure is intact enough
that Google Translate reconstructs the original meaning. The `small`
transcript, which has no obviously *wrong* words but lots of missing
spaces, produces a translation that's fluent but full of invented
material ("dizzy spells", "unsteady on my feet") and misses "since
yesterday" entirely.

**The design principle**: word-substitution errors (`jana` → `ghana`)
are recoverable by any competent downstream translator using context.
Word-boundary errors (`ninaposimama` → `Nina posimama`) destroy the
structural signal a translator needs to reconstruct meaning - and the
translator will still produce fluent output, just with invented
content plugged into the gaps. **If STT is going to fail, we want it
to fail on individual words, not on word boundaries.** The Swahili
fine-tune fails in the recoverable direction; multilingual Whisper
sizes fail in the destructive one.

Same lesson family as everything else in this demo: fluent output is
not evidence of accurate output. Google Translate paraphrasing
`hisiki zungu zungu` into "dizzy spells and feel unsteady on my feet"
is a hallucination that reads as reasonable clinical English.

## Why translate and extract use different models

Empirically found, live, during a workshop-prep session on this exact
codebase. Same broken transcript fed to a general model and a
medical-tuned model as the *translator*:

**How we got the broken transcript**:

1. Started from the correct Swahili sentence for "I have a headache
   since yesterday. I also have a fever and I feel dizzy when I stand
   up.":
   *"Nina maumivu ya kichwa tangu jana. Pia nina homa na ninahisi
   kizunguzungu ninaposimama."*
2. Played that through Google's TTS and recorded the audio - a
   synthetic voice, not a real Swahili speaker.
3. Ran the pipeline with `--whisper-size small` and **no** `--language`
   flag (so Whisper had to auto-detect). Whisper mis-detected the
   language as Portuguese at 0.39 confidence (essentially "I have no
   idea") and transcribed the audio *as if it were Portuguese*,
   producing:
   *"Nina maumivo ya quichua tango gana Pia nina roma na nina risi ki
   zungu zungu nina posi mama"*
4. That gibberish is what we then fed to each translator model below.

Three failures already stacked before either LLM even saw the text:
synthetic voice confused STT, low-confidence auto-detect wasn't
treated as a stop signal, and Whisper transcribed with Portuguese
phonetics anyway. All silent.

| Model                            | Output on garbled Swahili                                                                                                      | What a clinician does                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| `gemma4:e2b` (general)         | *"I have pain from [unclear reference]. I also have [symptom]. My head/neck area is very throbbing/painful. I have mother."* | **Stops.** Placeholders and "I have mother" are visibly wrong. |
| `medgemma1.5:latest` (medical) | *"Nina has ten years of language experience. Pia also has many mothers."*                                                    | **Charts it.** Sounds like a plausible (if odd) intake.        |

The general model failed loudly. The medical model failed silently.
That's a *safety* difference in exactly the direction that matters
clinically: honest ignorance is safer than confident fabrication.

**The mechanism:** MedGemma is fine-tuned on medical text where the
completion pattern is "given these symptoms/findings, reason
clinically." When given ambiguous input, its instinct is to *reason
clinically anyway* - inventing plausible medical content rather than
admitting it doesn't understand. The visible signature is a long
`<unused94>` reasoning trace: it spends its tokens "figuring out" what
each garbled fragment might mean, and each guess sounds authoritative
because it's dressed in numbered steps and "let me re-evaluate"
language.

A general model has more diverse training. When faced with nonsense,
it's more likely to refuse, transliterate literally-and-badly, or
insert placeholders - all of which are easier to catch as wrong.

**So**:

- Translation is a *language* task, not a medical one. A general
  multilingual model is the right tool - and (verified on this demo's
  clean sample) matches MedGemma's output on the good path while
  failing more visibly on the bad path.
- Extraction is a genuinely *medical* task on clean English input.
  Symptom taxonomies, standard medication names, spotting red flags -
  this is what the medical fine-tuning is for. Give it to MedGemma.

**Try the A/B live:**

```bash
# Default: gemma translates, medgemma extracts
./.venv/bin/python3 main.py --text "Nina maumivo ya quichua tango gana Pia nina roma na nina risi ki zungu zungu nina posi mama"

# Same input, but medgemma translates too - watch it confabulate
./.venv/bin/python3 main.py --text "Nina maumivo ya quichua tango gana Pia nina roma na nina risi ki zungu zungu nina posi mama" \
  --translate-model medgemma1.5:latest

# All-medgemma (the "one model does everything" baseline)
./.venv/bin/python3 main.py --text "..." \
  --translate-model medgemma1.5:latest --extract-model medgemma1.5:latest
```

**Caveat: chaining models can hide errors *between* stages.** If the
translator drops "fever", the extractor never sees it and can't
recover it. The pipeline mitigates this by printing intermediate
outputs at every step - a suspicious translation is a visible warning
to the clinician *before* extraction runs. Transparency of the
intermediate steps is what makes this architecture safe; hide those
prints and the safety guarantee disappears.

## How it works

0. **`transcribe_audio`** - `faster-whisper` (a lighter, offline-friendly
   alternative to OpenAI's original Whisper - no torch dependency, which
   matters on a Pi). Auto-detects language by default; force one with
   `--language <code>` when auto-detect struggles.
1. **`translate_to_english`** - one Ollama call, one job. Kept separate
   from extraction so the original patient text and the English
   translation can always be shown side by side (translation can lose
   nuance).
2. **`extract_structured_info`** - a second Ollama call that reads the
   English text and fills in a fixed set of fields (symptoms, onset,
   medications, etc.). The system prompt explicitly forbids diagnosis -
   the model organises what the patient said, it does not decide what's
   wrong with them. `format="json"` constrains the output to valid JSON.
3. **`generate_draft_note`** - plain Python, not an LLM call. Once the
   data is structured, formatting is deterministic - a template is
   simpler and more reliable than asking a model to do it again. Good
   rule of thumb: only call the LLM for the part of the task that
   genuinely needs language understanding.

## Real quirks worth knowing

- **`format="json"` constrains valid JSON, not the exact shape.** A
  field documented as a string can still come back as a list.
  `_as_text()` coerces defensively rather than trying to prompt this
  away entirely. Worth knowing generally about structured LLM output.
  Related failure seen in this codebase: when the translation step
  spills a long reasoning trace, the extraction step downstream
  sometimes returns `{"translation": "..."}` instead of the medical
  schema - `format="json"` accepts it, the pipeline silently loses
  every field, and the draft note ends up saying "Not stated." A
  future improvement would be to validate the returned dict against
  the expected schema and retry (or bail loudly) if the shape is
  wrong.
- **Whisper's language auto-detection can be unreliable on short
  clips**, even with real, clear speech. Short recordings routinely
  auto-detect at low confidence (0.3-0.4) and sometimes wrongly, with
  no warning - just a confident-looking transcript in the wrong
  phonetics (seen in this codebase: `pt` at 0.39 causing the whole
  garbled cascade documented above). `--language` forces the right
  one. A `language_probability` below ~0.7 is a real red flag worth
  building a refusal gate around; the demo doesn't do this yet, but
  it's the top of the "next improvements" list.
- **STT accuracy scales with model size, but so does speed and RAM
  - and size alone doesn't fix everything.** On a Pi 5, `small` is a
    reasonable default; `medium` catches more; the Swahili-specialist
    `kiswahili-sahihi-ct2` fine-tune preserves compound words that
    every multilingual size breaks (see the matrix above). Match model
    to language.
- **Reasoning-mode LLM output is expensive.** With MedGemma 1.5 or
  similar, per-question latency can vary 10x+ depending on whether
  the model decides to emit a `<unused94>` chain-of-thought. Watch
  the `translate` / `extract` numbers in the TIMING block if things
  feel slower than expected.

## Things to try next (exercises)

**On the STT layer:**

- Record the "headache/fever/dizzy" sentence with your *own* voice
  (not TTS) and re-run `--transcribe-only --compare small,medium, models/kiswahili-sahihi-ct2 --language sw`. Does `jana` finally
  land? If yes, you've isolated the failure to the synthetic-voice
  source - a fourth silent-failure axis worth naming live.
- Add a `--min-lang-confidence 0.7` gate: if Whisper's detected
  language probability is below the threshold, refuse to transcribe
  and print "pass `--language <code>` explicitly." Turns the silent
  cascade of the "Portuguese phonetics" run into a loud, actionable
  failure. Small change to `transcribe_audio`.
- Compare `--whisper-size tiny` all the way up to `large-v3-turbo` on
  the same audio. Where does the "fever" symptom first appear? Where
  does the `kizunguzungu` compound survive?

**On the LLM layer:**

- Take the medium-Whisper transcript
  (`Nina maumivu ya kichwa tangu kana ...`) and feed it via `--text`
  through both `--translate-model gemma4:e2b` and
  `--translate-model medgemma1.5:latest`. Confirm the loud-vs-silent
  failure contrast on your own hardware.
- Enforce the extraction schema shape - validate the returned dict
  contains the expected keys, retry once if not, warn loudly on
  second failure. Fixes the silent `{"translation": ...}` schema
  drift.
- Add a round-trip check: after step 1, translate the English back
  into the source language and print alongside the original. Crude
  but honest "did we lose anything obvious?" signal, one extra
  Ollama call.

**On the pipeline shape:**

- Feed the pipeline typed English text (`--text "..."`) and confirm
  the translation step is a no-op.
- Add a second sample audio in a different language and see how the
  auto-detect + translation + extraction combination handles it.
- Change one field in the extraction system prompt's schema and see
  how the JSON output changes.
- Try a sentence with no symptoms at all - what does
  `missing_information` look like?
- Add JSONL logging (per demo 2's `open_log` / `log_interaction`) so
  you can post-hoc analyse a workshop's worth of runs.
