# Demo 3: Medical image description

Same two-step pattern as demo 1: one vision-capable Ollama call to turn an
image into structured fields, then plain Python to format a readable
summary. The model only describes and flags quality issues - it never
diagnoses.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

`images/sample_ct_image.png` is a sample image already included (there's
also `images/sample_ct_abdomen_image.png`) - swap in your own image any
time by changing `IMAGE_PATH` in `main.py`.

`MODEL` is currently `"qwen3-vl:2b"`, a general vision model, since
MedGemma (`gemma1.5`) wasn't pullable on this dev machine. On the
Raspberry Pi, change `MODEL` back to `"gemma1.5:latest"` to use the
medical-tuned, vision-capable model instead.

## Run

```bash
./.venv/bin/python3 main.py
```

## How it works

1. **`extract_observations`** - one Ollama call with an image attached
   (`"images": [image_path]` in the message). The system prompt limits the
   model to: what kind of image, what's visible, whether the image quality
   is adequate, what a photo alone can't tell you, and a fixed
   "recommended next step" - never a diagnosis.
2. **`generate_summary`** - plain Python formatting, same reasoning as
   demo 1: once the data is structured, turning it into text doesn't need
   another model call.

## A real quirk you'll hit (and why it matters)

Some vision models - `qwen3-vl` among them - are "thinking" models: they
write a long internal reasoning trace before producing their actual
answer. Two consequences that show up directly in this code:

- `options={"num_predict": 4096}` gives the call enough token budget to
  get *through* the thinking and actually reach the JSON answer. With a
  low limit, the model can run out of budget mid-thought and return an
  empty or truncated response.
- Even with a generous budget, `format="json"` is a strong hint to the
  model, not a hard guarantee - it can still occasionally return malformed
  JSON. `extract_observations` retries up to 3 times before giving up
  instead of crashing on the first bad response. This is a normal,
  worth-knowing pattern for working with structured LLM output generally,
  not a hack specific to this demo.

MedGemma is not a "thinking" model, so you likely won't need as large a
token budget or as many retries once you switch to `gemma1.5:latest` on
the Pi - but the retry pattern is good practice to keep regardless.

## Why this streams instead of just waiting

A single non-streaming call here can look completely frozen for a minute
or more - all of that "thinking" trace happens before anything is
returned at all, so there's nothing to show until the model is done.
`extract_observations` uses `stream=True` instead, so it can print
something the whole time:

- a `.` roughly once a second while the model is in its thinking phase
  (throttled - printing one per token would be hundreds of dots for a
  long trace, which is just noise)
- a note the moment actual JSON content starts arriving
- how long the whole attempt took, once it's done

This doesn't make the model faster - a "thinking" model on modest
hardware can still take a minute or more per attempt - but it turns
"is this stuck?" into "I can see it's still working."

## Things to try next (exercises)

- Swap in a real photo (skin, wound, lab report) and see how the fields change.
- Remove the retry loop and the large `num_predict`, then run the script
  several times in a row - count how often it fails, to see the problem
  the fix actually solves.
- Add a `red_flags`-style field like in demo 1's schema, specific to images
  (e.g. "image_quality" issues that would make a clinician ask for a retake).
