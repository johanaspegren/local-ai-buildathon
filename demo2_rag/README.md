# Demo 2: Offline clinical RAG assistant

A from-scratch, in-memory RAG (retrieval-augmented generation) pipeline:
turn a PDF (or a set of curated Q&A pairs) into searchable chunks,
retrieve the most relevant ones for a question, and answer using only
that retrieved text - with citations.

No vector database - for a handful of documents, a Python list and cosine
similarity are simple, easy to understand, and fast enough. That's the
point of this demo: showing what a vector database is actually doing
underneath, before reaching for one.

The same script has a second job: showing what goes wrong when you
chunk sloppily, and how curation fixes it. Both stories run through the
same code - only the CLI flags change.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
ollama pull nomic-embed-text   # embedding model
ollama pull gemma4:e2b         # if not already on this machine
```

`CHAT_MODEL` is `"gemma4:e2b"` here for the same reason as the other
demos - `gemma1.5` (MedGemma) wasn't pullable on this dev machine. Switch
it to `"gemma1.5:latest"` on the Raspberry Pi.

The default document is `documents/who_pph_preeclampsia_excerpt.pdf` -
a 10-page slice of the WHO 2023 maternal-health recommendations covering
postpartum haemorrhage and pre-eclampsia. Real WHO text, clinically
meaningful for nurses, and the source of the trap questions used below.

## Run: interactive Q&A

```bash
./.venv/bin/python3 main.py
```

Indexes the default PDF once, then drops into a prompt where every
question you type is embedded, matched against the indexed chunks by
cosine similarity, and answered by the LLM using only the retrieved
context (with citations). Empty line or Ctrl-D exits.

Pass `--doc <path.pdf>` to point at a different PDF, or `--qa
<path.json>` to answer from a curated Q&A corpus instead. Indexing is
the slow part and only happens once per launch - subsequent questions
just do the embed + retrieve + generate loop.

## Run: the chunking demo (naive vs curated, side by side)

The interesting story: run the same 8 test questions against the same
source document twice, changing only the corpus, and watch the pass
rate change.

```bash
# 1. Naive chunking of the raw PDF - fixed 500 chars, no overlap.
#    This is the default a newcomer reaches for. Expect ~5/8 passes.
./.venv/bin/python3 main.py \
  --doc documents/who_pph_preeclampsia_excerpt.pdf \
  --chunk-size 500 --overlap 0 \
  --questions test_questions3.py

# 2. Same 8 questions, but the corpus is a hand-curated set of Q&A
#    pairs distilled from the same document. Expect 8/8 passes.
./.venv/bin/python3 main.py \
  --qa documents/qa3_curated.json \
  --questions test_questions3.py
```

Batch mode scores *retrieval*, not the LLM answer: PASS if every
expected keyword appears in the retrieved context, FAIL if none do,
PARTIAL if some do. That's deliberate - a chunking failure means the
right words never make it into the context in the first place, so
scoring retrieval alone isolates the failure being taught (no
prompt-tuning can rescue text that isn't there).

The three trap questions correspond to specific, verified failures of
naive 500-char chunking on the WHO excerpt:

- `oxytocin-ergometrine` is split mid-word across a chunk boundary
  (`...oxytoc` | `in-ergometrine...`), so no single retrieved chunk
  contains the intact drug name.
- `uterotonics` is split mid-word (`...other u` | `terotonics fail...`),
  so the phrase "uterotonics fail" never survives.
- A 4-item list of temporizing measures for uterine atony spans 3
  separate chunks; top-k retrieval only surfaces 2 of them, silently
  dropping at least one measure.

Curation fixes all three because a human decided what a self-contained
unit of knowledge is, rather than a character counter.

## How it works

1. **`load_pdf_pages` + `chunk_pages`** (or **`load_qa_corpus`**) -
   produce a list of chunks, each with a `text` field and a `source`
   label (e.g. `"page 3"` or `"Q&A #7"`) used for citations. PDF pages
   get fixed-size character chunks; curated Q&A pairs are used as-is,
   one chunk per pair.
2. **`embed_chunks`** - one batched Ollama call per batch turns every
   chunk into a vector. Runs once at index time, not per question.
3. **`retrieve`** - embeds the question, ranks every chunk by cosine
   similarity, returns the top `TOP_K`. Same code for both corpus types.
4. **`answer_question`** - retrieved chunks go straight into the system
   prompt, with an explicit instruction to answer only from that text
   and to say so if the answer isn't there. This is what keeps a RAG
   answer grounded instead of the model falling back on training data.
5. **`run_question_suite`** - retrieval-only batch scorer for the
   chunking demo. No LLM calls, so the pass/fail column reflects the
   corpus and nothing else.

## Speed: logging, streaming, and keeping both models loaded

This demo uses two different models back to back - an embedding model
(`nomic-embed-text`) and a chat model (`gemma4:e2b`, or `gemma1.5` on
the Pi). Three things matter for making that feel reasonably fast,
especially on constrained hardware:

1. **Progress logging.** `build_index` prints when each stage starts
   and how long it took, and `embed_chunks` prints after every batch.
   Without this, a slow embedding pass looks identical to a hung
   program.
2. **Streaming the answer.** `answer_question` uses `stream=True` and
   prints each piece of the response as it arrives, instead of waiting
   silently for the full answer. This doesn't make generation faster,
   but it's the difference between watching an answer get written and
   staring at a blank terminal.
3. **Keeping models loaded (the big one).** By default, Ollama may only
   keep one model resident in memory at a time. Since this demo calls
   the embedding model, then the chat model, then the embedding model
   again (for every question), that default can force Ollama to unload
   one model and reload the other on almost every call - and a reload
   can easily cost more time than the actual inference, especially on
   a Pi. Two things address this:
   - `keep_alive="30m"` is passed on every `ollama.embed` /
     `ollama.chat` call in this code, which tells Ollama not to unload
     that model for a while after use.
   - That alone isn't enough if the *server* is only willing to hold
     one model resident regardless. Set `OLLAMA_MAX_LOADED_MODELS=2`
     (or higher) as an environment variable for the Ollama server
     process itself, so it's willing to keep both models loaded at
     once. On macOS this typically means
     `launchctl setenv OLLAMA_MAX_LOADED_MODELS 2` and restarting
     Ollama; on the Pi (if Ollama runs as a systemd service) it means
     adding `Environment="OLLAMA_MAX_LOADED_MODELS=2"` to the service's
     config and restarting it.

   Check RAM headroom before raising this on the Pi 5 (8GB): both
   models need to fit in memory *at the same time* now, not one at a
   time. `nomic-embed-text` is small (~274MB), so this is usually a
   safe trade, but confirm with `ollama ps` while both are loaded
   rather than assuming.

## Things to try next (exercises)

- Ask a question the document doesn't answer - does the model actually
  say it doesn't know, or does it guess anyway? This is the most
  important thing to test in any RAG system.
- Print cosine similarity scores next to each retrieved chunk in
  `run_single_question` - how much better is the top match than the
  third?
- Try `--chunk-size 1000 --overlap 200` on the naive suite - does
  bigger chunking recover some of the traps? Which ones stay broken
  no matter what character count you pick?
- Add a second PDF and extend `build_index` to index multiple files at
  once, keeping the file name in each chunk's `source` label.
