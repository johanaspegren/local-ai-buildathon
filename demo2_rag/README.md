# Demo 2: Offline clinical RAG assistant

A from-scratch, in-memory RAG (retrieval-augmented generation) pipeline:
turn a PDF (or a set of curated Q&A pairs) into searchable chunks,
retrieve the most relevant ones for a question, and answer using only
that retrieved text - with citations and WHO evidence grades.

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
ollama pull nomic-embed-text
```

### Chat model: MedGemma 1.5 4B

We use MedGemma 1.5 (Google's medical Gemma variant) as the chat model.
Two ways to get it - pick whichever your network allows. Both produce a
model runnable via `ollama run medgemma1.5:latest`, and inference is
identical either way.

#### Option A: direct pull from Ollama's registry

Simplest when it works:

```bash
ollama pull medgemma1.5:latest
```

That's it - the tag is already `medgemma1.5:latest`, so `main.py` uses
it as-is with no aliasing step.

#### Option B: pull from Hugging Face (fallback when the Ollama registry is blocked)

Corporate proxies (Zscaler and similar) often block
`registry.ollama.ai` but allow `huggingface.co`. In that case:

```bash
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M
```

Then use the `Modelfile` in this folder to alias the HF pull to the
same `medgemma1.5:latest` tag `main.py` expects, so nothing in the code
has to change based on where the weights came from:

```bash
ollama create medgemma1.5:latest -f Modelfile
ollama run medgemma1.5:latest "hello"  # sanity check
```

MedGemma is a gated model on HF - accept Google's Health AI Developer
Foundations terms on the model card page (while logged in) before the
pull; if a proxy strips your session or the licence isn't accepted,
`ollama pull` will 401.

### Context window: `num_ctx` and the speed/quality trade

The `Modelfile` sets `PARAMETER num_ctx 8192`. This is worth
understanding because it's a real lever with a real cost.

Check what Ollama has for this model right now:

```bash
ollama show medgemma1.5:latest --parameters
```

- **Too small**: when the prompt (system prompt + retrieved chunks +
  question + generated answer so far) exceeds `num_ctx`, Ollama
  silently truncates the *front* of the prompt - which is where your
  retrieved chunks live. Text you needed the model to read just
  vanishes with no warning. In this demo, three 500-char chunks + the
  system prompt + question can easily reach 3-4k tokens, and MedGemma
  1.5 in reasoning mode can generate 300-600 more tokens of `<unused94> thought ...` before the answer. Undersized `num_ctx` is a stealth
  correctness failure.
- **Too big**: memory and inference cost both scale with `num_ctx`.
  The KV-cache reserved for the context grows linearly, and every
  prompt-eval token has to attend across the full window. On a Pi 5
  going from 2048 → 8192 is roughly 4× more KV-cache RAM per loaded
  model and noticeably slower prompt-eval.

Reasonable options for this demo:

- **2048**: fastest, lowest memory. Risky here because chunk-heavy
  RAG plus reasoning-mode generation can silently overflow. Consider
  only if you're also lowering `TOP_K` and disabling reasoning-mode
  output.
- **4096**: often the Ollama default that ships with a GGUF (check
  with `ollama show`). Comfortable for top_k=3 with modest chunks and
  short answers; still tight when the model spills a long
  chain-of-thought.
- **8192** (this Modelfile's setting): comfortable headroom for top_k=3
  at any chunk_size the demo uses, plus reasoning-mode traces. Worth
  the RAM for a demo where "the model silently ignored the top chunk"
  is exactly the failure we're trying to make visible.

If you tune `num_ctx`, watch the `prompt=N tok` value in the per-question
timing line and compare against the setting - if `prompt` is creeping
close to `num_ctx`, either raise `num_ctx` or shrink the context (lower
`TOP_K` / smaller chunks). To apply a change: edit `Modelfile`, then
`ollama create medgemma1.5:latest -f Modelfile` (no re-download, just
a lightweight reapply).

### Default document

`documents/who_pph_preeclampsia_excerpt.pdf` - a 10-page slice of the
WHO 2023 maternal-health recommendations covering postpartum
haemorrhage and pre-eclampsia. Real WHO text, clinically meaningful
for nurses, and the source of the trap questions used below.

`documents/qa_who_pph_preeclampsia_excerpt.json` - the same content
distilled into 21 hand-curated Q&A pairs, each tagged with the exact
WHO recommendation strength and evidence quality (e.g. "Strong
Recommendation, moderate-quality evidence", "Weak Recommendation
against, very low-quality evidence"). Grades are grounded verbatim in
the source PDF's recommendation boxes.

## Run: interactive Q&A

```bash
./.venv/bin/python3 main.py
```

Indexes the default PDF once, then drops into a prompt where every
question you type is embedded, matched against the indexed chunks by
cosine similarity, and answered by the LLM using only the retrieved
context (with citations and WHO grades where present). Empty line or
Ctrl-D exits.

Pass `--doc <path.pdf>` to point at a different PDF, or `--qa <path.json>` to answer from a curated Q&A corpus. Indexing i only happens once per launch - subsequent questions just do the embed + retrieve + generate loop.

Every question shows:

- **Retrieved chunks**, in full, with source label and cosine
  similarity score. For Q&A pairs, both the question and the answer
  are shown so an auditor sees everything the curator wrote.
- **The streamed LLM answer**, including any WHO grade tag from the
  source (the system prompt asks the model to include grades verbatim).
- **Timings** for each phase: question embedding, cosine scan,
  prompt-eval, generation. Tokens/second included.

Every interaction is also appended to a timestamped JSONL log under
`logs/`, self-describing header first, so any run can be replayed or
compared later.

## Run: the chunking demo (naive vs curated, side by side)

Run the same test questions against the same source document twice,
changing only the corpus:

```bash
# 1. Naive chunking of the raw PDF - fixed 500 chars, no overlap.
#    This is the default a newcomer reaches for.
./.venv/bin/python3 main.py \
  --doc documents/who_pph_preeclampsia_excerpt.pdf \
  --chunk-size 500 --overlap 0 \
  --questions test_questions3.py

# 2. Same 8 questions, but the corpus is a hand-curated set of Q&A
#    pairs distilled from the same document.
./.venv/bin/python3 main.py \
  --qa documents/qa_who_pph_preeclampsia_excerpt.json \
  --questions test_questions3.py
```

Batch mode scores *retrieval*, not the LLM answer: PASS if every
expected keyword appears in the concatenated top-3 retrieved chunks,
FAIL if none do, PARTIAL if some do. That's deliberate - it isolates
the retrieval layer from generation.

### Expected result

- **Naive 500/0**: **6/8** - Q2 and Q5 fail because their expected
  keyword is a WHO term that this codebase's per-page chunker splits
  every time it appears, so the intact phrase exists in *zero* chunks.
  No top-k combination can rescue text that isn't there.
- **Curated Q&A**: **8/8** - each curated pair is a self-contained
  unit, so the intact term always survives.

The two trap keywords, verified against the live chunk output:

- **`intrauterine balloon tamponade`** (Q2) - the WHO device name.
  Never appears intact in any single 500-char naive chunk. `balloon
  tamponade` alone would survive; the full clinical term is what
  breaks. Curated Q&A #7 has it intact.
- **`controlled cord traction`** (Q5) - split at a page-4 chunk
  boundary as `...c` | `ontrolled cord traction...`, so the intact
  phrase never appears in any retrieved chunk. Curated Q&A #9 has it
  intact.

Traps are chunker-implementation-specific. Change `chunk_size`,
`overlap`, or move from per-page to whole-document chunking, and these
splits move too. Re-verify with `--show-chunks` and a grep if you
change the chunker.

### The bigger catch: even 6/8 undersells how wrong naive can get

The batch scorer only sees *retrieval*. It doesn't run the LLM. So
even the two Q that fail here fail *silently* about how bad things
really are, and the six that pass fail *loudly* about how safe things
really are. Interactive mode against the same naive corpus repeatedly
produces:

- Wrong drug nomenclature (`ergometrine and oxytocin` instead of the
  official `oxytocin-ergometrine` fixed-dose product) even when the
  scorer PASSes because the words are technically present.
- Invented bullets from chunk fragments (`Surgical intervention (surg)`
  promoted to a recommendation from a mid-word cut of `surgical`).
- Contraindicated interventions listed as if recommended (`uterine
  packing`, which WHO specifically recommends *against*).

The intended reading:

> **Retrieval-only tests are necessary but not sufficient for RAG
> evaluation.** A keyword-match scorer can go partly green on a corpus
> that the LLM turns into dangerous answers - and even the FAILures it
> catches undersell the danger. Curation still wins, both because it
> passes the scorer *and* because it delivers self-contained,
> unambiguous, grade-tagged units the LLM can't decapitate, fragment,
> or misgroup.

Live-demo flow:

1. Run the batch suite naive → 6/8. Point at Q2 and Q5 red, note the
   other 6 green. "The failures are honest, the passes are not."
2. Run one of the six "passing" trap questions interactively against
   the same naive corpus → observe the actual answer. Wrong
   nomenclature, invented bullets, or contraindicated drug listed as
   recommended.
3. Run the same interactive questions against the curated corpus →
   clean answers with WHO grades.

The scorer plus the interactive comparison together are the story -
neither alone is enough.

### Show every chunk

For a live audience, `--show-chunks` dumps the entire indexed corpus to
the terminal after indexing (before dropping into the REPL or batch):

```bash
# See the naive splits in the wild - scroll to find the ...oxytoc |
# in-ergometrine... boundary and point at it.
./.venv/bin/python3 main.py \
  --doc documents/who_pph_preeclampsia_excerpt.pdf \
  --chunk-size 500 --overlap 0 --show-chunks

# Contrast: 21 clean Q&A pairs, each with WHO grade attached.
./.venv/bin/python3 main.py \
  --qa documents/qa_who_pph_preeclampsia_excerpt.json --show-chunks
```


## Failure modes seen here

Beyond the three keyword traps, the demo has surfaced several richer
failure modes worth narrating when they appear:

- **Retrieval PASS, LLM FAIL.** The scorer marks a naive run PASS
  because all keywords are in the retrieved context, but the LLM
  answer is still wrong - because the chunk had the words without the
  semantic dependencies. Example: `oxytocin-ergometrine` appears in a
  chunk that starts with `ne, / oxytocin-ergometrine fixed dose...`,
  headless because the conditional clause "If IV oxytocin is
  unavailable" is in the previous chunk. MedGemma reasoned safely
  ("can't tie this to the scenario asked") and refused to answer -
  correct behaviour, wrong root cause.
- **Hallucinated bullets from chunk fragments.** Naive answers to the
  temporizing-measures question sometimes include a bullet like
  `Surgical intervention (surg)` - the trailing `(surg)` is a chunk
  fragment (`surg|ical`) the model promoted to a recommendation.
- **Contraindicated interventions promoted to recommendations.**
  Similarly, a naive answer once listed `uterine packing` as a
  recommended measure - WHO specifically recommends *against* it.
  A hazard the batch scorer's keyword check misses entirely.
- **Reasoning traces are model output, not verified logic.** MedGemma
  1.5 emits chain-of-thought in `<unused94>...<unused95>` when it
  senses conflict between chunks (great to leave visible for a
  workshop). But those traces can be confidently wrong about the
  source - a nice reminder that they're text-shaped like reasoning,
  not proof.
- **Temperature 0.8 default causes run-to-run variance.** Identical
  retrieval, different answers between runs. Set `temperature` in
  `Modelfile` (or per-call `options={"temperature": 0.2}`) if you want
  reproducibility for slides, but the variance itself is a genuine
  RAG lesson worth showing.

## How it works

1. **`load_pdf_pages` + `chunk_pages`** (or **`load_qa_corpus`**) -
   produce a list of chunks, each with a `text` field (what goes to
   the LLM), an optional `display_text` (what's shown to a human -
   e.g. full Q&A pair), and a `source` label (`"page 3"`,
   `"Q&A #7"`) used for citations.
2. **`embed_chunks`** - one batched Ollama call per batch turns every
   chunk into a vector. Q&A pairs embed `"question + answer"` together
   so the curator's question phrasing helps retrieval, while only the
   answer is shown to the LLM.
3. **`retrieve`** - embeds the question, ranks every chunk by cosine
   similarity, returns the top `TOP_K` with scores attached. Same code
   for both corpus types. Returns timings alongside chunks.
4. **`answer_question`** - retrieved chunks go straight into the system
   prompt with instructions to answer only from that text, cite
   sources, include any WHO grade verbatim, and say so plainly when
   the answer isn't there. Returns Ollama's built-in prompt/gen
   token counts and durations.
5. **`run_question_suite`** - retrieval-only batch scorer for the
   chunking demo. No LLM calls, so the pass/fail column reflects the
   corpus and nothing else.
6. **`open_log` / `log_interaction`** - one JSONL file per run under
   `logs/`, first line is a header (corpus, chunk params, models, top_k),
   each subsequent line is a full interaction including retrieved
   chunks (full text, source, score) and timings.

## Speed: what to watch when tuning

The per-question timing line looks like:

```
TIMING
  retrieval:  embed=42ms  scan=0.3ms  (over 51 chunks)
  generation: wall=8.4s  prompt=612 tok in 1.2s (510 tok/s)  gen=87 tok in 7.2s (12.1 tok/s)
  (model load: 3.2s - first call after idle)
```

What each number tells you:

- **`embed`** - question embedding via `nomic-embed-text`. Should be
  30-80 ms. If it suddenly jumps to seconds, the embedding model got
  swapped out (see below).
- **`scan`** - pure-Python cosine loop over all chunks. Sub-millisecond
  for a few hundred chunks. If this ever becomes non-negligible you
  have enough scale to justify a real vector database.
- **`prompt`** tokens/duration - how long the model spent digesting
  the prompt (system prompt + retrieved chunks + question). Grows with
  `--chunk-size`, `TOP_K`, `num_ctx`. Often the biggest lever on a Pi.
- **`gen`** tokens/second - the "does streaming feel snappy" number.
  For MedGemma 1.5 4B Q4_K_M expect roughly 8-15 tok/s on a Pi 5.
- **`model load`** - only appears when Ollama had to bring the model
  into memory. If it fires on more than the first question, keep_alive
  isn't holding, or memory pressure evicted it.

Fastest wins for a Pi are almost always: (1) smaller `TOP_K`, (2)
shorter chunks (less prompt to eval), (3) making sure both models stay
loaded (see next section).

### Keeping both models loaded

This demo calls two models back-to-back per question. By default,
Ollama may only keep one model resident in memory at a time, forcing
it to unload one and reload the other on almost every call - a reload
can easily cost more time than the actual inference on a Pi.

Two things address this:

- **`keep_alive="30m"`** is passed on every `ollama.embed` /
  `ollama.chat` call in this code, so Ollama doesn't unload models
  aggressively.
- **`OLLAMA_MAX_LOADED_MODELS=2`** - set this as an environment
  variable for the Ollama *server* process, otherwise the server will
  hold only one model regardless of what clients ask. On macOS:
  `launchctl setenv OLLAMA_MAX_LOADED_MODELS 2` then restart Ollama.
  On the Pi (systemd): add
  `Environment="OLLAMA_MAX_LOADED_MODELS=2"` to the service unit and
  restart it.

Confirm both models are resident after warmup with `ollama ps`. On the
Pi 5 (8GB), MedGemma 4B Q4 + `nomic-embed-text` fits comfortably, but
watch RAM if you raise `num_ctx` significantly.

## Analysing a run afterwards

Every run's JSONL log is easy to slice with `jq`:

```bash
# Just the trap questions from a naive run
jq -c 'select(.type=="suite" and .trap==true) | {id, result, question}' \
  logs/*-doc-cs500-ov0.jsonl

# For each interactive question, what did the model actually see?
jq -c 'select(.type=="single") | {question, sources: [.retrieved[].source], scores: [.retrieved[].score]}' \
  logs/*.jsonl

# Which questions triggered <unused94> reasoning?
jq -c 'select(.type=="single" and (.answer | contains("<unused94>"))) | .question' \
  logs/*.jsonl
```

Pairing an answer with the scores of the chunks the model saw is the
killer analysis view - it lets you point directly at *why* an answer
was subtly wrong.

## Things to try next (exercises)

- Ask a question the document doesn't answer - does the model actually
  say it doesn't know, or does it guess anyway? Most important thing
  to test in any RAG system.
- Try `--chunk-size 1000 --overlap 200` on the naive suite - does
  bigger chunking recover some traps? Which ones stay broken no matter
  what character count you pick?
- Turn temperature down (add `PARAMETER temperature 0.2` to
  `Modelfile`, re-run `ollama create`) and re-run a trap question five
  times. Does the answer stabilise? Does it become more right, or just
  more consistently wrong?
- Add a second PDF and extend `build_index` to index multiple files at
  once, keeping the file name in each chunk's `source` label.
- Add a "refuse if top-1 score below 0.7" gate in `retrieve` - would
  that have caught any of the naive-run wrong answers? Any of the
  curated ones?
