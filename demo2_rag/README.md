# Demo 2: Offline clinical RAG assistant

A from-scratch, in-memory RAG (retrieval-augmented generation) pipeline:
turn a PDF into searchable chunks, retrieve the most relevant ones for a
question, and answer using only that retrieved text - with citations.

No vector database - for a handful of documents, a Python list and cosine
similarity are simple, easy to understand, and fast enough. That's the
point of this demo: showing what a vector database is actually doing
underneath, before reaching for one.

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

`documents/public_health_act.pdf` (Tanzania's Public Health Act) is the
sample document - swap in your own guideline PDFs by changing
`DOCUMENT_PATH`, or extend `build_index` to load a whole folder.

## Run

```bash
./.venv/bin/python3 main.py
```

## How it works

1. **`load_document` / `chunk_pages`** - extract each page's text, then
   split it into fixed-size overlapping chunks. Each chunk keeps the page
   number of the page it came from - that's what makes citations possible
   later. Real-world chunking would split on sentences or sections rather
   than a fixed character count, but fixed-size chunks are enough to show
   the pattern.
2. **`embed_chunks`** - one batched Ollama call turns every chunk's text
   into a vector. This runs once when the document is indexed, not once
   per question.
3. **`retrieve`** - embeds the question, then ranks every chunk by cosine
   similarity to find the most relevant ones. This is the whole idea
   behind "retrieval": compare meaning (via embeddings), not keywords.
4. **`answer_question`** - the retrieved chunks are placed directly into
   the system prompt, along with an explicit instruction to answer only
   from that text and to say so plainly if the answer isn't there. This
   is what keeps a RAG answer grounded instead of the model just
   answering from its own training data.

## Things to try next (exercises)

- Ask a question the document doesn't answer (e.g. about a topic outside
  this Act entirely) - does the model actually say it doesn't know, or
  does it guess anyway? This is the most important thing to test in any
  RAG system.
- Print the cosine similarity scores next to each retrieved chunk in
  `run_pipeline` - how much better is the top match than the third?
- Try a smaller `CHUNK_SIZE` (e.g. 300) and see whether retrieval quality
  or citation precision changes.
- Add a second PDF to `documents/` and extend `build_index` to index
  multiple files, keeping track of which file each chunk came from too.
