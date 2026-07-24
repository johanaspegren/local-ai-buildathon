"""
Demo 2: Offline clinical RAG assistant.

Pipeline (each step is one clear, separate function):
  1. Load a corpus - either a PDF (extract text + split into chunks, keeping
     page numbers for citations) or a curated JSON of Q&A pairs (each pair
     becomes one "chunk" with the answer as its text)
  2. Each chunk -> an embedding vector (one Ollama call per chunk, batched)
  3. A question -> its own embedding, compared against every chunk to find
     the most relevant ones (this is "retrieval")
  4. The retrieved chunks + the question -> an answer, with citations,
     using an Ollama chat call that is told to answer ONLY from the
     provided text and to say so if the answer isn't there

This is deliberately a from-scratch, in-memory RAG - no vector database.
That's the point: for a handful of short documents, a Python list and
cosine similarity are simple, easy to understand, and fast enough.

The CLI is parameterized so the same code can demo two very different
setups back-to-back and let students see the difference:

  # Naive chunking of a real PDF (fixed 500 chars, no overlap - the
  # default a newcomer reaches for, and the source of the trap questions):
  python main.py --doc documents/who_pph_preeclampsia_excerpt.pdf \\
                 --chunk-size 500 --overlap 0 \\
                 --questions test_questions3.py

  # Same questions, same retrieval code, but the corpus is hand-curated
  # Q&A pairs distilled from the same document:
  python main.py --qa documents/qa3_curated.json \\
                 --questions test_questions3.py

  # Interactive: index the corpus once, then keep asking questions.
  # Each question is embedded, matched against the indexed chunks, and
  # answered by the LLM using the retrieved context. Ctrl-D or an empty
  # line exits.
  python main.py --doc documents/who_pph_preeclampsia_excerpt.pdf
"""

import argparse
import datetime as dt
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader

# Structured log of every interaction (question, retrieved chunks with
# similarity scores, LLM answer, batch scoring result). One JSONL file
# per run, timestamped, in ./logs/. Great for replaying the demo, for
# post-hoc analysis of what naive-chunking retrieval actually looked
# like, and for pointing at when someone says "but the answer looked
# right" - the log shows exactly which chunks the model saw.
LOG_DIR = Path("logs")
LOG_FILE: Path | None = None  # set in main once we know the run mode

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "medgemma1.5:latest"

DEFAULT_DOC = "documents/who_pph_preeclampsia_excerpt.pdf"

# Sensible defaults for the "just show me a working RAG" run. The teaching
# demo overrides these on the CLI (chunk_size=500, overlap=0) to reproduce
# the naive-chunking failures documented in test_questions3.py.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

EMBED_BATCH_SIZE = 32

# Tells Ollama to keep a model loaded in memory for this long after a
# call, instead of unloading it right away. Both models used here
# (EMBED_MODEL and CHAT_MODEL) get called repeatedly in the same run, and
# without this, Ollama can swap one model out to load the other between
# calls - which on constrained hardware (like a Pi 5) can cost far more
# time than the actual inference. See the README for the server-side
# setting (OLLAMA_MAX_LOADED_MODELS) that avoids this swapping entirely.
KEEP_ALIVE = "30m"

TOP_K = 3


# ---------------------------------------------------------------------------
# Corpus loading: two ways in, one shape out.
#
# Every chunk is a dict with "text" (what gets embedded and shown to the
# LLM) and "source" (a short human-readable label used in citations - e.g.
# "page 3" for a PDF chunk, "Q&A #7" for a curated pair). That's all the
# rest of the pipeline cares about.
# ---------------------------------------------------------------------------


def load_pdf_pages(path: str) -> list[dict]:
    """Extract text from a PDF, one entry per page."""
    reader = PdfReader(path)
    return [
        {"page": page_number, "text": page.extract_text()}
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def chunk_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split each page's text into fixed-size chunks with optional overlap.

    Fixed-size character chunking is deliberately dumb: it cuts wherever
    the character counter lands, with no regard for word, sentence, or
    section boundaries. That's the whole point of the teaching demo -
    with chunk_size=500 and overlap=0 you can watch it slice a drug name
    like "oxytocin-ergometrine" clean in half, so no retrieved chunk ever
    contains the intact phrase the question is asking about.
    """
    chunks = []
    step = max(1, chunk_size - overlap)
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": f"page {page['page']}"})
            start += step
    return chunks


def load_qa_corpus(path: str) -> list[dict]:
    """Load a curated Q&A JSON file as an already-chunked corpus.

    Each pair becomes one chunk. We embed "question + answer" together
    (so the question phrasing helps retrieval find the right pair) but
    only the answer text is shown to the LLM as the retrieved context.
    The full question+answer is kept on the chunk as "display_text" so
    the console/log can show a human auditor everything the curator
    wrote, not just what the model saw. This is the "curated" side of
    the demo: no chunking happens at all, because a human already
    decided what a self-contained unit of knowledge is.
    """
    pairs = json.loads(Path(path).read_text())
    return [
        {
            "text": pair["answer"],
            "embed_text": f"{pair['question']} {pair['answer']}",
            "display_text": f"Q: {pair['question']}\nA: {pair['answer']}",
            "source": f"Q&A #{i + 1}",
        }
        for i, pair in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# Embedding + retrieval: identical for both corpus types.
# ---------------------------------------------------------------------------


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach an embedding vector to every chunk.

    Chunks are embedded in batches (not one call per chunk, and not one
    giant call for everything) so progress is visible and each request
    stays a reasonable size. If a chunk has an "embed_text" field it's
    used in preference to "text" - that lets curated Q&A pairs embed the
    question+answer together while still returning just the answer as
    retrieved context.
    """
    start = time.time()
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        response = ollama.embed(
            model=EMBED_MODEL,
            input=[c.get("embed_text", c["text"]) for c in batch],
            keep_alive=KEEP_ALIVE,
        )
        for chunk, embedding in zip(batch, response["embeddings"]):
            chunk["embedding"] = np.array(embedding)
        done = batch_start + len(batch)
        print(f"  embedded {done}/{len(chunks)} chunks ({time.time() - start:.1f}s elapsed)")
    return chunks


def build_index(args: argparse.Namespace) -> list[dict]:
    """Turn the requested corpus into a list of embedded chunks.

    This only needs to run once per corpus, not once per question.
    Prints per-stage timings so a slow run on a Pi doesn't look hung.
    """
    if args.qa:
        print(f"Loading curated Q&A corpus {args.qa}...")
        start = time.time()
        chunks = load_qa_corpus(args.qa)
        print(f"  {len(chunks)} Q&A pairs ({time.time() - start:.1f}s)")
    else:
        print(f"Loading {args.doc}...")
        start = time.time()
        pages = load_pdf_pages(args.doc)
        print(f"  loaded {len(pages)} pages ({time.time() - start:.1f}s)")

        print(f"Splitting into chunks (size={args.chunk_size}, overlap={args.overlap})...")
        start = time.time()
        chunks = chunk_pages(pages, args.chunk_size, args.overlap)
        print(f"  {len(chunks)} chunks ({time.time() - start:.1f}s)")

    print(f"Embedding {len(chunks)} chunks in batches of {EMBED_BATCH_SIZE}...")
    return embed_chunks(chunks)


def retrieve(question: str, chunks: list[dict], top_k: int = TOP_K) -> tuple[list[dict], dict]:
    """Find the chunks most relevant to the question, with phase timings.

    Cosine similarity between the question's embedding and every chunk's
    embedding - simple linear scan, which is entirely fine for a few
    hundred chunks. A real vector database earns its place at a much
    larger scale than a workshop demo.

    Returns (retrieved_chunks, timings) where timings breaks down where
    the seconds went: embedding the question (an Ollama call) vs the
    pure-Python cosine scan. Useful for judging whether it's worth
    reaching for a vector database yet (spoiler: at a few hundred
    chunks, the embed call dominates and the scan is negligible).
    """
    t0 = time.time()
    response = ollama.embed(model=EMBED_MODEL, input=question, keep_alive=KEEP_ALIVE)
    question_vector = np.array(response["embeddings"][0])
    t_embed = time.time() - t0

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    t1 = time.time()
    scored = [(cosine_similarity(question_vector, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    t_scan = time.time() - t1

    retrieved = [{**chunk, "score": score} for score, chunk in scored[:top_k]]
    timings = {"embed_question_s": t_embed, "scan_s": t_scan, "num_chunks": len(chunks)}
    return retrieved, timings


def open_log(args: argparse.Namespace) -> Path:
    """Create a timestamped JSONL log for this run and write a header line.

    The header records the corpus, chunking parameters, and models used,
    so a log file is self-describing - you can read one back later and
    know exactly which configuration produced it.
    """
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "qa" if args.qa else f"doc-cs{args.chunk_size}-ov{args.overlap}"
    path = LOG_DIR / f"{timestamp}-{tag}.jsonl"
    header = {
        "type": "run",
        "timestamp": dt.datetime.now().isoformat(),
        "corpus": args.qa or args.doc,
        "corpus_kind": "qa" if args.qa else "pdf",
        "chunk_size": None if args.qa else args.chunk_size,
        "overlap": None if args.qa else args.overlap,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "top_k": TOP_K,
    }
    path.write_text(json.dumps(header) + "\n")
    print(f"Logging to {path}\n")
    return path


def log_interaction(record: dict) -> None:
    """Append one JSON line to the current run's log file.

    Chunk text is logged in full (not truncated) - the whole point of
    the log is to let you see exactly what the model saw, including any
    mid-word splits or garbled fragments that naive chunking introduced.
    """
    if LOG_FILE is None:
        return
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Two ways to use the index: ask one question (with an LLM answer), or
# run a scored batch of test questions (retrieval-only, so PASS/FAIL
# reflects the chunking, not the model).
# ---------------------------------------------------------------------------


def answer_question(question: str, retrieved_chunks: list[dict]) -> tuple[str, dict]:
    """Answer using only the retrieved text, with citations, plus timings.

    The system prompt is explicit about the two failure modes we want to
    avoid: making things up, and citing sources that weren't actually
    used. stream=True prints each piece of the answer as it's generated
    instead of waiting for the whole response.

    Returns (answer_text, timings). The final streamed message from
    Ollama carries built-in timing counters we surface here:
      - prompt_eval_count / prompt_eval_duration: how long the model
        spent digesting the prompt (system prompt + retrieved chunks +
        question) before generating anything. Dominated by prompt length
        - bigger chunks or higher top_k grow this.
      - eval_count / eval_duration: generation itself, one token at a
        time. Divide to get tokens/second, the number to optimise if the
        answer feels slow to *stream*.
      - load_duration: time to bring the model into memory. Zero after
        the first call (assuming keep_alive holds it there).
    """
    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    system_prompt = """You are a clinical reference assistant. Answer the
question using ONLY the provided document excerpts below. Cite the
source label(s) (e.g. "page 3" or "Q&A #7") you used. If an excerpt
includes a WHO grade in parentheses (e.g. "(WHO: Strong Recommendation,
moderate-quality evidence.)"), include that grade verbatim in your
answer so the reader can judge how confident the guideline is. If the
excerpts don't contain the answer, say so plainly instead of guessing.

Document excerpts:
""" + context

    t0 = time.time()
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        keep_alive=KEEP_ALIVE,
        stream=True,
    )

    full_answer = ""
    final_chunk = None
    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        full_answer += piece
        final_chunk = chunk
    print()

    wall_s = time.time() - t0
    # Ollama reports durations in nanoseconds. Missing fields (e.g. if
    # the server is older) default to 0 rather than crashing the demo.
    def ns_to_s(ns) -> float:
        return (ns or 0) / 1e9
    prompt_tokens = (final_chunk or {}).get("prompt_eval_count", 0)
    prompt_s = ns_to_s((final_chunk or {}).get("prompt_eval_duration"))
    gen_tokens = (final_chunk or {}).get("eval_count", 0)
    gen_s = ns_to_s((final_chunk or {}).get("eval_duration"))
    load_s = ns_to_s((final_chunk or {}).get("load_duration"))
    timings = {
        "wall_s": wall_s,
        "load_s": load_s,
        "prompt_eval_tokens": prompt_tokens,
        "prompt_eval_s": prompt_s,
        "prompt_tokens_per_s": prompt_tokens / prompt_s if prompt_s else None,
        "gen_tokens": gen_tokens,
        "gen_s": gen_s,
        "gen_tokens_per_s": gen_tokens / gen_s if gen_s else None,
    }
    return full_answer.strip(), timings


def run_single_question(question: str, chunks: list[dict]) -> None:
    print("=" * 60)
    print("QUESTION")
    print(question)

    retrieved, retrieval_t = retrieve(question, chunks)
    print("\nRETRIEVED CHUNKS")
    for chunk in retrieved:
        print(f"  [{chunk['source']}] score={chunk['score']:.3f}")
        # Indent the chunk body two spaces so it's visually nested under
        # the source/score header and easy to skim, but show the full
        # text - the whole point of surfacing it is to let a human
        # audit exactly what the curator (or naive chunker) put here.
        # For Q&A corpora that's the full "Q: ... / A: ..." pair; for
        # PDF corpora it's the raw chunk that will go to the LLM.
        body = chunk.get("display_text", chunk["text"])
        for line in body.splitlines() or [""]:
            print(f"    {line}")
        print()

    print("\nANSWER")
    answer, gen_t = answer_question(question, retrieved)

    # Compact timing summary. Rule of thumb for reading it:
    # - embed_question is one Ollama call to nomic-embed; usually tens of ms.
    # - scan is pure Python over ~all chunks; sub-ms unless corpus is huge.
    # - prompt_eval scales with retrieved-context length; the number to
    #   watch when you change chunk_size / top_k / num_ctx.
    # - gen tokens/s is the "does streaming feel snappy" number.
    print("\nTIMING")
    print(f"  retrieval:  embed={retrieval_t['embed_question_s']*1000:.0f}ms  "
          f"scan={retrieval_t['scan_s']*1000:.1f}ms  "
          f"(over {retrieval_t['num_chunks']} chunks)")
    print(f"  generation: wall={gen_t['wall_s']:.1f}s"
          + (f"  prompt={gen_t['prompt_eval_tokens']} tok in {gen_t['prompt_eval_s']:.1f}s"
             f" ({gen_t['prompt_tokens_per_s']:.0f} tok/s)" if gen_t['prompt_tokens_per_s'] else "")
          + (f"  gen={gen_t['gen_tokens']} tok in {gen_t['gen_s']:.1f}s"
             f" ({gen_t['gen_tokens_per_s']:.1f} tok/s)" if gen_t['gen_tokens_per_s'] else ""))
    if gen_t["load_s"] > 0.05:
        print(f"  (model load: {gen_t['load_s']:.1f}s - first call after idle)")
    print("=" * 60)

    log_interaction({
        "type": "single",
        "timestamp": dt.datetime.now().isoformat(),
        "question": question,
        "retrieved": [
            {"source": c["source"], "score": c["score"], "text": c["text"]}
            for c in retrieved
        ],
        "answer": answer,
        "timings": {**retrieval_t, **gen_t},
    })


def score_retrieval(retrieved: list[dict], expected_keywords: list[str]) -> str:
    """PASS if every keyword appears in the retrieved context, PARTIAL if
    some do, FAIL if none do. Case-insensitive substring match, per spec.

    This scores *retrieval*, not the LLM answer - which is deliberate.
    A chunking failure means the right words never make it into the
    context in the first place; no amount of prompt-tuning can fix that,
    so a retrieval-only score isolates the failure mode being taught.
    """
    haystack = " ".join(c["text"] for c in retrieved).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in haystack)
    if hits == len(expected_keywords):
        return "PASS"
    if hits == 0:
        return "FAIL"
    return f"PARTIAL ({hits}/{len(expected_keywords)})"


def load_questions(path: str) -> list[dict]:
    """Load a QUESTIONS list from a Python file (test_questions3.py style)."""
    spec = importlib.util.spec_from_file_location("questions_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QUESTIONS


def run_question_suite(questions: list[dict], chunks: list[dict]) -> None:
    """Retrieve for every question, score against expected keywords, print
    a table. No LLM calls - the point is to show whether the *right text*
    survives chunking, not whether the model can paraphrase it.
    """
    print("\n" + "=" * 78)
    print(f"{'ID':<4} {'Trap':<5} {'Result':<20} Question")
    print("-" * 78)
    passes = 0
    total_retrieval_s = 0.0
    for q in questions:
        retrieved, retrieval_t = retrieve(q["question"], chunks)
        total_retrieval_s += retrieval_t["embed_question_s"] + retrieval_t["scan_s"]
        result = score_retrieval(retrieved, q["expected_keywords"])
        if result == "PASS":
            passes += 1
        trap = "yes" if q.get("trap") else ""
        preview = q["question"] if len(q["question"]) <= 44 else q["question"][:41] + "..."
        print(f"{q['id']:<4} {trap:<5} {result:<20} {preview}")
        log_interaction({
            "type": "suite",
            "timestamp": dt.datetime.now().isoformat(),
            "id": q["id"],
            "question": q["question"],
            "expected_keywords": q["expected_keywords"],
            "trap": bool(q.get("trap")),
            "result": result,
            "retrieved": [
                {"source": c["source"], "score": c["score"], "text": c["text"]}
                for c in retrieved
            ],
            "timings": retrieval_t,
        })
    print("-" * 78)
    print(f"{passes}/{len(questions)} passed  "
          f"(retrieval total: {total_retrieval_s:.1f}s, "
          f"avg {total_retrieval_s / len(questions) * 1000:.0f}ms/q)")
    print("=" * 78)


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument("--doc", default=DEFAULT_DOC, help="PDF to index (default: %(default)s)")
    corpus.add_argument("--qa", help="Curated Q&A JSON to use as the corpus instead of a PDF")

    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Characters per chunk when indexing a PDF (default: %(default)s)")
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help="Character overlap between consecutive chunks (default: %(default)s)")

    parser.add_argument("--questions", help="Path to a Python file exporting a QUESTIONS list; runs a scored batch and exits (non-interactive)")
    parser.add_argument("--show-chunks", action="store_true",
                        help="After indexing, dump every chunk to the console. Great for a live demo of what naive chunking actually produces (mid-word splits, orphaned fragments) vs a curated corpus.")

    return parser.parse_args()


def dump_chunks(chunks: list[dict]) -> None:
    """Print every chunk in the corpus with its source label and full text.

    On a naive-chunked PDF this is where the demo lands its punch: the
    audience can literally read the ...oxytoc | in-ergometrine... split
    in the wild, without having to trust the trap-question narration.
    """
    print("=" * 78)
    print(f"ALL {len(chunks)} INDEXED CHUNKS")
    print("=" * 78)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- chunk {i}/{len(chunks)}  [{chunk['source']}] ---")
        body = chunk.get("display_text", chunk["text"])
        for line in body.splitlines() or [""]:
            print(f"  {line}")
    print("\n" + "=" * 78 + "\n")


def interactive_loop(chunks: list[dict]) -> None:
    """Ask-and-answer REPL over the indexed corpus.

    Indexing happened once, up front; every question here is just an
    embed + cosine scan + LLM call, which is the fast part. Ctrl-D or
    an empty line exits.
    """
    print("Ready. Ask a question (empty line or Ctrl-D to quit).\n")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            break
        if not question:
            break
        run_single_question(question, chunks)
        print()


if __name__ == "__main__":
    args = parse_args()
    LOG_FILE = open_log(args)

    document_chunks = build_index(args)
    print(f"Indexed {len(document_chunks)} chunks.\n")

    if args.show_chunks:
        dump_chunks(document_chunks)

    if args.questions:
        run_question_suite(load_questions(args.questions), document_chunks)
    else:
        interactive_loop(document_chunks)
