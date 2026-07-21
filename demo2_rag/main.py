"""
Demo 2: Offline clinical RAG assistant.

Pipeline (each step is one clear, separate function):
  1. PDF -> text, split into small chunks (with page numbers kept for citations)
  2. Each chunk -> an embedding vector (one Ollama call per chunk, batched)
  3. A question -> its own embedding, compared against every chunk to find
     the most relevant ones (this is "retrieval")
  4. The retrieved chunks + the question -> an answer, with citations,
     using an Ollama chat call that is told to answer ONLY from the
     provided text and to say so if the answer isn't there

This is deliberately a from-scratch, in-memory RAG - no vector database.
That's the point: for a handful of short documents, a Python list and
cosine similarity are simple, easy to understand, and fast enough.

Run: python main.py
"""

import time

import numpy as np
import ollama
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4:e2b"

DOCUMENT_PATH = "documents/public_health_act.pdf"

# Simple fixed-size chunking. Real documents deserve smarter splitting
# (by section, by sentence boundary, etc.) but character chunks with a
# little overlap are enough to demonstrate the retrieval pattern.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Embed in batches rather than one huge call, so progress is visible and
# any single request stays a reasonable size.
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

SAMPLE_QUESTIONS = [
    "Who has a duty to notify infectious diseases under this Act?",
]


def load_document(path: str) -> list[dict]:
    """Step 1a: extract text from the PDF, one entry per page.

    We keep the page number attached to every page's text so that any
    chunk made from it can still be cited later.
    """
    reader = PdfReader(path)
    return [
        {"page": page_number, "text": page.extract_text()}
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Step 1b: split each page's text into overlapping chunks.

    A chunk keeps the page number of the page it came from, so the final
    answer can point back to "page N" instead of just trusting the model.
    """
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            chunk_text = text[start : start + CHUNK_SIZE].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "page": page["page"]})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Step 2: attach an embedding vector to every chunk.

    Chunks are embedded in batches (not one call per chunk, and not one
    giant call for everything) so progress is visible and each request
    stays a reasonable size.
    """
    start = time.time()
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        response = ollama.embed(
            model=EMBED_MODEL,
            input=[c["text"] for c in batch],
            keep_alive=KEEP_ALIVE,
        )
        for chunk, embedding in zip(batch, response["embeddings"]):
            chunk["embedding"] = np.array(embedding)
        done = batch_start + len(batch)
        print(f"  embedded {done}/{len(chunks)} chunks ({time.time() - start:.1f}s elapsed)")
    return chunks


def build_index(path: str) -> list[dict]:
    """Steps 1-2 combined: turn a PDF into a list of embedded chunks.

    This only needs to run once per document, not once per question.
    Each stage prints when it starts and how long it took, so a slow
    run - especially on a Pi - still shows it's making progress rather
    than looking stuck.
    """
    print(f"Loading {path}...")
    start = time.time()
    pages = load_document(path)
    print(f"  loaded {len(pages)} pages ({time.time() - start:.1f}s)")

    print("Splitting into chunks...")
    start = time.time()
    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks ({time.time() - start:.1f}s)")

    print(f"Embedding {len(chunks)} chunks in batches of {EMBED_BATCH_SIZE}...")
    return embed_chunks(chunks)


def retrieve(question: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Step 3: find the chunks most relevant to the question.

    Cosine similarity between the question's embedding and every chunk's
    embedding - simple linear scan, which is entirely fine for a few
    hundred chunks. A real vector database earns its place at a much
    larger scale than a workshop demo.
    """
    response = ollama.embed(model=EMBED_MODEL, input=question, keep_alive=KEEP_ALIVE)
    question_vector = np.array(response["embeddings"][0])

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored = [(cosine_similarity(question_vector, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def answer_question(question: str, retrieved_chunks: list[dict]) -> str:
    """Step 4: answer using only the retrieved text, with citations.

    The system prompt is explicit about the two failure modes we want to
    avoid: making things up, and citing pages that weren't actually used.

    stream=True prints each piece of the answer as it's generated instead
    of waiting for the whole response. It doesn't make generation faster,
    but on a slow model it's the difference between "is this even
    working?" and watching an answer actually being written.
    """
    context = "\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in retrieved_chunks
    )
    system_prompt = """You are a clinical reference assistant. Answer the
question using ONLY the provided document excerpts below. Cite the page
number(s) you used. If the excerpts don't contain the answer, say so
plainly instead of guessing.

Document excerpts:
""" + context

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
    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        full_answer += piece
    print()
    return full_answer.strip()


def run_pipeline(question: str, chunks: list[dict]) -> None:
    print("=" * 60)
    print("QUESTION")
    print(question)

    retrieved = retrieve(question, chunks)
    print("\nRETRIEVED CHUNKS")
    for chunk in retrieved:
        preview = chunk["text"][:80].replace("\n", " ")
        print(f"  page {chunk['page']}: {preview}...")

    print("\nANSWER")
    answer_question(question, retrieved)
    print("=" * 60)


if __name__ == "__main__":
    print(f"Indexing {DOCUMENT_PATH} (this only needs to happen once)...")
    document_chunks = build_index(DOCUMENT_PATH)
    print(f"Indexed {len(document_chunks)} chunks.\n")

    for question in SAMPLE_QUESTIONS:
        run_pipeline(question, document_chunks)
