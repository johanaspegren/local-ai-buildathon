# Spec: fold the chunking-failure teaching demo into the existing RAG app

## Goal

We have a working RAG app in the dev folder already. Separately, a standalone
teaching demo was built (plain Python scripts, no dependency on the dev app)
that shows students a concrete, reproducible failure mode of naive PDF
chunking, then shows how curating clean Q&A units fixes it. The ask: bring
that *insight and content* into the existing app as a demo mode, rather than
keeping it as a disconnected side project — reusing the existing app's
retrieval/embedding code wherever possible instead of the standalone demo's
throwaway TF-IDF fallback.

**Before implementing:** read the existing app's structure first (chunking
strategy, embedding backend, vector store, how corpora are loaded) and adapt
this spec's approach to fit those conventions. Don't assume the standalone
demo's file layout maps 1:1 onto the existing app.

## The core pattern to replicate

Two parallel pipelines over the *same* source document, sharing identical
retrieval mechanics (embed -> cosine/vector similarity -> top-k), differing
only in the corpus:

1. **Naive**: raw PDF -> extract text -> fixed-size chunking (500 characters,
   no overlap, no regard for sentence/section/list boundaries) -> embed each
   chunk -> retrieve.
2. **Curated**: same retrieval code, but the corpus is a hand-written set of
   complete (question, answer) pairs distilled from the same source document
   -> embed each pair -> retrieve, return the paired answer.

A fixed set of test questions (a mix of plain lookups and deliberate "traps")
is run against both pipelines and each answer is scored automatically:
`PASS` if every expected keyword/phrase appears (case-insensitive substring
match) in the retrieved context, `PARTIAL` if some appear, `FAIL` if none do.

The trap questions are not synthetic — each corresponds to a *verified* naive
chunking failure, found by actually running the chunker on the real text and
inspecting where the 500-char cuts land, not by guessing. That's what makes
the demo credible live: the naive pipeline visibly and reproducibly gets
5/8 correct, the curated pipeline gets 8/8, every time.

## Two source documents, already prepared

Both are already extracted, chunked, and validated — ready to be dropped
into the existing app's data layer rather than re-sourced.

### Source A — IntechOpen chapter (general risk factors + mitigation)

Betta Edu, "Maternal Health in Sub-Saharan Africa," in *Maternal and Child
Health – A Holistic Approach to Equity, Nutrition, and Psychosocial
Well-Being*, IntechOpen, 2025. DOI: 10.5772/intechopen.1011062. **License:
CC BY 4.0** (fully open, attribution only).

Verified naive-chunking traps at chunk_size=500, overlap=0:
- A statistic ("268 times higher") split mid-word across a chunk boundary.
- A 7-item numbered list of mitigation strategies (subsections 3.2.1–3.2.7)
  spread across 3 chunks — top-k retrieval surfaces only part of it.
- A reference-list entry split mid-word, separating a citation's title from
  its journal name and year.

Naive: 5/8 correct. Curated (22 Q&A pairs): 8/8 correct.

### Source B — WHO maternal health recommendations (practical, concrete-situation guidance)

World Health Organization, *WHO recommendations on maternal health:
guidelines approved by the WHO Guidelines Review Committee*, 2nd edition,
2023. ISBN 978-92-4-008059-1 (electronic). **License: CC BY-NC-SA 3.0 IGO**
(attribution, noncommercial, share-alike — see Licensing note below).

This is a 185-page compilation of every current WHO maternal-health
recommendation, each formatted as a colored recommendation box (Strong/Weak
Recommendation + evidence-quality tag) followed by "Remarks" bullets from the
Guideline Development Group. A 10-page excerpt (pages 129–138: Section 2.1
Haemorrhage + Section 2.2.1 Hypertensive Disorders — pre-eclampsia and
eclampsia) was sliced directly out of the real document for the demo, so it's
authentic WHO text, not a stand-in.

Verified naive-chunking traps at chunk_size=500, overlap=0:
- The compound drug name "oxytocin-ergometrine" split mid-word
  (`...oxytoc` | `in-ergometrine...`).
- The phrase "uterotonics fail" split mid-word (`...other u` | `terotonics
  fail...`).
- A 4-item list of temporizing measures for postpartum haemorrhage due to
  uterine atony (intrauterine balloon tamponade, bimanual uterine
  compression, external aortic compression, non-pneumatic anti-shock
  garments) spread across 3 chunks — top-k=2 retrieval surfaces only 2 of 3.

Naive: 5/8 correct. Curated (21 Q&A pairs): 8/8 correct.

**Bonus observation worth carrying over even if not built into an auto-graded
question:** around WHO section 2.2.2/2.2.4, the raw text-extraction order
itself gets scrambled (likely a 2-column/callout-box PDF layout interleaving
unrelated sentences from adjacent boxes). This is a distinct failure mode
from a simple character-count split — it happens *before* chunking, at
extraction time, and naive chunking can't fix it. Manual curation sidesteps
it because a human reads the rendered page, not raw extraction order. Good
live talking point: "chunking isn't the only place PDFs betray you."

## What to actually build in the existing app

1. Add both source documents (or reuse the app's existing document-ingestion
   path) as a selectable "demo corpus."
2. Add a "naive chunking" mode using the existing app's chunker configured to
   fixed-size/no-overlap (or add this as an explicit option if the app
   normally does smarter chunking) — the point is to reproduce the *realistic
   default* a newcomer would reach for.
3. Add the curated Q&A datasets as an alternate corpus the existing retrieval
   code can query directly (each pair embedded as `question + " " + answer`,
   answer returned as the retrieved context — see below for the reusable
   files).
4. Add the fixed test-question sets with expected keywords and an automatic
   PASS/PARTIAL/FAIL scorer, plus a side-by-side comparison view/script
   (console table is fine, matching the existing app's UI idiom if it has
   one).
5. Confirm the same 5/8 vs 8/8 split reproduces with the app's *real*
   embedding backend (not just the throwaway TF-IDF fallback used in the
   standalone version) — if results differ, re-tune chunk size or question
   wording rather than assuming the pattern holds automatically.

## Reusable files (already built, ready to copy/adapt)

Location: `rag_demo/` inside the "Local AI Buildathon" project folder.

| File | What it gives you |
|---|---|
| `data/maternal_health_ssa.pdf`, `.txt` | Source A, already extracted |
| `data/qa_curated.json` | Source A's 22 curated (question, answer) pairs |
| `test_questions.py` | Source A's 8 test questions + expected keywords |
| `data/who_pph_preeclampsia_excerpt.pdf` | Source B's real 10-page excerpt |
| `data/who_maternal_health_recommendations_2023.pdf` | Full 185-page source B, for citation/reference |
| `data/qa3_curated.json` | Source B's 21 curated (question, answer) pairs |
| `test_questions3.py` | Source B's 8 test questions + expected keywords |
| `chunking.py`, `embedder.py` | Reference implementation (naive fixed-size chunker, TF-IDF fallback embedder) — useful as a spec of the *behavior* expected, not necessarily code to port verbatim into a different stack |

Point Claude Code at this folder as reference input; it doesn't need to reuse
the code, just the validated data and the verified trap findings.

## Licensing note

Source B (WHO) is CC BY-NC-SA 3.0 IGO — noncommercial and share-alike.
Fine for an internal teaching/buildathon demo; flag to Legal/Compliance
before any commercial packaging or external client redistribution of this
specific document or derivative content built from it.
