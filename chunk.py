"""Milestone 3 — Chunking.

Splits ingested document text into chunks using a *deliberate, document-fit*
strategy rather than a blind fixed-width split. The UTEP mental-health corpus is
mixed, so chunking is adaptive per document type:

  - "prose" : prose handouts (depression, anxiety, stress, substance abuse) and
              service web pages. Boundary-aware packing of sentences up to a
              target size, with character overlap so guidance that spans
              paragraphs isn't split across a chunk boundary.
  - "entry" : self-contained records — community-referral directory entries,
              the online-resources link list, and individual Reddit comments.
              Each record is kept as a single chunk (preserves the complete
              entry and its attribution); only split if it exceeds the hard cap.

Hard constraint: the embedding model (all-MiniLM-L6-v2) truncates input at
256 tokens (~1000 chars). Anything beyond that is NOT embedded, so no chunk may
exceed `MAX_SIZE` or we silently lose text (e.g. the phone number at the end of
a referral entry). That ceiling is why the sizes below are what they are.
`MAX_SIZE` is a character approximation of the 256-token limit; the embedding
stage (Milestone 4) can additionally verify length in real tokens.

This module is decoupled from ingestion: it takes already-extracted text plus a
`doc_type` label, so it works the same whether the source was a .pdf, .html
page, or the Reddit thread. A `Chunk` is shaped to feed ChromaDB directly in
Milestone 4 (text -> document, the rest -> metadata).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# --- tunable defaults (see planning.md "Chunking Strategy" for the reasoning) ---
TARGET_SIZE = 800   # chars (~200 tokens): aim for prose chunks around here
MAX_SIZE = 1000     # chars (~256 tokens): hard ceiling = the model's input limit
OVERLAP = 200       # chars (~50 tokens): lead-in carried into the next prose chunk


@dataclass
class Chunk:
    """One chunk of text plus the metadata needed for retrieval + attribution."""
    text: str
    source: str        # human-readable source name (for citations)
    doc_id: str        # stable id of the originating document
    chunk_index: int   # position of this chunk within its document (0-based)
    char_start: int    # offset into the normalized document text
    char_end: int
    doc_type: str      # "prose" | "entry"


def normalize_whitespace(text: str) -> str:
    """Normalize line endings and collapse redundant whitespace.

    Per-line runs of spaces/tabs collapse to one space; 3+ blank lines collapse
    to a single paragraph break. This keeps paragraph structure (used as a
    chunk boundary) while removing noise from PDF extraction / scraped pages.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unit_spans(text: str) -> list[tuple[int, int]]:
    """Split text into contiguous (start, end) spans on sentence/paragraph
    boundaries. Spans cover the text in order, so any run of consecutive spans
    is exactly ``text[span[0].start : span[-1].end]`` — which is what lets chunk
    ranges (and their overlap) map back to a clean slice of the document.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    # Boundary = sentence-ending punctuation + whitespace, OR a blank line.
    for m in re.finditer(r"[.!?]+\s+|\n{2,}", text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for (s, e) in spans if text[s:e].strip()]


def _hard_split(start: int, end: int, max_size: int, overlap: int) -> list[tuple[int, int]]:
    """Last-resort char windowing for a single unit longer than max_size."""
    ranges: list[tuple[int, int]] = []
    step = max_size - max(overlap, 0)
    if step <= 0:
        step = max_size
    pos = start
    while pos < end:
        ranges.append((pos, min(pos + max_size, end)))
        if pos + max_size >= end:
            break
        pos += step
    return ranges


def _pack(spans: list[tuple[int, int]], target: int, max_size: int,
          overlap: int) -> list[tuple[int, int]]:
    """Greedily pack contiguous units into chunks of ~target chars (never over
    max_size), carrying ``overlap`` chars from each chunk's tail into the next.
    """
    ranges: list[tuple[int, int]] = []
    n = len(spans)
    i = 0
    while i < n:
        chunk_start = spans[i][0]
        j = i
        while j < n:
            length = spans[j][1] - chunk_start
            if j > i and length > target:
                break
            j += 1
        if j == i:               # always emit at least one unit
            j = i + 1
        chunk_end = spans[j - 1][1]

        if chunk_end - chunk_start > max_size:
            ranges.extend(_hard_split(chunk_start, chunk_end, max_size, overlap))
        else:
            ranges.append((chunk_start, chunk_end))

        if j >= n:
            break

        # Decide where the next chunk starts so it overlaps this one by ~overlap.
        if overlap > 0:
            k = j - 1
            while k - 1 > i and (chunk_end - spans[k - 1][0]) <= overlap:
                k -= 1
            next_i = k if k > i else j
        else:
            next_i = j
        i = next_i
    return ranges


def chunk_document(text: str, *, source: str, doc_id: str, doc_type: str = "prose",
                   target_size: int = TARGET_SIZE, max_size: int = MAX_SIZE,
                   overlap: int = OVERLAP) -> list[Chunk]:
    """Chunk a single document's text into a list of Chunk objects."""
    norm = normalize_whitespace(text)
    if not norm:
        return []

    if doc_type == "entry":
        # A self-contained record is one chunk; split only if oversized.
        if len(norm) <= max_size:
            ranges = [(0, len(norm))]
        else:
            ranges = _pack(_unit_spans(norm), target=max_size, max_size=max_size,
                           overlap=0)
    else:  # "prose" (default); unknown types fall back to prose behavior
        ranges = _pack(_unit_spans(norm), target=target_size, max_size=max_size,
                       overlap=overlap)

    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(ranges):
        body = norm[start:end].strip()
        if not body:
            continue
        chunks.append(Chunk(text=body, source=source, doc_id=doc_id,
                            chunk_index=idx, char_start=start, char_end=end,
                            doc_type=doc_type))
    return chunks


def chunk_corpus(documents: Iterable[tuple[str, str, str, str]]) -> list[Chunk]:
    """Chunk many documents.

    ``documents`` is an iterable of (text, source, doc_id, doc_type) tuples —
    the shape the ingestion stage (Milestone 3) will hand off. Returns a flat
    list of Chunk objects ready for the embedding stage (Milestone 4).
    """
    all_chunks: list[Chunk] = []
    for text, source, doc_id, doc_type in documents:
        all_chunks.extend(
            chunk_document(text, source=source, doc_id=doc_id, doc_type=doc_type)
        )
    return all_chunks


if __name__ == "__main__":
    # Smoke test + size histogram on samples shaped like the real UTEP corpus.
    handout = (  # prose: a depression/anxiety handout
        "Feeling down, hopeless, or overwhelmed for more than a couple of weeks "
        "is common among college students, and it is treatable. The first step "
        "is recognizing the signs: loss of interest in things you used to "
        "enjoy, trouble sleeping or sleeping too much, and difficulty "
        "concentrating on coursework.\n\n"
        "UTEP Counseling and Psychological Services offers free, confidential "
        "counseling to enrolled students. You can schedule an initial "
        "consultation by phone or by walking into the Union Building West. "
        "During that consultation a counselor will help you decide what kind of "
        "support fits your situation, whether that is short-term counseling, a "
        "group, or a referral.\n\n"
        "If you ever feel you might harm yourself, treat it as an emergency. "
        "Call or text 988 for the Suicide and Crisis Lifeline at any hour, or "
        "go to the nearest emergency room. You do not have to wait for an "
        "appointment to get help when it is urgent."
    )
    referral = (  # entry: one provider in the community-referral directory
        "Emergence Health Network — Crisis Services. 24/7 crisis line for El "
        "Paso County residents. Walk-in crisis clinic available. "
        "Phone: (915) 779-1800. Address: 1600 Montana Ave, El Paso, TX."
    )
    reddit = (  # entry: one Reddit comment
        "Honestly just walk into the counseling center, the online form took "
        "forever for me but when I showed up in person they got me an intake "
        "appointment the same week."
    )

    chunks = chunk_corpus([
        (handout, "Depression handout (PDF)", "depression_pdf", "prose"),
        (referral, "Community referral directory (PDF)", "referral_pdf", "entry"),
        (reddit, "r/UTEP thread", "reddit_thread", "entry"),
    ])

    print(f"Produced {len(chunks)} chunks\n")
    for c in chunks:
        print(f"[{c.doc_type:5}] {c.source:34} chunk {c.chunk_index} "
              f"({c.char_end - c.char_start} chars): {c.text[:55]}...")

    # --- assertions tied to the plan's verification section ---
    assert all(len(c.text) <= MAX_SIZE for c in chunks), "chunk exceeds MAX_SIZE!"

    entry_chunks = [c for c in chunks if c.doc_type == "entry"]
    assert len(entry_chunks) == 2, "each short entry should be exactly one chunk"

    prose_chunks = [c for c in chunks if c.doc_type == "prose"]
    assert len(prose_chunks) >= 2, "multi-paragraph handout should split"
    overlaps = [prose_chunks[i].char_start < prose_chunks[i - 1].char_end
                for i in range(1, len(prose_chunks))]
    assert any(overlaps), "expected overlap between prose chunks"

    assert all(c.source and c.doc_id and c.chunk_index is not None for c in chunks), \
        "missing metadata"

    print("\nAll assertions passed.")
