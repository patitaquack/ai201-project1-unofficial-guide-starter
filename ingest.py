"""Milestone 3 — Ingestion + cleaning.

Loads the raw documents in ``documents/``, cleans the extracted text, and runs
them through ``chunk.py`` to produce chunks matching the planning.md spec
(~200-token target, hard 256-token / ~1000-char cap, ~50-token overlap on prose,
one chunk per self-contained entry).

Supported file types:
  - .pdf            -> text via pdfplumber (the UTEP handouts + referral book)
  - .html / .htm    -> tags stripped via the stdlib html.parser (web pages)
  - .txt / .md      -> read directly (e.g. a pasted Reddit thread)

Each file is mapped (in SOURCES below) to a human-readable source label and a
mode:
  - "prose"      : a handout / article -> chunked with overlap.
  - "collection" : many self-contained records in one file (a referral
                   directory, a link list, a thread of comments) -> split into
                   records first, each emitted as one "entry" document so the
                   chunker keeps it whole.

Output: a flat list of Chunk objects, also written to ``chunks.jsonl`` for the
embedding stage (Milestone 4) to load.

Run:  python3 ingest.py
"""

from __future__ import annotations

import html
import json
import random
import re
from html.parser import HTMLParser
from pathlib import Path

import chunk as chunker  # local chunk.py

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNKS_OUT = Path(__file__).parent / "chunks.jsonl"

# Map each expected filename -> (source label for citations, mode).
# Save each source from planning.md into documents/ under the matching name.
# Files not listed here default to ("<filename>", "prose").
SOURCES: dict[str, tuple[str, str]] = {
    "timelycare.html":            ("TimelyCare virtual care",            "prose"),
    "counseling-services.html":   ("UTEP Counseling Services",           "prose"),
    "miner-support.html":         ("UTEP Miner Support",                 "prose"),
    "reddit-thread.txt":          ("r/UTEP mental health thread",        "collection"),
    "community-referral-book.pdf":("Community referral directory",       "collection"),
    "depression.pdf":             ("Depression handout",                 "prose"),
    "test-anxiety.pdf":           ("Test anxiety handout",               "prose"),
    "alcohol-or-drugs.pdf":       ("Alcohol/drugs handout",              "prose"),
    "stress-management.pdf":      ("Stress management handout",          "prose"),
    "online-resources.html":      ("Online counseling resources",        "collection"),
}

# Records within a "collection" file are separated by a blank line.
RECORD_SEPARATOR = re.compile(r"\n\s*\n")


# --------------------------------------------------------------------------- #
# Text extraction (one helper per file type)
# --------------------------------------------------------------------------- #
class _HTMLToText(HTMLParser):
    """Collect visible text from HTML, dropping script/style and inserting line
    breaks around block-level elements so paragraphs survive."""
    _SKIP = {"script", "style", "head", "noscript"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "ul", "ol", "header", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    return parser.text()


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on env
        raise SystemExit(
            "pdfplumber is required to read PDFs. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from exc
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def extract_text(path: Path) -> str:
    """Extract raw text from a document based on its file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in {".html", ".htm"}:
        return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    # .txt, .md, and anything else: read as plain text
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Clean extracted text before chunking.

    Fixes common PDF/HTML extraction artifacts: HTML entities (&amp;, &#39;,
    &nbsp;), non-breaking/zero-width characters, hyphenated words broken across
    lines, lines that are just a page number, and runs of blank lines. Final
    whitespace normalization is left to chunk.normalize_whitespace().
    """
    text = html.unescape(text)                            # &amp; &#39; &nbsp; -> & ' \xa0
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("​", "")  # nbsp, zero-width space
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate line breaks
    lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\s*\d+\s*", ln)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)                # collapse blank-line runs
    return text.strip()


# --------------------------------------------------------------------------- #
# Loading -> (text, source, doc_id, doc_type) tuples for chunk_corpus()
# --------------------------------------------------------------------------- #
def load_documents(documents_dir: Path = DOCUMENTS_DIR,
                   sources: dict[str, tuple[str, str]] = SOURCES
                   ) -> list[tuple[str, str, str, str]]:
    docs: list[tuple[str, str, str, str]] = []
    for path in sorted(documents_dir.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        label, mode = sources.get(path.name, (path.stem, "prose"))
        text = clean_text(extract_text(path))
        if not text:
            continue
        doc_id = path.stem
        if mode == "collection":
            records = [r.strip() for r in RECORD_SEPARATOR.split(text) if r.strip()]
            for i, record in enumerate(records):
                docs.append((record, label, f"{doc_id}#{i}", "entry"))
        else:
            docs.append((text, label, doc_id, "prose"))
    return docs


# --------------------------------------------------------------------------- #
# Output + reporting
# --------------------------------------------------------------------------- #
def write_jsonl(chunks: list[chunker.Chunk], path: Path = CHUNKS_OUT) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")


def report(chunks: list[chunker.Chunk]) -> None:
    print(f"Produced {len(chunks)} chunks")
    by_type: dict[str, int] = {}
    sizes = [len(c.text) for c in chunks]
    for c in chunks:
        by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:6}: {n}")
    if sizes:
        print(f"  chunk size chars  min={min(sizes)} "
              f"avg={sum(sizes)//len(sizes)} max={max(sizes)} "
              f"(cap={chunker.MAX_SIZE})")
    over = [c for c in chunks if len(c.text) > chunker.MAX_SIZE]
    if over:
        print(f"  WARNING: {len(over)} chunk(s) exceed the cap!")
    empty = [c for c in chunks if not c.text.strip()]
    if empty:
        print(f"  WARNING: {len(empty)} empty chunk(s)!")
    # Milestone 3 sanity range: ~50-2000 chunks across ~10 documents.
    n = len(chunks)
    if 0 < n < 50:
        print(f"  NOTE: only {n} chunks (<50) — chunks may be too large; "
              "specific queries won't match precisely.")
    elif n > 2000:
        print(f"  NOTE: {n} chunks (>2000) — chunks may be too small; "
              "each embedding carries little meaning.")


def inspect_chunks(chunks: list[chunker.Chunk], n: int = 5, seed: int = 0) -> None:
    """Print n random chunks in full so you can READ them (Milestone 3 checkpoint).

    For each, ask: is it readable, substantive, and self-contained? Could someone
    answer a question from this chunk alone? Watch for fragments, leftover HTML,
    or empty strings — bad chunks can't be fixed by tuning retrieval later.
    """
    if not chunks:
        print("\n(no chunks to inspect)")
        return
    rng = random.Random(seed)
    sample = chunks if len(chunks) <= n else rng.sample(chunks, n)
    print(f"\n=== Inspect {len(sample)} chunks (read each one) ===")
    for c in sample:
        print(f"\n--- {c.source} | doc={c.doc_id} | chunk #{c.chunk_index} | "
              f"{c.doc_type} | {len(c.text)} chars ---")
        print(c.text)


def main() -> None:
    if not DOCUMENTS_DIR.exists() or not any(
        p for p in DOCUMENTS_DIR.iterdir() if not p.name.startswith(".")
    ):
        print("No documents found in documents/.")
        print("Add your sources (see SOURCES in this file for filenames), then "
              "re-run. Running a self-test on synthetic files instead.\n")
        _selftest()
        return

    docs = load_documents()
    chunks = chunker.chunk_corpus(docs)
    write_jsonl(chunks)
    report(chunks)
    inspect_chunks(chunks)   # checkpoint: READ these before embedding
    print(f"\nWrote {CHUNKS_OUT.name} — ready for the embedding stage.")


# --------------------------------------------------------------------------- #
# Self-test (runs when documents/ is empty; needs no real files or pdfplumber)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    handout = (
        "Feeling overwhelmed is common, and it is treatable. The first step is "
        "noticing the signs: trouble sleeping, loss of interest, and difficulty "
        "concentrating on coursework over a sustained period of time.\n\n"
        "UTEP Counseling and Psychological Services offers free, confidential "
        "counseling to enrolled students. You can schedule an initial "
        "consultation by phone or by walking into the Union Building West, where "
        "a counselor will help you decide what kind of support fits you best.\n\n"
        "If you ever feel you might harm yourself, treat it as an emergency and "
        "call or text 988 for the Suicide and Crisis Lifeline at any hour."
    )
    directory = (
        "Emergence Health Network. 24/7 crisis line. Phone: (915) 779-1800.\n\n"
        "Family Service of El Paso. Sliding-scale counseling. Phone: (915) 532-9485.\n\n"
        "Aliviane Inc. Substance use treatment. Phone: (915) 782-4000."
    )
    page_html = (
        "<html><head><style>.x{color:red}</style></head><body>"
        "<h1>Online Resources</h1>"
        "<p>TimelyCare gives students 24/7 virtual access to licensed "
        "counselors&nbsp;&amp; psychiatrists at no cost. It&#39;s free.</p>"
        "<script>console.log('ignore me')</script>"
        "</body></html>"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "handout.txt").write_text(handout)
        (tmpdir / "directory.txt").write_text(directory)
        (tmpdir / "page.html").write_text(page_html)
        manifest = {
            "handout.txt":   ("Handout", "prose"),
            "directory.txt": ("Directory", "collection"),
            "page.html":     ("Web page", "prose"),
        }
        docs = load_documents(tmpdir, manifest)
        chunks = chunker.chunk_corpus(docs)

    report(chunks)
    inspect_chunks(chunks)

    # --- assertions ---
    assert all(len(c.text) <= chunker.MAX_SIZE for c in chunks), "chunk over cap!"

    entry_chunks = [c for c in chunks if c.doc_type == "entry"]
    assert len(entry_chunks) == 3, "directory should split into 3 entry chunks"
    assert all(c.text.count("Phone:") == 1 for c in entry_chunks), \
        "each entry should be one self-contained provider"

    prose_chunks = [c for c in chunks if c.doc_type == "prose"]
    assert len(prose_chunks) >= 2, "handout should split into multiple prose chunks"

    web = [c for c in chunks if c.source == "Web page"]
    web_text = " ".join(c.text for c in web)
    assert web and "<" not in web_text and "console.log" not in web_text, \
        "HTML tags / scripts not stripped"
    assert not any(e in web_text for e in ("&amp;", "&nbsp;", "&#39;")), \
        "HTML entities not unescaped"
    assert "&" in web_text and "It's free" in web_text, "entities mis-decoded"

    print("\nSelf-test assertions passed.")


if __name__ == "__main__":
    main()
