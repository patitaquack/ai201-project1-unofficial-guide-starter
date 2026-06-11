"""Milestone 4 — Embedding + retrieval.

Embeds the cleaned chunks from the ingestion stage with a local
sentence-transformers model and stores them in a persistent ChromaDB
collection, then exposes retrieve(query, k) for semantic search.

Pipeline position:  ingest.py -> chunks.jsonl -> [this file] -> retrieval

Run:  python3 embed.py
      (first install deps: pip install -r requirements.txt)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install them with:\n"
        "    pip install -r requirements.txt"
    ) from exc

import chunk as chunker
import ingest

HERE = Path(__file__).parent
CHUNKS_JSONL = HERE / "chunks.jsonl"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "utep_mental_health"
MODEL_NAME = "all-MiniLM-L6-v2"

# Bridge our chunk metadata to the citation metadata we want to store.
# Maps each chunk's `source` label -> (source_url, source_type, topic).
# URLs come from planning.md; source_type uses the assignment's categories.
SOURCE_META: dict[str, tuple[str, str, str]] = {
    "TimelyCare virtual care":      ("https://timelycare.com/utep/",
                                     "official_page", "virtual care"),
    "UTEP Counseling Services":     ("https://www.utep.edu/student-affairs/counsel/resources/services-students.html",
                                     "official_page", "counseling services"),
    "UTEP Miner Support":           ("https://www.utep.edu/student-affairs/miner-support/",
                                     "official_page", "student support"),
    "Stress management handout":    ("https://www.utep.edu/student-affairs/counsel/_files/docs/stress-management.pdf",
                                     "handout_pdf", "stress management"),
    "Test anxiety handout":         ("https://www.utep.edu/student-affairs/counsel/_files/docs/test-anxiety.pdf",
                                     "handout_pdf", "test anxiety"),
    "Alcohol/drugs handout":        ("https://www.utep.edu/student-affairs/counsel/_files/docs/problem-with-alcohol-or-drugs.pdf",
                                     "handout_pdf", "substance use"),
    "Depression handout":           ("https://www.utep.edu/student-affairs/counsel/_files/docs/depression.pdf",
                                     "handout_pdf", "depression"),
    "Community referral directory": ("https://www.utep.edu/student-affairs/counsel/_files/docs/community-referral-book-2025-20261.pdf",
                                     "referral_list", "community referrals"),
    "Online counseling resources":  ("https://www.utep.edu/student-affairs/counsel/resources/on-line-resources.html",
                                     "official_page", "online resources"),
    "r/UTEP mental health thread":  ("https://www.reddit.com/r/UTEP/comments/t9la07/how_does_mental_health_services_work/",
                                     "forum", "student experiences"),
}


# --------------------------------------------------------------------------- #
# Lazy singletons so retrieve() can reuse the loaded model + db client.
# --------------------------------------------------------------------------- #
_model: SentenceTransformer | None = None
_client = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once. Runs locally, no API key needed."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_client():
    """A PersistentClient writes the vector index to disk (CHROMA_DIR) so the
    embeddings survive between runs."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection():
    # get_or_create_collection: fetch the collection if it exists, else make it.
    # hnsw:space=cosine -> cosine distance, the right metric for MiniLM vectors
    # (lower distance = more similar).
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------------------------------- #
# Build the index
# --------------------------------------------------------------------------- #
def load_chunks() -> list[dict]:
    """Load chunks from chunks.jsonl, or build them from documents/ if missing."""
    if CHUNKS_JSONL.exists():
        with CHUNKS_JSONL.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    docs = ingest.load_documents()
    return [c.__dict__ for c in chunker.chunk_corpus(docs)]


def build_index() -> int:
    """Embed every chunk and store it (text + metadata) in ChromaDB."""
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(
            "No chunks found. Add your sources to documents/ and run "
            "`python3 ingest.py` first."
        )

    texts = [c["text"] for c in chunks]
    # Embed all chunk texts locally with sentence-transformers.
    embeddings = get_model().encode(texts, show_progress_bar=True).tolist()

    ids: list[str] = []
    metadatas: list[dict] = []
    for i, c in enumerate(chunks):
        url, source_type, topic = SOURCE_META.get(
            c.get("source", ""), ("", c.get("doc_type", ""), c.get("source", ""))
        )
        ids.append(str(i))                       # chunk index as the id
        metadatas.append({                       # stored alongside each vector
            "source": c.get("source", ""),
            "source_url": url,
            "source_type": source_type,
            "topic": topic,
            "doc_id": c.get("doc_id", ""),
            "chunk_index": c.get("chunk_index", i),
        })

    collection = get_collection()
    # upsert = add new ids, overwrite existing ones — makes re-running idempotent
    # (plain .add() would raise on ids that are already present).
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings,
                      metadatas=metadatas)
    return len(ids)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def retrieve(query: str, k: int = 5) -> list[tuple[str, dict, float]]:
    """Embed the query and return the top-k chunks as (text, metadata, distance)."""
    # Embed the query with the same model used for the documents.
    query_embedding = get_model().encode([query]).tolist()
    # query: nearest-neighbour search over the stored embeddings.
    result = get_collection().query(query_embeddings=query_embedding, n_results=k)
    # Chroma returns each field as a list-per-query; we sent one query -> index 0.
    return list(zip(result["documents"][0],
                    result["metadatas"][0],
                    result["distances"][0]))


# --------------------------------------------------------------------------- #
# Test block
# --------------------------------------------------------------------------- #
TEST_QUERIES = [
    "I get anxious and forget everything during exams, does UTEP have help?",
    "Can I get free counseling at UTEP?",
    "How can I get mental health help without seeing someone in person?",
]


def run_query(query: str, k: int = 5) -> None:
    """Print the top-k results for one query."""
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    for text, meta, distance in retrieve(query, k=k):
        print(f"\n[distance {distance:.3f}] {meta['source']}  ({meta['topic']})")
        print(f"  source_url: {meta['source_url']}")
        print(f"  {text}")
    print()


def ensure_index(rebuild: bool = False) -> None:
    """Build the index if it's empty (or if rebuild is forced)."""
    collection = get_collection()
    if rebuild or collection.count() == 0:
        count = build_index()
        print(f"Indexed {count} chunks into '{COLLECTION_NAME}'.\n")
    else:
        print(f"Using existing index ({collection.count()} chunks). "
              "Pass --rebuild to re-embed.\n")


def main() -> None:
    # Usage:
    #   python3 embed.py                -> run the 3 built-in test queries
    #   python3 embed.py "your query"   -> run one query of your own
    #   python3 embed.py -i             -> interactive: type queries until blank
    #   add --rebuild to force re-embedding the chunks
    args = sys.argv[1:]
    rebuild = "--rebuild" in args
    args = [a for a in args if a != "--rebuild"]

    ensure_index(rebuild)

    if args and args[0] in ("-i", "--interactive"):
        print("Type a question (blank line to quit).")
        while True:
            try:
                query = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                break
            run_query(query)
    elif args:
        run_query(" ".join(args))
    else:
        for query in TEST_QUERIES:
            run_query(query)


if __name__ == "__main__":
    main()
