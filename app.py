"""Milestone 5 — Grounded generation + Gradio interface.

Ties the pipeline together: retrieve() finds the most relevant chunks, those
chunks are injected into a tightly-grounded prompt, Groq's Llama 3.3 writes an
answer using ONLY that context, and the source URLs are attached
programmatically from the chunk metadata (never trusting the model to cite).

Run:  python3 app.py        (opens the Gradio app on localhost)
      python3 app.py --cli  (quick terminal test without the UI)

Requires GROQ_API_KEY in .env  (get a free key at https://console.groq.com).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency. Run: pip install -r requirements.txt") from exc

from embed import retrieve

load_dotenv()  # read GROQ_API_KEY from .env

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

# The exact fallback the system prompt must emit when context is insufficient.
INSUFFICIENT = ("I don't have enough information on that — please contact UTEP "
                "Counseling Services directly.")

# Hard grounding instructions. The model must not answer beyond the context.
SYSTEM_PROMPT = f"""You are a careful assistant that helps UTEP students find \
mental-health resources. You answer using ONLY the numbered context passages \
provided in the user's message.

STRICT RULES:
1. Use ONLY the information in the context. Never use outside knowledge or \
make assumptions.
2. If the context does not clearly contain the answer, reply with EXACTLY this \
sentence and nothing else:
{INSUFFICIENT}
3. When passages conflict, prefer official UTEP sources (source_type \
official_page, handout_pdf, or referral_list) over forum posts (source_type \
forum). Forum posts are student opinions and may be inaccurate.
4. Be concise, supportive, and specific. Include any phone numbers, addresses, \
or service names that appear in the context.
5. Never invent URLs, phone numbers, hours, or services that are not in the \
context. Do not append your own source list — that is added separately."""


def _build_context(results: list[tuple[str, dict, float]]) -> str:
    """Format retrieved chunks into a numbered context block, labeling each with
    its source_type so the model can apply the official-over-forum rule."""
    blocks = []
    for i, (text, meta, _distance) in enumerate(results, start=1):
        blocks.append(
            f"[{i}] (source_type: {meta.get('source_type', 'unknown')}; "
            f"source: {meta.get('source', '')})\n{text}"
        )
    return "\n\n".join(blocks)


def _collect_sources(results: list[tuple[str, dict, float]]) -> list[str]:
    """Unique source URLs from the retrieved chunks, in retrieval order."""
    seen: set[str] = set()
    urls: list[str] = []
    for _text, meta, _distance in results:
        url = meta.get("source_url", "")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def ask(question: str, k: int = TOP_K) -> dict:
    """Retrieve context, ask Groq to answer strictly from it, attach sources."""
    results = retrieve(question, k=k)
    if not results:
        return {"answer": INSUFFICIENT, "sources": []}

    context = _build_context(results)
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, following the strict rules."
    )

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # low temperature -> stays close to the context
    )
    answer = completion.choices[0].message.content.strip()

    # Attach sources programmatically from metadata. If the model said it
    # couldn't answer, don't show sources (they'd be misleading).
    sources = [] if answer == INSUFFICIENT else _collect_sources(results)
    return {"answer": answer, "sources": sources}


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
def launch_app() -> None:
    import gradio as gr

    def respond(question: str):
        if not question or not question.strip():
            return "", ""
        result = ask(question)
        sources = "\n".join(result["sources"]) if result["sources"] else "(none)"
        return result["answer"], sources

    with gr.Blocks(title="The Unofficial Guide — UTEP Mental Health") as demo:
        gr.Markdown(
            "# UTEP Mental Health — Unofficial Guide\n"
            "Answers are grounded only in UTEP counseling resources. "
            "For emergencies, call or text **988**."
        )
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. Can I get free counseling at UTEP?",
            lines=2,
        )
        ask_button = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=6)
        sources = gr.Textbox(label="Retrieved from", lines=5)

        ask_button.click(respond, inputs=question, outputs=[answer, sources])
        question.submit(respond, inputs=question, outputs=[answer, sources])

    demo.launch()  # serves on http://127.0.0.1:7860 (localhost)


# The 5 evaluation questions from planning.md (plus an off-domain control).
EVAL_QUESTIONS = [
    "I keep getting really anxious and forgetting everything I have studied "
    "during exams — does UTEP have resources for students like me?",
    "Can I get free counseling at UTEP?",
    "I have been feeling depressed, and I would like to know if UTEP offers "
    "anything to help me.",
    "I feel overwhelmed and have so much stress — what services does UTEP "
    "offer that may help with stress?",
    "If I don't want to see a counselor in person, how can I get help for my "
    "mental health as a UTEP student?",
    "What is the capital of France?",  # off-domain control -> should refuse
]


def _cli() -> None:
    """Run the planning.md evaluation questions and print answers + citations."""
    for q in EVAL_QUESTIONS:
        result = ask(q)
        print("=" * 80)
        print(f"Q: {q}")
        print(f"\nA: {result['answer']}")
        if result["sources"]:
            print("\nSources (cited from retrieved chunks):")
            for url in result["sources"]:
                print(f"  - {url}")
        else:
            print("\nSources: (none)")
        print()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        _cli()
    else:
        launch_app()
