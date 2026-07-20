# ============================================================
# 07_prompting.py
# Grounded financial prompt + OpenRouter chat completion.
#
# HYBRID answering:
#   - If relevant passages are found in the local data, the model
#     answers ONLY from them and cites sources  -> a grounded RAG answer.
#   - If nothing relevant is found, the model answers from its own
#     general knowledge, and the app flags the answer as NOT grounded
#     in the stored sources.
# ============================================================

from importlib import import_module
import os

from dotenv import load_dotenv
from openai import OpenAI

build_context = import_module("06_retrieve_context").build_context

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def build_grounded_prompt(question, context):
    """Used when we DID find relevant passages: answer only from them."""
    return f"""You are a careful financial assistant answering questions about company filings (10-K reports).
Answer using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the context is not sufficient, say so clearly.
3. Cite the source numbers you used, like [Source 1].
4. Be concise and precise, and keep figures exactly as written.

Question:
{question}

Context:
{context}
"""


def build_general_prompt(question):
    """Used when NO relevant passage was found: answer from general knowledge."""
    return f"""You are a knowledgeable financial assistant.
The user's question was not found in the local filing database, so answer from your
own general knowledge. Be concise and accurate. If figures may be outdated or you are
unsure, say so briefly rather than inventing precise numbers.

Question:
{question}
"""


def ask_openrouter(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def _looks_like_no_answer(text):
    """Detect when the grounded model reply admits the context lacks the answer."""
    if not text:
        return True
    low = text.lower()
    signals = [
        "does not include",
        "does not contain",
        "not include information",
        "no information",
        "not enough information",
        "insufficient",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not found in the context",
        "not mentioned",
        "not provided",
        "not specified",
        "not available in the context",
    ]
    return any(s in low for s in signals)


def answer_question(index, question):
    """Return (answer, sources, grounded).

    grounded == True  -> answer came from the stored filings (real RAG).
    grounded == False -> answer came from the model's general knowledge.
    """
    context, sources = build_context(index, question)

    if not OPENROUTER_API_KEY:
        return "Missing OPENROUTER_API_KEY. Add it to your .env or Streamlit secrets.", sources, bool(context)

    # 1) If we retrieved relevant context, try a grounded answer first.
    if context:
        grounded_answer = ask_openrouter(build_grounded_prompt(question, context))

        # If the model says the context doesn't actually contain the answer,
        # fall back to general knowledge instead of dead-ending.
        if not _looks_like_no_answer(grounded_answer):
            return grounded_answer, sources, True

    # 2) Fallback: answer from the model's general knowledge (not grounded).
    general_answer = ask_openrouter(build_general_prompt(question))
    return general_answer, [], False
