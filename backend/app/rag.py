
import ollama

from .retriever import hybrid_search
from .reranker import rerank_results
from .conversation_memory import get_history, save_message
from .guardrails import (
    validate_question,
    build_guarded_prompt,
    validate_answer,
    check_retrieval_confidence,
    verify_answer_grounding
)


def rewrite_followup_question(question, history):

    if not history:
        return question

    conversation = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in history
    )

    prompt = f"""
Convert the latest user question into a standalone
question using the conversation history.

Rules:
- Preserve the user's original intent.
- Resolve references such as "it", "that", and "first one".
- Do not answer the question.
- Return only the rewritten question.

Conversation:
{conversation}

Latest question:
{question}

Standalone question:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"].strip()


def generate_answer(question, session_id):

    question = validate_question(question)

    # Get previous conversation
    history = get_history(session_id)

    # Understand follow-up questions
    search_question = rewrite_followup_question(
        question, history
    )

    # Hybrid retrieval
    results = hybrid_search(search_question, top_k=5)

    # Reranking
    ranked_results = rerank_results(
        search_question, results, top_k=3
    )

    # Confidence checking
    ranked_results = check_retrieval_confidence(
        ranked_results, threshold=0.0
    )

    if not ranked_results:
        return {
            "answer": "I couldn't find sufficiently relevant information in the uploaded documents.",
            "sources": []
        }

    # Build context
    context = "\n\n".join(
        item["text"] for item in ranked_results
    )

    # Generate answer
    prompt = build_guarded_prompt(
        search_question, context
    )

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    answer = validate_answer(
        response["message"]["content"]
    )

    # Verify grounding
    is_grounded = verify_answer_grounding(
        answer, context
    )

    if not is_grounded:
        return {
            "answer": "I couldn't verify that the generated answer is supported by the uploaded documents.",
            "sources": []
        }

    # Save conversation
    save_message(session_id, question, answer)

    return {
        "answer": answer,
        "sources": [
            {
                "document_id": item["metadata"]["document_id"],
                "chunk_id": item["id"],
                "text": item["text"],
                "relevance_score": item["rerank_score"]
            }
            for item in ranked_results
        ]
    }
