
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load the reranking model once.
reranker_model = CrossEncoder(MODEL_NAME)


# --------------------------------------------------
# RERANK RETRIEVED CHUNKS
# --------------------------------------------------

def rerank_results(question, results, top_k=3):
    """
    Reorder retrieved chunks according to their
    relevance to the user's question.

    Returns:
        list: Highest-ranked document chunks.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not results:
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ValueError("top_k must be a positive integer.")

    if top_k <= 0:
        return []

    # Keep only valid chunks.
    valid_results = [
        item for item in results
        if isinstance(item.get("text"), str)
        and item["text"].strip()
    ]

    if not valid_results:
        return []

    # Create question and document pairs.
    pairs = [
        [question.strip(), item["text"]]
        for item in valid_results
    ]

    # Calculate relevance scores.
    scores = reranker_model.predict(
        pairs,
        show_progress_bar=False
    )

    if len(scores) != len(valid_results):
        raise RuntimeError(
            "Reranker returned an unexpected number of scores."
        )

    # Attach scores to retrieved chunks.
    ranked_results = []

    for item, score in zip(valid_results, scores):
        result = item.copy()
        result["rerank_score"] = float(score)
        ranked_results.append(result)

    # Highest relevance score first.
    ranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked_results[:top_k]
