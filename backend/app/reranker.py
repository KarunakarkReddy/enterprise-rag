from sentence_transformers import CrossEncoder

# Load reranking model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(question, results, top_k=3):

    if not results:
        return []

    # Create question and document pairs
    pairs = [
        [question, item["text"]]
        for item in results
    ]

    # Calculate relevance scores
    scores = reranker_model.predict(pairs)

    # Attach scores to results
    for i, score in enumerate(scores):
        results[i]["rerank_score"] = float(score)

    # Sort by relevance score
    results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return results[:top_k]