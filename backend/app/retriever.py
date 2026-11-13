from rank_bm25 import BM25Okapi

from .vector_store import collection, semantic_search


def keyword_search(question, top_k=5):

    # Get all stored chunks
    data = collection.get(include=["documents", "metadatas"])

    documents = data["documents"]
    ids = data["ids"]
    metadatas = data["metadatas"]

    if not documents:
        return []

    # Tokenize document chunks
    tokenized_documents = [
        doc.lower().split() for doc in documents
    ]

    # Create BM25 search model
    bm25 = BM25Okapi(tokenized_documents)

    # Search using question keywords
    scores = bm25.get_scores(question.lower().split())

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        {
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i]
        }
        for i in ranked_indices
    ]


def hybrid_search(question, top_k=5):

    semantic_results = semantic_search(question, top_k)
    keyword_results = keyword_search(question, top_k)

    # Combine results using Reciprocal Rank Fusion
    scores = {}
    documents = {}

    for results in [semantic_results, keyword_results]:

        for rank, item in enumerate(results):
            doc_id = item["id"]

            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (60 + rank + 1)
            )

            documents[doc_id] = item

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids[:top_k]
    ]