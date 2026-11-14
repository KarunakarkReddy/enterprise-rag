
import re

from rank_bm25 import BM25Okapi

from .vector_store import collection, semantic_search


# --------------------------------------------------
# TEXT TOKENIZATION
# --------------------------------------------------

def tokenize(text):
    """
    Convert text into lowercase words.
    Removes unnecessary punctuation.
    """

    if not isinstance(text, str):
        return []

    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# KEYWORD SEARCH USING BM25
# --------------------------------------------------

def keyword_search(question, top_k=5, document_id=None):
    """
    Retrieve document chunks based on keyword relevance.
    """

    if not isinstance(question, str) or not question.strip():
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ValueError("top_k must be a positive integer.")

    if top_k <= 0:
        return []

    search_args = {
        "include": ["documents", "metadatas"]
    }

    if document_id is not None:
        try:
            document_id = int(document_id)
        except (TypeError, ValueError):
            raise ValueError("Invalid document ID.")

        if document_id <= 0:
            raise ValueError("Document ID must be positive.")

        search_args["where"] = {
            "document_id": document_id
        }

    data = collection.get(**search_args)

    documents = data["documents"] or []
    ids = data["ids"] or []
    metadatas = data["metadatas"] or []

    if not documents:
        return []

    # Tokenize document chunks.
    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    # Ignore chunks containing no searchable words.
    valid_indices = [
        index
        for index, tokens in enumerate(tokenized_documents)
        if tokens
    ]

    if not valid_indices:
        return []

    valid_documents = [
        tokenized_documents[index]
        for index in valid_indices
    ]

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    # Build BM25 index.
    bm25 = BM25Okapi(valid_documents)

    scores = bm25.get_scores(query_tokens)

    # Rank chunks by keyword relevance.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        # Ignore chunks without keyword matches.
        if scores[index] <= 0:
            continue

        original_index = valid_indices[index]

        results.append({
            "id": ids[original_index],
            "text": documents[original_index],
            "metadata": metadatas[original_index],
            "keyword_score": float(scores[index])
        })

        if len(results) >= top_k:
            break

    return results


# --------------------------------------------------
# HYBRID SEARCH
# --------------------------------------------------

def hybrid_search(question, top_k=5, document_id=None):
    """
    Combine semantic search and keyword search
    using Reciprocal Rank Fusion (RRF).
    """

    if not isinstance(question, str) or not question.strip():
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ValueError("top_k must be a positive integer.")

    if top_k <= 0:
        return []

    # Retrieve additional candidates before ranking.
    candidate_count = max(top_k * 3, 10)

    # Semantic retrieval.
    semantic_results = semantic_search(
        question,
        top_k=candidate_count,
        document_id=document_id
    )

    # Keyword retrieval.
    keyword_results = keyword_search(
        question,
        top_k=candidate_count,
        document_id=document_id
    )

    scores = {}
    documents = {}

    # Reciprocal Rank Fusion.
    for results in [semantic_results, keyword_results]:

        for rank, item in enumerate(results):

            chunk_id = item["id"]

            rrf_score = 1 / (60 + rank + 1)

            scores[chunk_id] = (
                scores.get(chunk_id, 0) + rrf_score
            )

            # Keep the complete result metadata.
            documents[chunk_id] = item

    if not scores:
        return []

    # Sort by combined RRF score.
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    final_results = []

    for chunk_id in ranked_ids[:top_k]:

        item = documents[chunk_id].copy()

        item["rrf_score"] = scores[chunk_id]

        final_results.append(item)

    return final_results
