
from pathlib import Path

import chromadb

from .embeddings import (
    generate_embeddings,
    generate_query_embedding
)


# --------------------------------------------------
# CHROMADB CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="enterprise_documents"
)

CHROMA_BATCH_SIZE = 500


# --------------------------------------------------
# VALIDATION HELPER
# --------------------------------------------------

def validate_document_id(document_id):
    """Validate and convert a document ID."""

    try:
        document_id = int(document_id)
    except (TypeError, ValueError):
        raise ValueError("Invalid document ID.")

    if document_id <= 0:
        raise ValueError("Document ID must be positive.")

    return document_id


# --------------------------------------------------
# STORE DOCUMENT CHUNKS
# --------------------------------------------------

def store_chunks(chunks, document_id):
    """
    Generate embeddings and store document chunks in ChromaDB.
    """

    document_id = validate_document_id(document_id)

    if not chunks:
        return 0

    if not all(
        isinstance(chunk, str) and chunk.strip()
        for chunk in chunks
    ):
        raise ValueError("All chunks must contain valid text.")

    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        batch = chunks[start:start + CHROMA_BATCH_SIZE]
        batch_end = start + len(batch)
        embeddings = generate_embeddings(batch)

        # Upsert inserts new chunks or updates existing IDs.
        collection.upsert(
            ids=[
                f"doc_{document_id}_chunk_{index}"
                for index in range(start, batch_end)
            ],
            documents=batch,
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": document_id,
                    "chunk_index": index
                }
                for index in range(start, batch_end)
            ]
        )

    return len(chunks)


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def semantic_search(question, top_k=5, document_id=None):
    """
    Find relevant document chunks using embedding similarity.
    """

    if not isinstance(question, str) or not question.strip():
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ValueError("top_k must be a positive integer.")

    if top_k <= 0:
        return []

    if document_id is not None:
        document_id = validate_document_id(document_id)

    query_embedding = generate_query_embedding(question)

    search_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas"]
    }

    if document_id is not None:
        search_args["where"] = {
            "document_id": document_id
        }

    results = collection.query(**search_args)

    search_results = []

    if not results["ids"] or not results["ids"][0]:
        return []

    for index, chunk_id in enumerate(results["ids"][0]):
        search_results.append({
            "id": chunk_id,
            "text": results["documents"][0][index],
            "metadata": results["metadatas"][0][index]
        })

    return search_results


# --------------------------------------------------
# DELETE DOCUMENT VECTORS
# --------------------------------------------------

def delete_document_vectors(document_id):
    """
    Delete all ChromaDB chunks belonging to a document.
    """

    document_id = validate_document_id(document_id)

    collection.delete(
        where={"document_id": document_id}
    )


# --------------------------------------------------
# GET ALL CHUNKS OF A DOCUMENT
# --------------------------------------------------

def get_all_document_chunks(document_id):
    """
    Retrieve every chunk of a document in original order.
    """

    document_id = validate_document_id(document_id)

    results = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"]
    )

    chunks = []

    for chunk_id, text, metadata in zip(
        results["ids"],
        results["documents"],
        results["metadatas"]
    ):
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata
        })

    return sorted(
        chunks,
        key=lambda item: item["metadata"]["chunk_index"]
    )


# --------------------------------------------------
# GET NEIGHBORING CHUNKS
# --------------------------------------------------

def get_neighboring_chunks(results, window=1):
    """
    Expand retrieved chunks with nearby chunks from
    the same document.

    Example:
    Retrieved chunk 5 with window=1 returns chunks 4, 5 and 6.
    """

    if not results:
        return []

    try:
        window = int(window)
    except (TypeError, ValueError):
        raise ValueError("Window must be a non-negative integer.")

    if window < 0:
        raise ValueError("Window cannot be negative.")

    # Group original results by document.
    document_ids = set()

    for item in results:
        metadata = item.get("metadata", {})

        if "document_id" not in metadata:
            raise ValueError("Retrieved chunk has no document ID.")

        document_ids.add(
            validate_document_id(metadata["document_id"])
        )

    expanded_chunks = {}

    # Retrieve all chunks for each relevant document.
    for document_id in document_ids:

        all_chunks = get_all_document_chunks(document_id)

        chunk_map = {
            item["metadata"]["chunk_index"]: item
            for item in all_chunks
        }

        # Expand every retrieved chunk.
        for original_item in results:

            metadata = original_item["metadata"]

            if int(metadata["document_id"]) != document_id:
                continue

            chunk_index = int(metadata["chunk_index"])

            start_index = max(0, chunk_index - window)
            end_index = chunk_index + window

            for nearby_index in range(
                start_index,
                end_index + 1
            ):

                if nearby_index not in chunk_map:
                    continue

                chunk = chunk_map[nearby_index].copy()

                # Preserve scores and other fields from
                # the original retrieved result.
                if chunk["id"] == original_item["id"]:
                    chunk.update(original_item)

                expanded_chunks[chunk["id"]] = chunk

    # Return chunks in document and page order.
    return sorted(
        expanded_chunks.values(),
        key=lambda item: (
            item["metadata"]["document_id"],
            item["metadata"]["chunk_index"]
        )
    )
