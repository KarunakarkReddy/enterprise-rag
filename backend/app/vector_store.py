import chromadb

from .embeddings import generate_embeddings, generate_query_embedding

# Create persistent ChromaDB storage
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="enterprise_documents"
)


def store_chunks(chunks, document_id):
    embeddings = generate_embeddings(chunks)

    ids = [
        f"doc_{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
    )

    return len(chunks)


def semantic_search(question, top_k=5):
    query_embedding = generate_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    search_results = []

    for i in range(len(results["ids"][0])):
        search_results.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i]
        })

    return search_results