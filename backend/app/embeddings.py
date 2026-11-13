from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(text_chunks):
    embeddings = model.encode(text_chunks)
    return embeddings.tolist()


def generate_query_embedding(question):
    embedding = model.encode([question])
    return embedding[0].tolist()