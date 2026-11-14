
import logging

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

# Load the embedding model once when the application starts.
model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# GENERATE DOCUMENT EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(text_chunks):
    """
    Convert document text chunks into numerical vectors.

    Returns:
        list: Embedding vectors for all chunks.
    """

    if not text_chunks:
        return []

    if not all(
        isinstance(chunk, str) and chunk.strip()
        for chunk in text_chunks
    ):
        raise ValueError(
            "Every text chunk must contain valid text."
        )

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )

    logger.info(
        "Generated embeddings for %s text chunks",
        len(text_chunks)
    )

    return embeddings.tolist()


# --------------------------------------------------
# GENERATE QUESTION EMBEDDING
# --------------------------------------------------

def generate_query_embedding(question):
    """
    Convert a user's question into a numerical vector.

    Returns:
        list: Embedding vector for the question.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding = model.encode(
        [question.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embedding[0].tolist()
