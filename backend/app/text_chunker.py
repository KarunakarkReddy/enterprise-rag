
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_text(text):
    """
    Split extracted PDF text into smaller overlapping chunks.

    Smaller chunks improve retrieval accuracy.
    Overlap helps preserve context between neighboring chunks.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        length_function=len,
        is_separator_regex=False
    )

    chunks = splitter.split_text(text)

    # Remove empty chunks
    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]
