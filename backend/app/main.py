
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pypdf import PdfReader

from .database import get_db, Base, engine
from .models import Document, DocumentChunk
from .document_loader import extract_text_from_pdf
from .text_chunker import split_text
from .vector_store import (
    store_chunks,
    delete_document_vectors
)
from .rag import generate_answer


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG Application",
    description="PDF-based Retrieval Augmented Generation system",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024

# Create database tables if they do not exist.
Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000
    )

    session_id: str = Field(
        min_length=1,
        max_length=100
    )

    document_id: int | None = Field(
        default=None,
        gt=0
    )


# --------------------------------------------------
# HELPER: CLEAN UP FAILED UPLOADS
# --------------------------------------------------

def cleanup_failed_upload(file_path, document_id, db):
    """
    Remove partially stored data after upload failure.
    """

    db.rollback()

    # Remove any vectors already stored in ChromaDB.
    if document_id is not None:
        try:
            delete_document_vectors(document_id)
        except Exception:
            logger.exception(
                "Could not clean up vectors for document %s",
                document_id
            )

    # Remove the uploaded PDF.
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            logger.exception(
                "Could not remove uploaded PDF: %s",
                file_path
            )


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)


@app.get("/")
def home():
    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found"
        )

    return FileResponse(index_file)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    original_filename = file.filename or "uploaded.pdf"

    # Keep only the filename, not any supplied directory path.
    original_filename = Path(
        original_filename.replace("\\", "/")
    ).name

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    file_content = await file.read(MAX_UPLOAD_SIZE + 1)

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )

    if len(file_content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF size must not exceed 20 MB"
        )

    if b"%PDF-" not in file_content[:1024]:
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file"
        )

    # Use a unique name to avoid overwriting uploaded files.
    unique_filename = f"{uuid4()}.pdf"
    file_path = UPLOAD_DIR / unique_filename

    document_id = None

    try:
        # Save the PDF.
        file_path.write_bytes(file_content)

        # Extract text from the PDF.
        extracted_text = extract_text_from_pdf(file_path)

        if not extracted_text or not extracted_text.strip():
            raise ValueError(
                "No extractable text found in PDF. "
                "Scanned PDFs require OCR."
            )

        # Read the actual number of PDF pages.
        pdf_reader = PdfReader(str(file_path))
        total_pages = len(pdf_reader.pages)

        # Split extracted text into chunks.
        chunks = split_text(extracted_text)

        if not chunks:
            raise ValueError(
                "Could not generate text chunks from PDF."
            )

        # Create the document record.
        new_document = Document(
            filename=original_filename,
            file_path=str(file_path),
            status="processed",
            total_pages=total_pages,
            total_chunks=len(chunks)
        )

        db.add(new_document)
        db.flush()

        document_id = new_document.id

        # Generate embeddings and store chunks in ChromaDB.
        store_chunks(
            chunks,
            document_id
        )

        # Store the same chunks in MySQL.
        for index, chunk in enumerate(chunks):
            db.add(
                DocumentChunk(
                    document_id=document_id,
                    chunk_text=chunk,
                    chunk_index=index
                )
            )

        # Commit MySQL changes.
        db.commit()

        logger.info(
            "PDF processed successfully. Document ID: %s",
            document_id
        )

        return {
            "message": "PDF processed successfully",
            "document_id": document_id,
            "filename": original_filename,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "status": "processed"
        }

    except HTTPException:
        cleanup_failed_upload(
            file_path,
            document_id,
            db
        )
        raise

    except ValueError as error:
        cleanup_failed_upload(
            file_path,
            document_id,
            db
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        cleanup_failed_upload(
            file_path,
            document_id,
            db
        )

        logger.exception("PDF processing failed.")

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF processing failed. "
                "Please check the server logs."
            )
        ) from error

    finally:
        await file.close()


# --------------------------------------------------
# CHAT / QUESTION ANSWERING
# --------------------------------------------------

@app.post("/chat")
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    # Check whether the selected document exists.
    if request.document_id is not None:

        document = (
            db.query(Document)
            .filter(
                Document.id == request.document_id
            )
            .first()
        )

        if not document:
            raise HTTPException(
                status_code=404,
                detail="Selected document not found"
            )

    try:
        response = generate_answer(
            request.question,
            request.session_id,
            request.document_id
        )

        source_document_ids = {
            source.get("document_id")
            for source in response.get("sources", [])
            if isinstance(source.get("document_id"), int)
        }

        if source_document_ids:
            documents = (
                db.query(Document)
                .filter(Document.id.in_(source_document_ids))
                .all()
            )
            filenames = {
                document.id: document.filename
                for document in documents
            }

            for source in response["sources"]:
                source["filename"] = filenames.get(
                    source.get("document_id")
                )

        return response

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception("Chat processing failed.")

        raise HTTPException(
            status_code=500,
            detail="Could not process your question."
        ) from error


# --------------------------------------------------
# LIST UPLOADED DOCUMENTS
# --------------------------------------------------

@app.get("/documents")
def list_documents(
    db: Session = Depends(get_db)
):
    documents = (
        db.query(Document)
        .order_by(desc(Document.created_at))
        .all()
    )

    return [
        {
            "id": document.id,
            "filename": document.filename,
            "status": document.status,
            "total_pages": document.total_pages,
            "total_chunks": document.total_chunks,
            "created_at": document.created_at
        }
        for document in documents
    ]


# --------------------------------------------------
# DELETE DOCUMENT
# --------------------------------------------------

@app.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    file_path = Path(document.file_path)

    try:
        # Delete vectors from ChromaDB.
        delete_document_vectors(document_id)

        # Delete chunks from MySQL.
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete(synchronize_session=False)

        # Delete the document record.
        db.delete(document)
        db.commit()

    except Exception as error:
        db.rollback()

        logger.exception(
            "Document deletion failed: %s",
            document_id
        )

        raise HTTPException(
            status_code=500,
            detail="Could not delete document data"
        ) from error

    # Remove the physical PDF after database deletion.
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            logger.exception(
                "Could not remove uploaded PDF: %s",
                file_path
            )

    return {
        "message": "Document deleted successfully",
        "document_id": document_id
    }
