from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
from uuid import uuid4

from .database import get_db, Base, engine
from .models import Document, DocumentChunk
from .document_loader import extract_text_from_pdf
from .text_chunker import split_text
from .vector_store import store_chunks
from pydantic import BaseModel
from .rag import generate_answer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Enterprise RAG Application")
class ChatRequest(BaseModel):
    question: str
    session_id: str

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)

# Create database tables
Base.metadata.create_all(bind=engine)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_content = await file.read()

    if not file_content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    unique_filename = f"{uuid4()}.pdf"
    file_path = UPLOAD_DIR / unique_filename
    file_path.write_bytes(file_content)

    try:
        # Extract PDF text
        extracted_text = extract_text_from_pdf(file_path)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in PDF"
            )

        # Split text into chunks
        chunks = split_text(extracted_text)
        

        # Save document details
        new_document = Document(
            filename=file.filename,
            file_path=str(file_path),
            status="processed",
            total_chunks=len(chunks)
        )

        db.add(new_document)
        db.flush()

        # Store chunks and embeddings in ChromaDB
        store_chunks(chunks, new_document.id)

        # Save chunks in MySQL
        for index, chunk in enumerate(chunks):
            db.add(
                DocumentChunk(
                    document_id=new_document.id,
                    chunk_text=chunk,
                    chunk_index=index
                )
            )

        db.commit()
        db.refresh(new_document)

        return {
            "message": "PDF processed successfully!",
            "document_id": new_document.id,
            "filename": new_document.filename,
            "total_chunks": new_document.total_chunks,
            "status": new_document.status
        }

    except Exception:
        db.rollback()
        if file_path.exists():
            file_path.unlink()
        raise
@app.post("/chat")
def chat(request: ChatRequest):
    return generate_answer(
        request.question,
        request.session_id
    )