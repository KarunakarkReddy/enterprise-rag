from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from .database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(50), default="pending")
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)