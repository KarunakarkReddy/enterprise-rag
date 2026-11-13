import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker

# Find and load the root .env file
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path, override=True)

# Read database credentials separately
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")

# Build database connection safely
DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username="root",
    password=DATABASE_PASSWORD,
    host="localhost",
    port=3306,
    database="enterprise_rag"
)

# Create database engine
engine = create_engine(DATABASE_URL)

# Create database sessions
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

# Base class for database models
Base = declarative_base()

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()