
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

DATABASE_PASSWORD = os.getenv("DB_PASSWORD")

if not DATABASE_PASSWORD:
    raise ValueError(
        "DB_PASSWORD is missing from the root .env file"
    )


DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("DB_USER", "root"),
    password=DATABASE_PASSWORD,
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME", "enterprise_rag")
)


# --------------------------------------------------
# DATABASE ENGINE
# --------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo=False
)


# --------------------------------------------------
# DATABASE SESSION
# --------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


# --------------------------------------------------
# SQLALCHEMY BASE
# --------------------------------------------------

Base = declarative_base()


# --------------------------------------------------
# DATABASE DEPENDENCY
# --------------------------------------------------

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
