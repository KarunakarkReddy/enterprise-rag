from sqlalchemy import text

from .database import engine, Base
from .models import Document

try:
    Base.metadata.create_all(bind=engine)

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    print("MySQL connected successfully!")
    print("Documents table created successfully!")

except Exception as error:
    print("Error:", error)