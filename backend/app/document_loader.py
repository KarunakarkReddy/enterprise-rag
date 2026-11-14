
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path):
    """
    Extract text from every page of a PDF.

    Returns:
        str: Combined text with page numbers.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError("The provided path is not a file.")

    try:
        reader = PdfReader(str(file_path), strict=False)

        # Check whether the PDF is password protected.
        if reader.is_encrypted:
            try:
                decryption_result = reader.decrypt("")

                if not decryption_result:
                    raise ValueError(
                        "This PDF is password protected. "
                        "Please upload an unprotected PDF."
                    )

            except PdfReadError as error:
                raise ValueError(
                    "Unable to decrypt this PDF."
                ) from error

        extracted_pages = []

        # Extract text from every page.
        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text()

                if text and text.strip():
                    extracted_pages.append(
                        f"\n--- Page {page_number} ---\n{text.strip()}"
                    )

            except Exception:
                logger.exception(
                    "Text extraction failed on PDF page %s",
                    page_number
                )

        extracted_text = "\n".join(extracted_pages)

        # Reject PDFs without extractable text.
        if not extracted_text.strip():
            raise ValueError(
                "No extractable text found. "
                "This PDF may contain scanned images. "
                "OCR is required to read scanned PDFs."
            )

        return extracted_text

    except PdfReadError as error:
        raise ValueError(
            "The uploaded PDF is corrupted or cannot be read."
        ) from error

    except OSError as error:
        raise ValueError(
            "Unable to open or read the uploaded PDF."
        ) from error
