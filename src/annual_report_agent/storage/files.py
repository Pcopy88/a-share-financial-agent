"""Content-addressed files: supplied names never become filesystem paths."""

import hashlib
from pathlib import Path
from uuid import uuid4


def save_pdf(directory: Path, filename: str, content: bytes, *, max_bytes: int) -> tuple[Path, str]:
    if len(content) > max_bytes:
        raise ValueError("PDF too large")
    if not filename.lower().endswith(".pdf") or b"%PDF-" not in content[:1024]:
        raise ValueError("a valid PDF file is required")
    digest = hashlib.sha256(content).hexdigest()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.pdf"
    if not destination.exists():
        temp = directory / f"{digest}-{uuid4().hex}.part"
        try:
            temp.write_bytes(content)
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
    return destination, digest


def render_page(path: str | Path, page_number: int, scale: float = 1.5) -> bytes:
    import io

    import pypdfium2
    with pypdfium2.PdfDocument(str(path)) as document:
        if not 1 <= page_number <= len(document):
            raise ValueError("PDF page outside valid range")
        page = document[page_number - 1]
        try:
            bitmap = page.render(scale=scale)
            try:
                output = io.BytesIO()
                bitmap.to_pil().save(output, format="PNG")
                return output.getvalue()
            finally:
                bitmap.close()
        finally:
            page.close()
