"""Upload validation and safe file handling."""
from __future__ import annotations
import re
from pathlib import Path

ALLOWED_EXT = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MAGIC = {".pdf": (b"%PDF",), ".png": (b"\x89PNG",), ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",)}

class UploadError(ValueError): pass

def safe_filename(name: str) -> str:
    base = Path(name or "upload").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem)[:80].strip("._") or "upload"
    return f"{stem}{Path(base).suffix.lower()}"

def validate_upload(filename: str, data: bytes, max_mb: int) -> str:
    """Returns the lower-case extension or raises UploadError."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT: raise UploadError(f"Unsupported file type '{ext}'. Allowed: PDF, PNG, JPG, JPEG.")
    if not data: raise UploadError("The file is empty.")
    if len(data) > max_mb * 1024 * 1024: raise UploadError(f"File is larger than the {max_mb} MB limit.")
    if not any(data.startswith(m) for m in MAGIC[ext]): raise UploadError("File content does not match its extension.")
    return ext

def store_upload(data: bytes, filename: str, upload_dir: Path, file_hash: str) -> Path:
    """Stored under the content hash - the original filename is never used as a path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{file_hash[:32]}{Path(filename).suffix.lower()}"
    if not path.exists(): path.write_bytes(data)
    return path
