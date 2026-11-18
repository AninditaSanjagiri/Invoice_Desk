"""OCR engines. DoctrEngine runs real DocTR; ProvidedOCREngine reads DocILE's pre-computed DocTR-format OCR."""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Protocol
from .models import OCRDocument
from .serialize import ocr_from_export

log = logging.getLogger(__name__)
class OCRError(RuntimeError): pass

class OCREngine(Protocol):
    def run(self, path: Path, max_pages: int) -> OCRDocument: ...

class DoctrEngine:
    """DocTR (CPU is sufficient; the first call downloads model weights). The predictor is created lazily and reused."""
    _predictor = None
    def _get(self):
        if DoctrEngine._predictor is None:
            try:
                from doctr.models import ocr_predictor
            except Exception as e:
                raise OCRError("python-doctr is not installed (pip install 'python-doctr[torch]').") from e
            DoctrEngine._predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
        return DoctrEngine._predictor
    def run(self, path: Path, max_pages: int = 10) -> OCRDocument:
        try:
            from doctr.io import DocumentFile
            path = Path(path)
            pages = DocumentFile.from_pdf(str(path)) if path.suffix.lower() == ".pdf" else DocumentFile.from_images(str(path))
            if not pages: raise OCRError("No pages could be read from the document.")
            if len(pages) > max_pages: raise OCRError(f"Document has {len(pages)} pages (limit {max_pages}).")
            export = self._get()(pages).export()
        except OCRError: raise
        except Exception as e:
            raise OCRError(f"OCR failed: {type(e).__name__}: {str(e)[:200]}") from e
        doc = ocr_from_export(export, "doctr", max_pages)
        if not doc.words: raise OCRError("OCR found no text in the document.")
        return doc

class ProvidedOCREngine:
    """Reads <ocr_dir>/<doc_id>.json (DocILE-provided OCR, DocTR format). Used by the evaluation script."""
    def __init__(self, ocr_dir: Path): self.ocr_dir = Path(ocr_dir)
    def load(self, doc_id: str) -> OCRDocument:
        p = self.ocr_dir / f"{doc_id}.json"
        if not p.exists(): raise OCRError(f"No provided OCR for {doc_id}")
        return ocr_from_export(json.loads(p.read_text(encoding="utf-8")), "docile_provided")
    def run(self, path: Path, max_pages: int = 10) -> OCRDocument: return self.load(Path(path).stem)

def cached_ocr(engine: OCREngine, path: Path, file_hash: str, cache_dir: Path, max_pages: int) -> tuple[OCRDocument, bool]:
    """OCR results are cached by document hash (the LLM result is never cached across documents)."""
    cache_dir.mkdir(parents=True, exist_ok=True); cp = cache_dir / f"{file_hash[:32]}.json"
    if cp.exists():
        try: return OCRDocument.from_dict(json.loads(cp.read_text(encoding="utf-8"))), True
        except Exception: log.warning("corrupt OCR cache entry %s - recomputing", cp.name)
    doc = engine.run(path, max_pages); tmp = cp.with_suffix(".tmp"); tmp.write_text(json.dumps(doc.to_dict()), encoding="utf-8"); tmp.replace(cp)
    return doc, False
