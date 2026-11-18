"""Document preview and OCR-evidence rendering (PIL)."""
from __future__ import annotations
from pathlib import Path

def render_pages(path: str | Path, max_pages: int = 3, scale: float = 1.4):
    """Returns a list of PIL images (PDF via pypdfium2, images directly). Empty list if the file cannot be rendered."""
    from PIL import Image
    path = Path(path)
    try:
        if path.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(path)); return [pdf[i].render(scale=scale).to_pil().convert("RGB") for i in range(min(len(pdf), max_pages))]
        return [Image.open(path).convert("RGB")]
    except Exception:
        return []

def highlight(img, bbox, color=(220, 38, 38)):
    """Draw a normalised bbox [x0,y0,x1,y1] on a copy of the page image."""
    from PIL import ImageDraw
    out = img.copy(); w, h = out.size; d = ImageDraw.Draw(out); x0, y0, x1, y1 = bbox; pad = 3
    d.rectangle([x0 * w - pad, y0 * h - pad, x1 * w + pad, y1 * h + pad], outline=color, width=3); return out
