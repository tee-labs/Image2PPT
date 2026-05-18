"""Quality metrics for one fixture's reconstruction.

Four categories:
  - structural: pass/fail integrity checks on the PPTX package
  - text:       character error rate, keyword recall, extra-text ratio
  - visual:     pixel-match ratios + color histogram similarity
  - counts:     textbox / image-object counts vs expected ranges

Each category produces a small dict. A composite score (0-1) is then
derived from a fixed weighting over the continuous metrics.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
PLACEHOLDER_WORDS = ("Click to add", "Lorem ipsum", "Replace with", "TODO", "TBD")


# ---------------------------------------------------------------------- helpers

def _read_pptx_texts(pptx_path: Path) -> tuple[list[str], int, int]:
    """Return (texts, image_object_count, textbox_count) from a PPTX file."""
    texts: list[str] = []
    image_count = 0
    textbox_count = 0
    with zipfile.ZipFile(pptx_path) as zf:
        slide_names = sorted(
            n for n in zf.namelist()
            if n.startswith("ppt/slides/slide") and n.endswith(".xml")
        )
        media_names = [
            n for n in zf.namelist() if n.startswith("ppt/media/")
        ]
        image_count = len(media_names)
        for slide_name in slide_names:
            root = ET.fromstring(zf.read(slide_name))
            for sp in root.iter():
                tag = sp.tag.split("}")[-1]
                if tag == "sp":
                    has_text = sp.find(".//a:t", NS) is not None
                    if has_text:
                        textbox_count += 1
            for node in root.findall(".//a:t", NS):
                value = node.text or ""
                if value:
                    texts.append(value)
    return texts, image_count, textbox_count


def _placeholder_text_count(texts: list[str]) -> int:
    return sum(
        1 for t in texts if any(w in t for w in PLACEHOLDER_WORDS)
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------- structural

def structural_metrics(pptx_path: Path) -> dict:
    failures: list[str] = []
    if not pptx_path.exists():
        return {
            "ok": False,
            "slide_count": 0,
            "media_count": 0,
            "text_run_count": 0,
            "placeholder_count": 0,
            "zero_byte_media": 0,
            "failures": ["PPTX not produced."],
        }
    with zipfile.ZipFile(pptx_path) as zf:
        names = zf.namelist()
        slides = [n for n in names if n.startswith("ppt/slides/slide")
                  and n.endswith(".xml")]
        media = [n for n in names if n.startswith("ppt/media/")]
        zero_byte = [n for n in media if zf.getinfo(n).file_size == 0]
        texts, _, _ = _read_pptx_texts(pptx_path)
        placeholder = _placeholder_text_count(texts)

    if not slides:
        failures.append("PPTX contains no slides.")
    if zero_byte:
        failures.append(f"PPTX contains {len(zero_byte)} zero-byte media.")
    if placeholder:
        failures.append(f"PPTX contains {placeholder} placeholder text runs.")

    return {
        "ok": not failures,
        "slide_count": len(slides),
        "media_count": len(media),
        "text_run_count": len(texts),
        "placeholder_count": placeholder,
        "zero_byte_media": len(zero_byte),
        "failures": failures,
    }


# ---------------------------------------------------------------------- text

def text_metrics(pptx_path: Path, must_appear_text: list[str],
                 source_text_reference: str | None) -> dict:
    """Compute text fidelity metrics.

    - keyword_recall: fraction of must_appear_text strings present in PPTX.
    - cer: char edit distance / len(reference), if reference text supplied.
    - extra_ratio: chars in PPTX text that are not in reference, as ratio.
    """
    texts, _, _ = _read_pptx_texts(pptx_path)
    joined = " ".join(texts)
    joined_norm = _normalize_text(joined)

    # Keywords may be split across multiple <a:t> runs in the PPTX (e.g. a title
    # that becomes two runs: "为什么需要" + "DeckWeaver"). Match against the
    # whitespace-stripped version so split runs still count as found.
    def _has_keyword(k: str) -> bool:
        return _normalize_text(k) in joined_norm

    found = [k for k in must_appear_text if _has_keyword(k)]
    missing = [k for k in must_appear_text if not _has_keyword(k)]
    keyword_recall = len(found) / max(len(must_appear_text), 1)

    cer = None
    extra_ratio = None
    if source_text_reference:
        ref_norm = _normalize_text(source_text_reference)
        if ref_norm:
            distance = _edit_distance(ref_norm, joined_norm)
            cer = distance / len(ref_norm)
            ref_chars = set(ref_norm)
            extra_chars = sum(1 for c in joined_norm if c not in ref_chars)
            extra_ratio = extra_chars / max(len(joined_norm), 1)

    return {
        "keyword_recall": round(keyword_recall, 4),
        "cer": round(cer, 4) if cer is not None else None,
        "extra_ratio": round(extra_ratio, 4) if extra_ratio is not None else None,
        "missing_keywords": missing,
        "found_keyword_count": len(found),
        "total_text_chars": len(joined_norm),
    }


# ---------------------------------------------------------------------- visual

# Visual comparison target size — both images resized to this for fair pixel
# comparisons. Aspect ratios may differ across fixtures; we always letterbox
# to this canonical size so per-pixel ops are comparable.
COMPARE_W = 1280
COMPARE_H = 720


def _load_for_compare(path: Path) -> np.ndarray:
    """Return an RGB image resized into a COMPARE_W x COMPARE_H canvas."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((COMPARE_W, COMPARE_H), Image.LANCZOS)
    canvas = Image.new("RGB", (COMPARE_W, COMPARE_H), (255, 255, 255))
    off_x = (COMPARE_W - img.width) // 2
    off_y = (COMPARE_H - img.height) // 2
    canvas.paste(img, (off_x, off_y))
    return np.asarray(canvas)


def _gray(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def visual_metrics(source_png: Path, reconstructed_png: Path) -> dict:
    if not reconstructed_png.exists():
        return {
            "exact_match_ratio": 0.0,
            "near_match_ratio": 0.0,
            "blurred_match_ratio": 0.0,
            "color_histogram_sim": 0.0,
            "available": False,
        }
    src = _load_for_compare(source_png)
    rec = _load_for_compare(reconstructed_png)

    src_g = _gray(src).astype(np.int16)
    rec_g = _gray(rec).astype(np.int16)

    diff = np.abs(src_g - rec_g)
    n = diff.size
    exact = float((diff == 0).sum() / n)
    near = float((diff <= 10).sum() / n)

    src_blur = cv2.GaussianBlur(_gray(src), (0, 0), sigmaX=3.0)
    rec_blur = cv2.GaussianBlur(_gray(rec), (0, 0), sigmaX=3.0)
    diff_blur = np.abs(src_blur.astype(np.int16) - rec_blur.astype(np.int16))
    blurred = float((diff_blur <= 10).sum() / n)

    hist_a = cv2.calcHist([src], [0, 1, 2], None, [16, 16, 16],
                          [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([rec], [0, 1, 2], None, [16, 16, 16],
                          [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    hist_sim = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

    return {
        "exact_match_ratio": round(exact, 4),
        "near_match_ratio": round(near, 4),
        "blurred_match_ratio": round(blurred, 4),
        "color_histogram_sim": round(hist_sim, 4),
        "available": True,
    }


# ---------------------------------------------------------------------- counts

def count_metrics(pptx_path: Path, expected: dict) -> dict:
    _, image_count, textbox_count = _read_pptx_texts(pptx_path)
    img_lo, img_hi = expected["expected_image_objects"]["min"], \
                     expected["expected_image_objects"]["max"]
    tb_lo, tb_hi = expected["expected_textboxes"]["min"], \
                   expected["expected_textboxes"]["max"]
    return {
        "textbox_count": textbox_count,
        "textbox_in_range": tb_lo <= textbox_count <= tb_hi,
        "textbox_expected": [tb_lo, tb_hi],
        "image_object_count": image_count,
        "image_in_range": img_lo <= image_count <= img_hi,
        "image_expected": [img_lo, img_hi],
    }


# ---------------------------------------------------------------------- composite

# Weights chosen to keep visual + text dominant; counts is secondary.
WEIGHTS = {
    "visual_blurred": 0.30,
    "visual_hist": 0.10,
    "text_recall": 0.30,
    "text_cer_inv": 0.20,  # 1 - cer, when available
    "counts_in_range": 0.10,
}


def composite_score(structural: dict, text: dict, visual: dict, counts: dict
                    ) -> float:
    if not structural.get("ok"):
        return 0.0
    parts = {}
    parts["visual_blurred"] = visual.get("blurred_match_ratio", 0.0)
    parts["visual_hist"] = max(0.0, visual.get("color_histogram_sim", 0.0))
    parts["text_recall"] = text.get("keyword_recall", 0.0)
    cer = text.get("cer")
    parts["text_cer_inv"] = (1.0 - cer) if cer is not None else parts["text_recall"]
    parts["counts_in_range"] = (
        (1.0 if counts.get("textbox_in_range") else 0.0) * 0.5
        + (1.0 if counts.get("image_in_range") else 0.0) * 0.5
    )
    score = sum(parts[k] * w for k, w in WEIGHTS.items())
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------- driver

@dataclass
class FixtureScores:
    fixture: str
    structural: dict
    text: dict
    visual: dict
    counts: dict
    composite: float
    runtime_seconds: float
    cached_pipeline: bool

    def to_dict(self) -> dict:
        return asdict(self)


def score_fixture(*, fixture_name: str, source_png: Path, expected: dict,
                  pptx_path: Path, reconstructed_png: Path,
                  source_text_reference: str | None,
                  runtime_seconds: float, cached_pipeline: bool) -> FixtureScores:
    structural = structural_metrics(pptx_path)
    text = text_metrics(pptx_path, expected.get("must_appear_text", []),
                        source_text_reference)
    visual = visual_metrics(source_png, reconstructed_png)
    counts = count_metrics(pptx_path, expected)
    composite = composite_score(structural, text, visual, counts)
    return FixtureScores(
        fixture=fixture_name,
        structural=structural,
        text=text,
        visual=visual,
        counts=counts,
        composite=composite,
        runtime_seconds=round(runtime_seconds, 2),
        cached_pipeline=cached_pipeline,
    )


def load_source_text_reference(work_dir: Path) -> str | None:
    """Concatenate text content from OCR JSON of the source image.

    The pipeline stores OCR output at work_dir/ocr/page_NN.ocr.json. We use
    the *source* OCR (not the reconstructed PPTX's OCR) as the text reference
    for CER calculation — it's the closest thing we have to truth.
    """
    ocr_dir = work_dir / "ocr"
    if not ocr_dir.is_dir():
        return None
    chunks: list[str] = []
    for path in sorted(ocr_dir.glob("page_*.ocr.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in _walk_text_nodes(data):
            chunks.append(item)
    if not chunks:
        return None
    return "\n".join(chunks)


def _walk_text_nodes(obj) -> list[str]:
    """Best-effort: pull any string under a 'text' key out of arbitrary JSON."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "text" and isinstance(v, str):
                found.append(v)
            else:
                found.extend(_walk_text_nodes(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_walk_text_nodes(item))
    return found
