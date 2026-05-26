#!/usr/bin/env python
"""Three-engine OCR cross-verification: PaddleOCR + EasyOCR + Tesseract 5.

Used by prepare_ocr.py to assign a consensus tier to each low-confidence
PaddleOCR detection, pre-fill a best-guess `corrected_text`, and surface
the three candidate texts to the agent for optional post-build review.

Engines are loaded lazily and reused across calls (loaders are cached).
Tesseract uses chi_sim+eng with --psm 7 (single text line) which matches
the per-text-line crops PaddleOCR produces.

Consensus tiers (color in the annotated page render):

  🟢 GREEN   — strong consensus:
                 [A] all 3 engines normalize to the same string, OR
                 [B] any 2 of 3 normalize to the same string
              suggested_text = consensus form (original casing/spacing
              from the highest-conf engine that produced it)

  🟡 YELLOW  — weak consensus / single-engine high conf:
                 [D] all 3 within pairwise edit distance ≤ 1 (and min
                     length ≥ 3) — small spelling/spacing diffs only, OR
                 [D'] one engine confidence ≥ 0.95 — single strong vote, OR
                 [G] no consensus BUT the strongest engine has conf ≥
                     0.85 — handles the common case of CJK small text
                     where EasyOCR/Tesseract produce garbage but
                     PaddleOCR is reliably right (calibration on real
                     decks: Paddle ≥ 0.85 is right ~95%+ of the time
                     even when other engines disagree).
              suggested_text = highest-conf engine's text (with the
              script-strength override below)

  🔴 RED     — no consensus AND no engine even moderately confident
              suggested_text = highest-conf engine's text (still
              pre-filled so the deck can build without agent input)

For RED entries an OPTIONAL 4th engine pass runs via
`paddle_crop_rescue()`: PaddleOCR re-applied to the isolated crop
(with width-based upscaling). On dense CJK text this often produces a
different and usually better result than the full-page pass. If
paddle_crop agrees with easyocr/tesseract OR has moderate-high conf
on its own, the entry is upgraded to YELLOW with the paddle_crop text.

Script-strength override (applied during yellow/red tie-breaking):
  - CJK-dominant text  → prefer PaddleOCR (strongest on Chinese)
  - ASCII-dominant     → prefer Tesseract (strongest on Latin/digits)
  - mixed              → prefer the engine with the highest reported conf

Tesseract has no built-in line-level confidence in the simple call.
We extract it from `image_to_data` (TSV mode) by averaging per-word
confs > 0. If Tesseract returns empty text, conf = 0.0.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.gpu import easyocr_use_gpu  # noqa: E402

# Lazy engine loaders. The first call constructs each engine; subsequent
# calls reuse the cached instance — this keeps prepare_ocr.py warm across
# pages and avoids paying the ~1.3 s EasyOCR startup per page.
_easy_reader = None
_pytesseract = None


def _get_easyocr():
    global _easy_reader
    if _easy_reader is None:
        import easyocr  # noqa: E402
        # GPU only when CUDA is actually present (see shared/gpu.py — MPS
        # stays off by default because it spams a per-load warning and
        # CPU is already ~50 ms / small crop).
        _easy_reader = easyocr.Reader(
            ["ch_sim", "en"], gpu=easyocr_use_gpu(), verbose=False,
        )
    return _easy_reader


def _get_tesseract():
    global _pytesseract
    if _pytesseract is None:
        import pytesseract  # noqa: E402
        _pytesseract = pytesseract
    return _pytesseract


# ---------- text normalization & similarity ----------

_PUNCT_MAP = str.maketrans({
    "，": ",", "。": ".", "：": ":", "；": ";",
    "！": "!", "？": "?", "（": "(", "）": ")",
    "【": "[", "】": "]", "「": "[", "」": "]",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "／": "/", "、": ",", "～": "~",
})


def _normalize(s: str) -> str:
    """NFKC + drop whitespace + unify common CJK/ASCII punctuation + lowercase ASCII."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    s = s.translate(_PUNCT_MAP)
    # Lowercase if the (now whitespace-free) string is pure ASCII letters/digits/punct.
    if s.isascii():
        s = s.lower()
    return s


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein. Strings are short (≤ ~50 chars) so O(n·m) is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1,
                         prev[j] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


def _is_cjk_dominant(s: str) -> bool:
    if not s:
        return False
    cjk = sum(1 for c in s if "一" <= c <= "鿿")
    return cjk * 2 >= len(s)  # CJK chars are at least half the text


# ---------- engine adapters ----------

@dataclass
class EnginePred:
    text: str
    conf: float


def run_easyocr(crop_path: Path) -> EnginePred:
    """Run EasyOCR on a single small crop (single-line region).

    EasyOCR returns a list of (bbox, text, conf) detections; for a tight
    crop there is usually just one. If there are multiple line fragments
    we concatenate texts and take the max conf — matches the way Paddle
    reports a single line per detection.
    """
    reader = _get_easyocr()
    try:
        res = reader.readtext(str(crop_path), detail=1)
    except Exception:
        return EnginePred("", 0.0)
    if not res:
        return EnginePred("", 0.0)
    text = "".join(r[1] for r in res)  # concat without spaces; normalize will strip them anyway
    conf = max((float(r[2]) for r in res), default=0.0)
    return EnginePred(text, conf)


def run_paddle_crop(crop_image, paddle_model, *,
                    upscale_threshold: int = 200,
                    upscale_factor: int = 3) -> EnginePred:
    """Run PaddleOCR on a single cropped region, with optional upscaling.

    Why this exists: PaddleOCR's full-page pass and crop-only pass can
    produce meaningfully different results on the same bbox — internal
    normalization, batching, and neighbor-text suppression behave
    differently on an isolated single-line crop. Empirically the crop
    pass beats the full-page pass on ~60-80% of dense-CJK RED entries.

    For short/medium crops (width < `upscale_threshold`, default 200px)
    we feed the crop as-is; LANCZOS upscaling on short text tends to
    introduce edge artifacts that hurt recognition. For long crops
    (≥ threshold), we upscale 3x with LANCZOS — the recognizer benefits
    from more pixels per character on dense lines.

    `paddle_model` is a pre-loaded PaddleOCR instance passed in from
    prepare_ocr.py; this module does not create one on its own (the
    model is heavy, ~6 s load, and the main pipeline already has one
    warm).
    """
    from PIL import Image
    import tempfile

    # Lazy import to avoid a load-order tangle with prepare_ocr.py.
    from ocr_paddle import extract_items  # noqa: PLC0415

    try:
        # Heuristic: long crops benefit from upscaling, short ones don't.
        if crop_image.width >= upscale_threshold:
            img = crop_image.resize(
                (crop_image.width * upscale_factor,
                 crop_image.height * upscale_factor),
                Image.LANCZOS,
            )
        else:
            img = crop_image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = tf.name
        img.save(tmp_path)
        result = paddle_model.predict(tmp_path)
        items = extract_items(result, min_conf=0.0)
        if not items:
            return EnginePred("", 0.0)
        # Concatenate detected lines (a crop should normally produce one
        # line but the detector occasionally fragments it).
        txt = "".join(i["text"] for i in items)
        conf = max((float(i["confidence"]) for i in items), default=0.0)
        return EnginePred(txt, conf)
    except Exception:
        return EnginePred("", 0.0)


def run_tesseract(crop_path: Path) -> EnginePred:
    """Run Tesseract 5 on a single small crop.

    Uses chi_sim+eng so the engine has both alphabets available; --psm 7
    tells it to treat the image as a single text line, which matches the
    crops we feed it. We pull per-line conf from image_to_data (TSV) —
    the simple image_to_string call doesn't return one.
    """
    pt = _get_tesseract()
    cfg = "--psm 7"
    try:
        text = pt.image_to_string(
            str(crop_path), lang="chi_sim+eng", config=cfg
        ).strip()
    except Exception:
        return EnginePred("", 0.0)
    if not text:
        return EnginePred("", 0.0)

    # Per-word conf via TSV; average positive values for a line-level estimate.
    try:
        data = pt.image_to_data(
            str(crop_path), lang="chi_sim+eng", config=cfg,
            output_type=pt.Output.DICT,
        )
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) > 0]
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    except Exception:
        conf = 0.0
    return EnginePred(text, conf)


# ---------- consensus algorithm ----------

@dataclass
class CrossVerifyResult:
    tier: str                              # "green" | "yellow" | "red"
    suggested_text: str                    # pre-filled corrected_text
    reason: str                            # short rule name: A, B, D, D_prime, F
    candidates: dict[str, dict[str, float | str]] = field(default_factory=dict)
    # candidates: {"paddle": {"text": "...", "conf": 0.91}, ...}

    def as_dict(self):
        return asdict(self)


def cross_verify(paddle: EnginePred, easy: EnginePred, tess: EnginePred,
                 *, high_conf: float = 0.95,
                 moderate_conf: float = 0.85) -> CrossVerifyResult:
    """Compute consensus over the three engine predictions.

    `paddle`/`easy`/`tess` are EnginePred(text, conf). Returns a tier
    and suggested text (always non-null so the deck can build without
    agent intervention).

    high_conf      — single-engine confidence to trigger D' (default 0.95)
    moderate_conf  — strongest-engine confidence to trigger G fallback
                     (default 0.85). Below this, full disagreement → red.
    """
    candidates = {
        "paddle":    {"text": paddle.text, "conf": round(paddle.conf, 4)},
        "easyocr":   {"text": easy.text,   "conf": round(easy.conf, 4)},
        "tesseract": {"text": tess.text,   "conf": round(tess.conf, 4)},
    }
    norms = {
        "paddle":    _normalize(paddle.text),
        "easyocr":   _normalize(easy.text),
        "tesseract": _normalize(tess.text),
    }

    # ---- All three empty → not real text. Caller may auto-clear this. ----
    if not any(norms.values()):
        return CrossVerifyResult(
            tier="green", suggested_text="", reason="C_all_empty",
            candidates=candidates,
        )

    # ---- [A] all three normalize identically (non-empty) ----
    nv = list(norms.values())
    if nv[0] == nv[1] == nv[2] and nv[0]:
        return CrossVerifyResult(
            tier="green", suggested_text=paddle.text or easy.text or tess.text,
            reason="A_strict_consensus", candidates=candidates,
        )

    # ---- [B] any two normalize identically (non-empty) ----
    from collections import Counter
    counter = Counter(v for v in nv if v)
    if counter:
        most, cnt = counter.most_common(1)[0]
        if cnt >= 2:
            # Pick original-cased text from the highest-conf engine that
            # produced this normalized form.
            winners = [
                (name, candidates[name]["text"], candidates[name]["conf"])
                for name in ("paddle", "easyocr", "tesseract")
                if norms[name] == most
            ]
            winners.sort(key=lambda w: -float(w[2]))
            return CrossVerifyResult(
                tier="green", suggested_text=winners[0][1],
                reason="B_majority", candidates=candidates,
            )

    # ---- [D] all three within pairwise edit distance ≤ 1 (and min len ≥ 3) ----
    if all(v for v in nv):
        d_pe = _edit_distance(norms["paddle"], norms["easyocr"])
        d_pt = _edit_distance(norms["paddle"], norms["tesseract"])
        d_et = _edit_distance(norms["easyocr"], norms["tesseract"])
        min_len = min(len(v) for v in nv)
        if max(d_pe, d_pt, d_et) <= 1 and min_len >= 3:
            best = _pick_best(paddle, easy, tess)
            return CrossVerifyResult(
                tier="yellow", suggested_text=best.text,
                reason="D_edit_close", candidates=candidates,
            )

    # ---- [D'] one engine confidence ≥ high_conf threshold ----
    high = [
        ("paddle", paddle),
        ("easyocr", easy),
        ("tesseract", tess),
    ]
    high = [(n, p) for n, p in high if p.conf >= high_conf and p.text]
    if high:
        # If multiple engines pass the threshold, prefer by script strength.
        text_for_script = max(high, key=lambda np: len(np[1].text))[1].text
        best = _pick_best_from_high(high, text_for_script)
        return CrossVerifyResult(
            tier="yellow", suggested_text=best.text,
            reason="D_prime_single_high_conf", candidates=candidates,
        )

    # ---- [G] strongest engine has moderate confidence → yellow ----
    # This is the safety net for cases where the weak engines fail
    # entirely (CJK small text → EasyOCR/Tesseract garbage) but
    # PaddleOCR is still calibrated and right. We track this as a
    # SEPARATE reason so the annotated page can hint "no consensus but
    # one engine is fairly sure".
    best = _pick_best(paddle, easy, tess)
    if best.conf >= moderate_conf:
        return CrossVerifyResult(
            tier="yellow", suggested_text=best.text,
            reason="G_moderate_conf_fallback", candidates=candidates,
        )

    # ---- [F] no consensus and no confident vote → red ----
    return CrossVerifyResult(
        tier="red", suggested_text=best.text,
        reason="F_disagree", candidates=candidates,
    )


def paddle_crop_rescue(initial: CrossVerifyResult,
                       paddle_crop: EnginePred,
                       *, moderate_conf: float = 0.85) -> CrossVerifyResult:
    """Optional 4th-engine rescue, ONLY runs on RED-tier results.

    PaddleOCR run on an isolated crop frequently produces a different
    (and often better) recognition vs the original full-page pass. We
    test this 4th vote against the existing 3 candidates and the
    initial suggestion:

    [G'-A] paddle_crop normalizes the same as paddle_full (the original
           OCR) — no new information; keep the initial result but
           record paddle_crop as a candidate.

    [G'-B] paddle_crop normalizes the same as easyocr OR tesseract —
           crop pass corroborates a weaker engine. Upgrade RED → YELLOW
           and use that agreed text.

    [G'-C] paddle_crop disagrees with all 3 BUT has conf ≥ moderate_conf
           (default 0.85) — Paddle's crop-pass confidence is generally
           well-calibrated on dense Chinese text where the full-page
           pass struggled. Upgrade RED → YELLOW and prefer paddle_crop's
           text.

    Otherwise: keep RED but add paddle_crop to the candidates so the
    annotated page surfaces it as a fourth option for the agent.
    """
    if initial.tier != "red":
        return initial  # Only rescue red entries

    new_candidates = dict(initial.candidates)
    new_candidates["paddle_crop"] = {
        "text": paddle_crop.text,
        "conf": round(paddle_crop.conf, 4),
    }

    if not paddle_crop.text:
        return CrossVerifyResult(
            tier="red",
            suggested_text=initial.suggested_text,
            reason=initial.reason,
            candidates=new_candidates,
        )

    pc_norm = _normalize(paddle_crop.text)
    if not pc_norm:
        return CrossVerifyResult(
            tier="red",
            suggested_text=initial.suggested_text,
            reason=initial.reason,
            candidates=new_candidates,
        )

    # [G'-A] paddle_crop == paddle_full → nothing new, keep red.
    paddle_full_norm = _normalize(initial.candidates.get("paddle", {}).get("text", ""))
    if pc_norm == paddle_full_norm:
        return CrossVerifyResult(
            tier="red",
            suggested_text=initial.suggested_text,
            reason=initial.reason,
            candidates=new_candidates,
        )

    # [G'-B] paddle_crop agrees with easyocr or tesseract → yellow.
    for name in ("easyocr", "tesseract"):
        other_norm = _normalize(initial.candidates.get(name, {}).get("text", ""))
        if other_norm and other_norm == pc_norm:
            return CrossVerifyResult(
                tier="yellow",
                suggested_text=paddle_crop.text,
                reason=f"G_prime_paddle_crop_agrees_with_{name}",
                candidates=new_candidates,
            )

    # [G'-C] paddle_crop alone has moderate-high conf → yellow.
    if paddle_crop.conf >= moderate_conf:
        return CrossVerifyResult(
            tier="yellow",
            suggested_text=paddle_crop.text,
            reason="G_prime_paddle_crop_moderate_conf",
            candidates=new_candidates,
        )

    # Still red — but paddle_crop is now visible in the candidate list.
    return CrossVerifyResult(
        tier="red",
        suggested_text=initial.suggested_text,
        reason=initial.reason,
        candidates=new_candidates,
    )


def _pick_best(paddle: EnginePred, easy: EnginePred, tess: EnginePred) -> EnginePred:
    """Tie-break by script strength when engines disagree.

    For CJK-dominant text PaddleOCR is the strongest (often by a wide
    margin on small Chinese type). For ASCII-dominant text Tesseract 5
    is the strongest. Otherwise pick by reported confidence.

    Important: we look at PaddleOCR's output to determine the script,
    NOT the longest of the three. The "longest" heuristic backfires
    when one of the weak engines returns long garbage (Tesseract often
    emits a long ASCII string for CJK input it can't read) — that would
    flip the script vote to ASCII and then suggest the garbage itself.
    Trust PaddleOCR for the script-type signal even when it's wrong on
    individual characters.
    """
    reference = paddle.text or easy.text or tess.text or ""
    # Script-strength preference applies only when the preferred engine
    # also has a non-trivial confidence — otherwise we'd swap a
    # confident Paddle answer for a zero-conf Tesseract answer just
    # because the text happens to be ASCII.
    MIN_PREFERRED_CONF = 0.30
    if _is_cjk_dominant(reference):
        if paddle.text and paddle.conf >= MIN_PREFERRED_CONF:
            return paddle
    elif reference.isascii() and reference.strip():
        if tess.text and tess.conf >= MIN_PREFERRED_CONF:
            return tess
    # Fall back to highest reported conf among non-empty.
    candidates = [p for p in (paddle, easy, tess) if p.text]
    if not candidates:
        return paddle  # all empty; preserve paddle as-is
    return max(candidates, key=lambda p: p.conf)


def _pick_best_from_high(high_list, text_for_script: str) -> EnginePred:
    """Same script-aware preference, but only among engines that passed
    the high-conf threshold."""
    engines = dict(high_list)
    if _is_cjk_dominant(text_for_script) and "paddle" in engines:
        return engines["paddle"]
    if text_for_script.isascii() and "tesseract" in engines:
        return engines["tesseract"]
    return max(engines.values(), key=lambda p: p.conf)


# ---------- CLI entry point for standalone verification ----------

def _cli() -> int:
    """Stand-alone mode: run 3-engine cross-verify over an existing
    `inventory/page_NN.ocr.json` + source `page_NN.*`, print per-entry
    consensus, and emit a stats summary. Does not modify any files.

    Useful for measuring agreement rate on real data before integrating
    into prepare_ocr.py.
    """
    import argparse
    import json
    import time
    from PIL import Image

    p = argparse.ArgumentParser(description=_cli.__doc__)
    p.add_argument("--ocr", required=True, help="page_NN.ocr.json from PaddleOCR")
    p.add_argument("--image", required=True, help="page_NN source image")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Only verify entries with Paddle conf < this (default 0.95)")
    p.add_argument("--padding", type=int, default=6)
    p.add_argument("--high-conf", type=float, default=0.95,
                   help="Threshold for the D' single-engine high-conf rule")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap entries verified (0 = no cap, sorted by ascending conf)")
    args = p.parse_args()

    ocr_data = json.loads(Path(args.ocr).read_text(encoding="utf-8"))
    img = Image.open(args.image).convert("RGB")
    W, H = img.size

    candidates = [
        (idx, e) for idx, e in enumerate(ocr_data)
        if float(e.get("confidence", 1.0)) < args.threshold
    ]
    candidates.sort(key=lambda p: float(p[1].get("confidence", 1.0)))
    if args.limit:
        candidates = candidates[: args.limit]

    print(f"Verifying {len(candidates)} entries (Paddle conf < {args.threshold})")
    print("Loading EasyOCR + Tesseract...")
    _get_easyocr()
    _get_tesseract()

    tier_count = {"green": 0, "yellow": 0, "red": 0}
    reason_count: dict[str, int] = {}
    t0 = time.time()

    for slot, (idx, e) in enumerate(candidates):
        pad = args.padding
        x1 = max(0, int(e["x1"]) - pad)
        y1 = max(0, int(e["y1"]) - pad)
        x2 = min(W, int(e["x2"]) + pad)
        y2 = min(H, int(e["y2"]) + pad)
        crop_path = Path(f"/tmp/_xv_{idx:03d}.png")
        img.crop((x1, y1, x2, y2)).save(crop_path)
        paddle = EnginePred(e["text"], float(e["confidence"]))
        easy = run_easyocr(crop_path)
        tess = run_tesseract(crop_path)
        r = cross_verify(paddle, easy, tess, high_conf=args.high_conf)
        tier_count[r.tier] += 1
        reason_count[r.reason] = reason_count.get(r.reason, 0) + 1
        emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[r.tier]
        print(f"  #{idx:3d} {emoji} {r.reason:<28} "
              f"sugg={r.suggested_text!r}")
        print(f"        PP:{paddle.text!r:<35} c={paddle.conf:.2f}")
        print(f"        EZ:{easy.text!r:<35} c={easy.conf:.2f}")
        print(f"        TS:{tess.text!r:<35} c={tess.conf:.2f}")

    dt = time.time() - t0
    n = len(candidates) or 1
    print(f"\n=== Summary ({n} entries, {dt:.1f}s total, {dt/n*1000:.0f}ms/entry) ===")
    for t, c in tier_count.items():
        emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[t]
        print(f"  {emoji} {t}: {c} ({c/n*100:.0f}%)")
    print("  by reason:")
    for r, c in sorted(reason_count.items()):
        print(f"    {r}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
