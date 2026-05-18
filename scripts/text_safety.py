"""Text normalization for PPT render compatibility.

LibreOffice's PDF export can route a few separator-dot codepoints through
symbol fallback even when the selected font's cmap contains them. Normalize
that family to a nearby codepoint that renders reliably with Microsoft YaHei.
This is intentionally narrow: bullets used as list markers stay as bullets.
"""

from __future__ import annotations

PPT_SEPARATOR_DOT_FALLBACKS = {
    "\u00b7": "\u2027",  # MIDDLE DOT -> HYPHENATION POINT
    "\u2219": "\u2027",  # BULLET OPERATOR -> HYPHENATION POINT
    "\u22c5": "\u2027",  # DOT OPERATOR -> HYPHENATION POINT
}

PPT_TEXT_TRANSLATION = str.maketrans(PPT_SEPARATOR_DOT_FALLBACKS)


def ppt_safe_text(value: object) -> str:
    """Normalize text before measuring or writing PPT runs."""
    return str(value or "").translate(PPT_TEXT_TRANSLATION)
