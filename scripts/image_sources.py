"""Helpers for locating slide source images.

Source pages are named `page_NN.<ext>`. The pipeline writes extracted
assets and previews as PNG, but source pages can use any extension in
SUPPORTED_IMAGE_EXTENSIONS.
"""
from __future__ import annotations

import re
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)

_PAGE_RE = re.compile(r"^page_(\d+)$", re.IGNORECASE)


def supported_image_formats() -> str:
    return "PNG, JPEG/JPG, WebP, BMP, TIFF/TIF"


def _page_number(path: Path) -> str | None:
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return None
    match = _PAGE_RE.match(path.stem)
    if not match:
        return None
    return match.group(1).zfill(2)


def _sort_key(num: str) -> tuple[int, int | str]:
    if num.isdigit():
        return (0, int(num))
    return (1, num)


def discover_page_images(src_dir: Path) -> dict[str, Path]:
    """Return `{page_number: image_path}` for supported page images.

    Raises ValueError when the same page number has multiple supported
    files, because silently choosing between `page_01.png` and
    `page_01.jpg` can make a run hard to reproduce.
    """
    pages: dict[str, Path] = {}
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        num = _page_number(path)
        if num is None:
            continue
        if num in pages:
            raise ValueError(
                f"multiple source images found for page_{num}: "
                f"{pages[num].name}, {path.name}"
            )
        pages[num] = path
    return dict(sorted(pages.items(), key=lambda item: _sort_key(item[0])))


def discover_page_numbers(src_dir: Path) -> list[str]:
    return list(discover_page_images(src_dir).keys())


def find_page_image(src_dir: Path, num: str) -> Path:
    target = str(num).zfill(2)
    pages = discover_page_images(src_dir)
    if target not in pages:
        raise FileNotFoundError(
            f"no page_{target} source image found in {src_dir}; "
            f"supported formats: {supported_image_formats()}"
        )
    return pages[target]
