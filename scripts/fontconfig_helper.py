#!/usr/bin/env python
"""Fontconfig helpers for local Office fonts used by LibreOffice/Pillow."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


_FONTCONFIG_ENV: dict[str, str] | None = None


def _office_yahei_dirs() -> list[Path]:
    roots = [
        Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts"),
        Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts"),
        Path("/Applications/Microsoft Excel.app/Contents/Resources/DFonts"),
        Path("/Applications/Microsoft OneNote.app/Contents/Resources/DFonts"),
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        if (root / "msyh.ttc").exists():
            out.append(root)
            seen.add(root)
    return out


def fontconfig_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an env where Office's YaHei TTCs are visible to fontconfig."""
    global _FONTCONFIG_ENV
    env = dict(base or os.environ)
    dirs = _office_yahei_dirs()
    if not dirs:
        return env
    if _FONTCONFIG_ENV is None:
        conf_path = Path(tempfile.gettempdir()) / "image2ppt-fonts.conf"
        dir_xml = "\n".join(f"  <dir>{d}</dir>" for d in dirs)
        conf_path.write_text(
            "<?xml version=\"1.0\"?>\n"
            "<!DOCTYPE fontconfig SYSTEM \"fonts.dtd\">\n"
            "<fontconfig>\n"
            "  <include ignore_missing=\"yes\">"
            "/opt/homebrew/etc/fonts/fonts.conf</include>\n"
            f"{dir_xml}\n"
            "</fontconfig>\n",
            encoding="utf-8",
        )
        _FONTCONFIG_ENV = {"FONTCONFIG_FILE": str(conf_path)}
    env.update(_FONTCONFIG_ENV)
    return env


def fontconfig_font_path(query: str) -> str | None:
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", query],
            check=True,
            capture_output=True,
            text=True,
            env=fontconfig_env(),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    path = Path(result.stdout.strip())
    return str(path) if path.exists() else None
