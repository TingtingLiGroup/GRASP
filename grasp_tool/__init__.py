"""GRASP tool package.

This repository is being reorganized into an installable Python package.
The canonical import name is `grasp_tool` (distribution name: `grasp-tool`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

__all__ = ["__version__"]

try:
    __version__ = _pkg_version("grasp-tool")
except PackageNotFoundError:
    # Fallback for editable/uninstalled usage.
    __version__ = "0.1.1"
