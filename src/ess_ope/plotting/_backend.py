from __future__ import annotations

import os

import matplotlib


def ensure_headless_backend() -> None:
    if os.environ.get("MPLBACKEND"):
        return
    matplotlib.use("Agg", force=False)
