from __future__ import annotations

import os
import tempfile

import matplotlib


def ensure_headless_backend() -> None:
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
    if os.environ.get("MPLBACKEND"):
        return
    matplotlib.use("Agg", force=False)
