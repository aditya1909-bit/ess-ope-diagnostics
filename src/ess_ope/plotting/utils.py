from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig: plt.Figure, output_dir: str | Path, name: str, dpi: int = 200) -> None:
    out = ensure_dir(output_dir)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
