"""Rendering helpers for MNEMO-Bench results.

The core artifact for the camera-ready paper is a radar chart showing
per-memory-type success rates. This module produces the chart with
matplotlib (optional extra) and falls back to a plain-text table if
matplotlib is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from .metrics import BenchmarkReport
from .tasks import MemoryType


def plot_radar(
    report: BenchmarkReport,
    output_path: str | Path,
    title: str = "MNEMO-Bench",
) -> Path:
    """Render the per-memory-type radar chart to a PNG file.

    Requires matplotlib. Raises ImportError otherwise.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
        import numpy as np  # noqa: WPS433
    except ImportError as err:  # pragma: no cover - optional extra
        raise ImportError(
            "matplotlib is required for radar plotting; "
            "install the `embeddings` or add it yourself."
        ) from err

    labels = [mt.value for mt in MemoryType]
    values = list(report.radar_scores)

    # Close the radar polygon.
    values_closed = values + values[:1]
    angles = [
        n / float(len(labels)) * 2 * math.pi for n in range(len(labels))
    ]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"projection": "polar"},
    )
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_ylim(0, 1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)

    ax.plot(angles_closed, values_closed, linewidth=2, color="tab:blue")
    ax.fill(angles_closed, values_closed, alpha=0.25, color="tab:blue")

    ax.set_title(
        f"{title}  (overall={report.overall_score:.1%})",
        fontsize=12,
        y=1.08,
    )
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def ascii_radar(report: BenchmarkReport, width: int = 30) -> str:
    """Zero-dependency fallback: plain-text bar chart."""
    lines = [f"MNEMO-Bench (overall={report.overall_score:.1%})", ""]
    for mt, score in report.per_memory_type.items():
        filled = int(round(score * width))
        bar = "#" * filled + "-" * (width - filled)
        lines.append(f"  {mt.value:<15s} |{bar}| {score:.1%}")
    return "\n".join(lines)


def render(
    report: BenchmarkReport,
    png_path: Optional[str | Path] = None,
    title: str = "MNEMO-Bench",
) -> str:
    """Return the ASCII radar and optionally write a PNG version."""
    ascii_art = ascii_radar(report)
    if png_path is not None:
        try:
            plot_radar(report, png_path, title=title)
        except ImportError:
            pass
    return ascii_art
