"""Generate ``paper/figures/architecture.png`` from matplotlib.

The PNG is referenced by ``paper/main.tex`` as the architecture
figure. Re-run this script whenever the three-tier layout changes.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib is not installed; skip figure generation.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def draw_box(cx, cy, w, h, label, color):
        rect = plt.Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            linewidth=1.3,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10)

    # Observation
    draw_box(2.2, 5.2, 3.4, 0.7, "Current Observation", "#fff1c4")

    # Three memory tiers
    draw_box(2.2, 4.2, 3.4, 0.7, "Hot (seconds)\nvideo + tactile buffer", "#e7f0fb")
    draw_box(2.2, 3.2, 3.4, 0.7, "Cold (minutes)\nlanguage memory", "#e7f0fb")
    draw_box(2.2, 2.2, 3.4, 0.7, "Persistent (days)\nFAISS summary store", "#d9ede3")

    # Planner + executor column
    draw_box(7.2, 4.0, 3.4, 0.7, "Hierarchical Planner\n(Qwen3-VL)", "#ffe6cc")
    draw_box(7.2, 3.0, 3.4, 0.7, "Executor + SubtaskEnd\nClassifier", "#ffe6cc")
    draw_box(7.2, 2.0, 3.4, 0.7, "Action Chunk", "#fff1c4")

    # Vertical arrows on the left (between tiers)
    for y_start, y_end in [(5.2, 4.2), (4.2, 3.2), (3.2, 2.2)]:
        ax.annotate(
            "",
            xy=(2.2, y_end + 0.35),
            xytext=(2.2, y_start - 0.35),
            arrowprops=dict(arrowstyle="-|>", color="black"),
        )

    # Right column vertical arrows
    for y_start, y_end in [(4.0, 3.0), (3.0, 2.0)]:
        ax.annotate(
            "",
            xy=(7.2, y_end + 0.35),
            xytext=(7.2, y_start - 0.35),
            arrowprops=dict(arrowstyle="-|>", color="black"),
        )

    # Convergence arrows from three tiers into planner
    for y in [4.2, 3.2, 2.2]:
        ax.annotate(
            "",
            xy=(5.5, 4.0),
            xytext=(3.9, y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="tab:blue",
                alpha=0.8,
            ),
        )

    out = Path("paper/figures/architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
