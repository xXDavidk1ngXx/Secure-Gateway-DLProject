"""
=============================================================================
  SECURE GATEWAY — Confidence Scale Diagram
=============================================================================

  Generates a professional horizontal gauge chart showing the three-tier
  confidence decision system used by the fusion model:

      RED   (0–50%)   → Immediate rejection
      YELLOW (50–85%) → Gray area — cosine similarity fallback
      GREEN (85–100%) → Immediate access granted

  Designed to match the visual style of the existing evaluation suite
  (dark theme, 300 DPI, same color palette).

  Output:
      evaluation/figures/7_confidence_scale.png

  Usage:
      python evaluation/confidence_scale.py

=============================================================================
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ============================================================
#  PATH SETUP
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "evaluation", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
#  VISUAL STYLE  (matches visualizer.py)
# ============================================================
STYLE_CONFIG = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
}
plt.rcParams.update(STYLE_CONFIG)

# Zone colors
RED_FILL    = "#c0392b"
RED_DARK    = "#922b21"
YELLOW_FILL = "#f39c12"
YELLOW_DARK = "#d68910"
GREEN_FILL  = "#27ae60"
GREEN_DARK  = "#1e8449"

# Accent colors
BG_COLOR    = "#1a1a2e"
PANEL_COLOR = "#16213e"
TEXT_COLOR  = "#e0e0e0"
DIM_TEXT    = "#aaaaaa"


def generate_confidence_scale(save_path=None):
    """
    Draw a professional horizontal confidence gauge with three decision zones,
    threshold markers, action labels, and a detailed description panel.
    """
    print("\n  Generating confidence scale diagram...")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-4.5, 9.2)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────
    ax.text(50, 8.7, "Three-Tier Confidence Decision System",
            fontsize=20, fontweight="bold", color=TEXT_COLOR,
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)])

    ax.text(50, 7.9, "How the Fusion Model's output confidence maps to access decisions",
            fontsize=11, color=DIM_TEXT, ha="center", va="center", style="italic")

    # ── Main gauge bar ─────────────────────────────────────
    bar_y      = 4.0
    bar_height = 2.2
    corner_r   = 0.25

    zones = [
        (0,  50, RED_FILL,    RED_DARK),
        (50, 85, YELLOW_FILL, YELLOW_DARK),
        (85, 100, GREEN_FILL,  GREEN_DARK),
    ]

    for x_start, x_end, fill, border in zones:
        width = x_end - x_start
        rect = mpatches.FancyBboxPatch(
            (x_start, bar_y), width, bar_height,
            boxstyle=f"round,pad={corner_r}",
            facecolor=fill, edgecolor=border, linewidth=2.5, alpha=0.92
        )
        ax.add_patch(rect)

    # ── Zone labels (on the bar) ───────────────────────────
    zone_labels = [
        (25,  "DENIED",       14, "white"),
        (67.5, "GRAY AREA",   14, "white"),
        (92.5, "GRANTED",     13, "white"),
    ]
    for x, text, size, color in zone_labels:
        ax.text(x, bar_y + bar_height / 2 + 0.35, text,
                ha="center", va="center", fontsize=size,
                fontweight="bold", color=color,
                path_effects=[pe.withStroke(linewidth=3, foreground="#00000088")])

    # Percentage ranges below zone labels (still on the bar)
    range_labels = [
        (25,  "0 % – 50 %"),
        (67.5, "50 % – 85 %"),
        (92.5, "85 % – 100 %"),
    ]
    for x, text in range_labels:
        ax.text(x, bar_y + bar_height / 2 - 0.55, text,
                ha="center", va="center", fontsize=10,
                color="#ffffffcc",
                path_effects=[pe.withStroke(linewidth=2, foreground="#00000066")])

    # ── Threshold markers (vertical dashed lines) ──────────
    for thresh_x, label in [(50, "50 %"), (85, "85 %")]:
        ax.plot([thresh_x, thresh_x], [bar_y - 0.4, bar_y + bar_height + 0.4],
                color="white", linewidth=2, linestyle="--", alpha=0.85, zorder=5)
        # Place badges just above the bar, not overlapping subtitle
        ax.text(thresh_x, bar_y + bar_height + 0.55, label,
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="white", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#333355",
                          edgecolor="#666688", alpha=0.95))

    # ── Percentage tick marks along the bottom ─────────────
    for pct in range(0, 101, 10):
        ax.plot([pct, pct], [bar_y - 0.15, bar_y - 0.45],
                color=DIM_TEXT, linewidth=1, alpha=0.6)
        ax.text(pct, bar_y - 0.7, f"{pct}%",
                ha="center", va="top", fontsize=8, color=DIM_TEXT, alpha=0.7)

    # ── Action description cards (below the bar) ───────────
    card_y = 0.0
    card_h = 2.6
    cards = [
        {
            "x": 0, "w": 33, "color": RED_DARK,
            "border": RED_FILL,
            "icon": "ACCESS DENIED",
            "lines": [
                "Confidence < 50 %  OR",
                "classified as \"unknown\"",
                "",
                "Immediate rejection.",
                "No fallback applied.",
            ],
        },
        {
            "x": 34, "w": 33, "color": YELLOW_DARK,
            "border": YELLOW_FILL,
            "icon": "COSINE SIMILARITY FALLBACK",
            "lines": [
                "50 % < Confidence < 85 %",
                "",
                "Compare live embeddings to",
                "enrolled profiles. Grant if",
                "face & voice similarity > 0.4",
            ],
        },
        {
            "x": 68, "w": 33, "color": GREEN_DARK,
            "border": GREEN_FILL,
            "icon": "ACCESS GRANTED",
            "lines": [
                "Confidence >= 85 %",
                "",
                "Model is highly certain.",
                "Immediate access granted.",
                "No additional checks needed.",
            ],
        },
    ]

    for card in cards:
        # Card background
        rect = mpatches.FancyBboxPatch(
            (card["x"], card_y), card["w"], card_h,
            boxstyle="round,pad=0.3",
            facecolor=card["color"], edgecolor=card["border"],
            linewidth=2, alpha=0.35
        )
        ax.add_patch(rect)

        # Card title
        cx = card["x"] + card["w"] / 2
        ax.text(cx, card_y + card_h - 0.35, card["icon"],
                ha="center", va="top", fontsize=10,
                fontweight="bold", color=card["border"])

        # Card body lines
        for i, line in enumerate(card["lines"]):
            ax.text(cx, card_y + card_h - 0.9 - i * 0.4, line,
                    ha="center", va="top", fontsize=9,
                    color=TEXT_COLOR, alpha=0.85)

    # ── Connecting arrows from bar to cards ────────────────
    arrow_props = dict(arrowstyle="-|>", lw=1.5, mutation_scale=12)

    for bar_x, card_cx, color in [
        (25, 16.5, RED_FILL),
        (67.5, 50.5, YELLOW_FILL),
        (92.5, 84.5, GREEN_FILL),
    ]:
        ax.annotate("", xy=(card_cx, card_y + card_h),
                    xytext=(bar_x, bar_y - 0.8),
                    arrowprops=dict(**arrow_props, color=color, alpha=0.6))

    # ── Gradient arrow along bottom (low → high) ──────────
    ax.annotate("", xy=(98, -3.5), xytext=(2, -3.5),
                arrowprops=dict(arrowstyle="-|>", color=DIM_TEXT,
                                lw=1.5, alpha=0.4))
    ax.text(50, -3.9, "Fusion Model Output Confidence",
            ha="center", va="top", fontsize=10, color=DIM_TEXT,
            style="italic", alpha=0.7)

    # ── Save ───────────────────────────────────────────────
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    [OK] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
    print("  Done.\n")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    output_path = os.path.join(FIGURES_DIR, "7_confidence_scale.png")
    generate_confidence_scale(save_path=output_path)
