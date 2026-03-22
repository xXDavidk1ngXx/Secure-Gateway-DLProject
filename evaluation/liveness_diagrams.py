"""
=============================================================================
  SECURE GATEWAY — Liveness Detection Diagrams
=============================================================================

  Generates three diagrams for the video presentation (Scene 6):

      8_ear_diagram.png           — EAR anatomy: open vs. closed eye with
                                    landmark points, distance labels, formula,
                                    and ratio values
      9_ear_over_time.png         — Simulated EAR-over-time plot showing a
                                    live person (with blink dips) vs. a static
                                    photo (flat line)
      10_head_pose_over_time.png  — Simulated yaw/pitch traces showing a
                                    live person (natural micro-movements)
                                    vs. a static photo (flat lines)

  These are INDEPENDENT of the existing visualizer.py and do NOT read,
  write, or modify any other files or figures.

  Output:
      evaluation/figures/8_ear_diagram.png           (300 DPI)
      evaluation/figures/9_ear_over_time.png          (300 DPI)
      evaluation/figures/10_head_pose_over_time.png   (300 DPI)

  Usage:
      python evaluation/liveness_diagrams.py

  Dependencies:
      pip install matplotlib numpy
=============================================================================
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ============================================================
#  PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "evaluation", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
#  STYLE (matches visualizer.py dark theme)
# ============================================================
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
TEXT_COLOR = "#e0e0e0"
DIM_TEXT = "#aaaaaa"
GRID_COLOR = "#2a2a4a"

STYLE_CONFIG = {
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}
plt.rcParams.update(STYLE_CONFIG)

# Accent colors
GREEN = "#27ae60"
RED = "#e74c3c"
YELLOW = "#f1c40f"
BLUE = "#2196F3"
ORANGE = "#FF9800"
CYAN = "#00BCD4"
PURPLE = "#9C27B0"

# Liveness config values (from utils/config.py)
EAR_THRESHOLD = 0.22
MIN_BLINKS = 1
EAR_VAR_THRESHOLD = 0.003
HEAD_POSE_MIN_STD = 0.3


# ============================================================
#  DIAGRAM 1 — EAR Anatomy (open eye vs. closed eye)
# ============================================================

def _draw_eye(ax, cx, cy, scale, openness, title, ear_value, color):
    """
    Draw a stylised eye with 6 labeled landmark points.

    Args:
        ax:        matplotlib axes
        cx, cy:    center of the eye
        scale:     horizontal radius
        openness:  0.0 = fully closed, 1.0 = fully open
        title:     text above the eye ("Open Eye" / "Closed Eye")
        ear_value: the EAR number to display
        color:     accent color for this eye
    """
    # --- Eye outline (almond shape via two arcs) ---
    t = np.linspace(0, np.pi, 120)
    vert = scale * 0.45 * openness  # vertical radius scales with openness

    upper_x = cx + scale * np.cos(t + np.pi)
    upper_y = cy + vert * np.sin(t)
    lower_x = cx + scale * np.cos(t + np.pi)
    lower_y = cy - vert * np.sin(t)

    # Fill
    all_x = np.concatenate([upper_x, lower_x[::-1]])
    all_y = np.concatenate([upper_y, lower_y[::-1]])
    ax.fill(all_x, all_y, color=color, alpha=0.12)

    # Outline
    ax.plot(upper_x, upper_y, color=color, linewidth=2.5, solid_capstyle="round")
    ax.plot(lower_x, lower_y, color=color, linewidth=2.5, solid_capstyle="round")

    # --- Iris / pupil (only visible when open enough) ---
    if openness > 0.25:
        iris_r = scale * 0.22 * min(openness, 1.0)
        theta = np.linspace(0, 2 * np.pi, 80)
        ax.fill(cx + iris_r * np.cos(theta), cy + iris_r * np.sin(theta),
                color="#3a2010", alpha=0.7)
        pupil_r = iris_r * 0.45
        ax.fill(cx + pupil_r * np.cos(theta), cy + pupil_r * np.sin(theta),
                color="#111111", alpha=0.9)
        # Highlight dot
        ax.plot(cx + pupil_r * 0.35, cy + pupil_r * 0.35, "o",
                color="white", markersize=3, alpha=0.8)

    # --- 6 Landmark points ---
    #     p1=outer corner, p2=upper outer, p3=upper inner,
    #     p4=inner corner, p5=lower inner,  p6=lower outer
    p1 = (cx - scale, cy)
    p4 = (cx + scale, cy)
    p2 = (cx - scale * 0.35, cy + vert * 0.92)
    p3 = (cx + scale * 0.35, cy + vert * 0.92)
    p5 = (cx + scale * 0.35, cy - vert * 0.92)
    p6 = (cx - scale * 0.35, cy - vert * 0.92)

    points = [p1, p2, p3, p4, p5, p6]
    labels = ["p1", "p2", "p3", "p4", "p5", "p6"]
    label_offsets = [
        (-0.15, -0.12),   # p1 outer
        (-0.12, 0.08),    # p2 upper outer
        (0.06, 0.08),     # p3 upper inner
        (0.08, -0.12),    # p4 inner
        (0.06, -0.14),    # p5 lower inner
        (-0.12, -0.14),   # p6 lower outer
    ]

    for (px, py), lbl, (ox, oy) in zip(points, labels, label_offsets):
        ax.plot(px, py, "o", color=YELLOW, markersize=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(px + ox * scale, py + oy * scale, lbl,
                fontsize=10, fontweight="bold", color=YELLOW,
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#000000aa")])

    # --- Distance lines (dashed) ---
    # Vertical pair 1: p2 ↔ p6
    ax.plot([p2[0], p6[0]], [p2[1], p6[1]],
            "--", color=CYAN, linewidth=1.5, alpha=0.8)
    # Vertical pair 2: p3 ↔ p5
    ax.plot([p3[0], p5[0]], [p3[1], p5[1]],
            "--", color=CYAN, linewidth=1.5, alpha=0.8)
    # Horizontal: p1 ↔ p4
    ax.plot([p1[0], p4[0]], [p1[1] - vert * 0.02, p4[1] - vert * 0.02],
            "--", color=ORANGE, linewidth=1.5, alpha=0.8)

    # Distance labels
    mid_v1 = ((p2[0] + p6[0]) / 2, (p2[1] + p6[1]) / 2)
    mid_v2 = ((p3[0] + p5[0]) / 2, (p3[1] + p5[1]) / 2)
    mid_h = ((p1[0] + p4[0]) / 2, p1[1] - vert * 0.02)

    ax.text(mid_v1[0] - scale * 0.18, mid_v1[1], "||p2-p6||",
            fontsize=8, color=CYAN, ha="center", va="center", rotation=90,
            path_effects=[pe.withStroke(linewidth=2, foreground="#000000aa")])
    ax.text(mid_v2[0] + scale * 0.18, mid_v2[1], "||p3-p5||",
            fontsize=8, color=CYAN, ha="center", va="center", rotation=90,
            path_effects=[pe.withStroke(linewidth=2, foreground="#000000aa")])
    ax.text(mid_h[0], mid_h[1] - scale * 0.2, "||p1-p4||",
            fontsize=8, color=ORANGE, ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="#000000aa")])

    # --- Title and EAR value ---
    ax.text(cx, cy + scale * 0.85, title,
            fontsize=15, fontweight="bold", color=TEXT_COLOR,
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)])

    ear_color = GREEN if ear_value >= EAR_THRESHOLD else RED
    ax.text(cx, cy - scale * 0.78, f"EAR = {ear_value:.2f}",
            fontsize=16, fontweight="bold", color=ear_color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR,
                      edgecolor=ear_color, linewidth=2, alpha=0.95))


def generate_ear_diagram(save_path=None):
    """
    Generate the EAR anatomy diagram: open eye vs. closed eye, with
    labeled landmarks, distance lines, formula, and threshold info.
    """
    print("\n  [1/3] Generating EAR anatomy diagram...")

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(-1.0, 7.5)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────
    ax.text(5.0, 7.0, "Eye Aspect Ratio (EAR) — Blink Detection",
            fontsize=22, fontweight="bold", color=TEXT_COLOR,
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)])

    ax.text(5.0, 6.35, "MediaPipe tracks 6 eye landmarks per frame to measure openness",
            fontsize=11, color=DIM_TEXT, ha="center", va="center", style="italic")

    # ── Draw the two eyes ──────────────────────────────────
    eye_scale = 1.3
    _draw_eye(ax, cx=2.5, cy=4.0, scale=eye_scale, openness=1.0,
              title="Open Eye", ear_value=0.30, color=GREEN)
    _draw_eye(ax, cx=7.5, cy=4.0, scale=eye_scale, openness=0.12,
              title="Closed Eye (Blink)", ear_value=0.15, color=RED)

    # ── Arrow between eyes ─────────────────────────────────
    ax.annotate("", xy=(5.7, 4.0), xytext=(4.3, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=DIM_TEXT,
                                lw=2.5, mutation_scale=18))
    ax.text(5.0, 4.3, "BLINK", fontsize=10, fontweight="bold",
            color=DIM_TEXT, ha="center", va="center")

    # ── Formula box ────────────────────────────────────────
    formula_y = 1.7
    formula_box = mpatches.FancyBboxPatch(
        (1.5, formula_y - 0.55), 7.0, 1.3,
        boxstyle="round,pad=0.3",
        facecolor=PANEL_COLOR, edgecolor="#555588", linewidth=2, alpha=0.95
    )
    ax.add_patch(formula_box)

    ax.text(5.0, formula_y + 0.35, "EAR  =  ( ||p2 - p6||  +  ||p3 - p5|| )  /  ( 2  *  ||p1 - p4|| )",
            fontsize=14, fontweight="bold", color=TEXT_COLOR,
            ha="center", va="center", family="monospace")
    ax.text(5.0, formula_y - 0.15,
            "Vertical distances (eyelid opening)  /  Horizontal distance (eye width)",
            fontsize=9, color=DIM_TEXT, ha="center", va="center", style="italic")

    # ── Threshold info panel ───────────────────────────────
    info_y = 0.35
    info_items = [
        (f"Threshold:  EAR < {EAR_THRESHOLD}", YELLOW),
        (f"Open eye:  EAR ~ 0.25 - 0.35", GREEN),
        (f"Blink:  EAR < 0.20", RED),
        (f"Min consecutive frames:  2  (~67ms at 30fps)", CYAN),
    ]

    total_width = sum(len(item[0]) for item in info_items) * 0.085 + 0.6
    start_x = 5.0 - total_width / 2
    cursor_x = start_x

    for text, color in info_items:
        item_w = len(text) * 0.085 + 0.15
        # Colored dot
        ax.plot(cursor_x + 0.08, info_y, "s", color=color, markersize=7)
        ax.text(cursor_x + 0.25, info_y, text,
                fontsize=8.5, color=TEXT_COLOR, ha="left", va="center")
        cursor_x += item_w + 0.35

    # ── Save ───────────────────────────────────────────────
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    [OK] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
#  DIAGRAM 2 — EAR Over Time (live person vs. photo)
# ============================================================

def _simulate_live_ear(n_frames=150, fps=30, seed=42):
    """
    Simulate realistic EAR values for a live person over n_frames.
    Includes natural micro-fluctuations and 2-3 blink events.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_frames) / fps  # seconds

    # Baseline with gentle noise
    baseline = 0.29 + 0.012 * np.sin(2 * np.pi * 0.3 * t)  # slow drift
    noise = rng.normal(0, 0.008, n_frames)  # micro-fluctuations
    ear = baseline + noise

    # Insert 3 blink events (sharp V-shaped dips)
    blink_centers = [28, 72, 118]  # frame indices
    blink_width = 4     # frames for half-blink
    blink_depth = 0.18  # how far EAR drops

    for center in blink_centers:
        for offset in range(-blink_width, blink_width + 1):
            idx = center + offset
            if 0 <= idx < n_frames:
                # Gaussian-shaped dip
                dip = blink_depth * np.exp(-0.5 * (offset / (blink_width * 0.4)) ** 2)
                ear[idx] -= dip

    ear = np.clip(ear, 0.05, 0.40)
    return t, ear


def _simulate_photo_ear(n_frames=150, fps=30, seed=99):
    """
    Simulate EAR values for a static photograph — near-constant with
    minimal sensor noise.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_frames) / fps

    # Constant with tiny sensor noise (camera jitter)
    ear = 0.28 + rng.normal(0, 0.001, n_frames)
    ear = np.clip(ear, 0.20, 0.35)
    return t, ear


def generate_ear_over_time(save_path=None):
    """
    Generate a dual-panel plot:
      Top:    EAR over time for a LIVE person (with blink dips)
      Bottom: EAR over time for a PHOTO (flat line)
    With threshold line, blink annotations, and statistics.
    """
    print("  [2/3] Generating EAR-over-time plot...")

    n_frames = 150
    fps = 30
    t_live, ear_live = _simulate_live_ear(n_frames, fps)
    t_photo, ear_photo = _simulate_photo_ear(n_frames, fps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(BG_COLOR)

    # ── Suptitle ───────────────────────────────────────────
    fig.suptitle("EAR Over Time — Live Person vs. Static Photo",
                 fontsize=20, fontweight="bold", color=TEXT_COLOR, y=0.97)

    # ================ TOP PANEL: LIVE PERSON ================
    ax1.set_facecolor(PANEL_COLOR)
    ax1.plot(t_live, ear_live, color=GREEN, linewidth=1.8, alpha=0.9,
             label="Live Person EAR")
    ax1.fill_between(t_live, ear_live, alpha=0.08, color=GREEN)

    # Threshold line
    ax1.axhline(y=EAR_THRESHOLD, color=YELLOW, linewidth=1.5,
                linestyle="--", alpha=0.8, label=f"Blink Threshold ({EAR_THRESHOLD})")

    # Mark blink events
    blink_centers = [28, 72, 118]
    for bc in blink_centers:
        blink_t = bc / fps
        blink_ear = ear_live[bc]
        ax1.plot(blink_t, blink_ear, "v", color=RED, markersize=12,
                 zorder=5, markeredgecolor="white", markeredgewidth=1)
        ax1.annotate("blink", xy=(blink_t, blink_ear),
                     xytext=(blink_t + 0.15, blink_ear - 0.035),
                     fontsize=9, fontweight="bold", color=RED,
                     arrowprops=dict(arrowstyle="-", color=RED, lw=1, alpha=0.6))

    # Stats box
    ear_std_live = np.std(ear_live)
    ear_mean_live = np.mean(ear_live)
    stats_text = (f"Mean EAR: {ear_mean_live:.3f}    "
                  f"Std: {ear_std_live:.4f}    "
                  f"Blinks: 3")
    ax1.text(0.98, 0.92, stats_text, transform=ax1.transAxes,
             fontsize=9, color=TEXT_COLOR, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                       edgecolor=GREEN, alpha=0.9))

    # Verdict badge
    ax1.text(0.02, 0.92, "LIVENESS: PASS", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", color="#1a1a2e", ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=GREEN,
                       edgecolor="white", linewidth=1.5))

    ax1.set_ylabel("Eye Aspect Ratio", fontsize=12)
    ax1.set_ylim(0.05, 0.40)
    ax1.set_title("Live Person — Natural Blink Pattern",
                  fontsize=13, fontweight="bold", color=GREEN, pad=8)
    ax1.legend(loc="lower right", framealpha=0.8,
               facecolor=BG_COLOR, edgecolor="#555")
    ax1.grid(True, alpha=0.15)

    # =============== BOTTOM PANEL: STATIC PHOTO ==============
    ax2.set_facecolor(PANEL_COLOR)
    ax2.plot(t_photo, ear_photo, color=RED, linewidth=1.8, alpha=0.9,
             label="Photo EAR")
    ax2.fill_between(t_photo, ear_photo, alpha=0.08, color=RED)

    # Threshold line
    ax2.axhline(y=EAR_THRESHOLD, color=YELLOW, linewidth=1.5,
                linestyle="--", alpha=0.8, label=f"Blink Threshold ({EAR_THRESHOLD})")

    # Flat-line annotation
    mid_t = t_photo[n_frames // 2]
    ax2.annotate("No variation — EAR stays constant\n(eyes never close)",
                 xy=(mid_t, ear_photo[n_frames // 2]),
                 xytext=(mid_t + 0.8, 0.15),
                 fontsize=10, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5),
                 ha="center")

    # Stats box
    ear_std_photo = np.std(ear_photo)
    ear_mean_photo = np.mean(ear_photo)
    stats_text = (f"Mean EAR: {ear_mean_photo:.3f}    "
                  f"Std: {ear_std_photo:.4f}    "
                  f"Blinks: 0")
    ax2.text(0.98, 0.92, stats_text, transform=ax2.transAxes,
             fontsize=9, color=TEXT_COLOR, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                       edgecolor=RED, alpha=0.9))

    # Verdict badge
    ax2.text(0.02, 0.92, "LIVENESS: FAIL", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", color="white", ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=RED,
                       edgecolor="white", linewidth=1.5))

    ax2.set_ylabel("Eye Aspect Ratio", fontsize=12)
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylim(0.05, 0.40)
    ax2.set_title("Static Photo — No Blinks Detected (Spoof Attempt)",
                  fontsize=13, fontweight="bold", color=RED, pad=8)
    ax2.legend(loc="lower right", framealpha=0.8,
               facecolor=BG_COLOR, edgecolor="#555")
    ax2.grid(True, alpha=0.15)

    # ── Save ───────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    [OK] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
#  DIAGRAM 3 — Head Pose Over Time (live person vs. photo)
# ============================================================

# Head pose config values (from utils/config.py)
POSE_MIN_STD_YAW = 0.3    # degrees
POSE_MIN_STD_PITCH = 0.3  # degrees
POSE_MIN_FRAMES = 10

# Pose landmark indices used by solvePnP (MediaPipe face mesh)
POSE_LANDMARK_NAMES = [
    ("Nose tip", 1),
    ("Chin", 152),
    ("Left eye corner", 33),
    ("Right eye corner", 263),
    ("Left mouth corner", 61),
    ("Right mouth corner", 291),
]


def _simulate_live_pose(n_frames=150, fps=30, seed=77):
    """
    Simulate realistic yaw/pitch angles for a live person.
    Includes slow drift (natural sway), micro-jitter, and a few
    deliberate small head turns.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_frames) / fps

    # Slow natural sway (breathing, body movement)
    yaw_drift = 1.2 * np.sin(2 * np.pi * 0.15 * t) + 0.6 * np.sin(2 * np.pi * 0.4 * t)
    pitch_drift = 0.8 * np.sin(2 * np.pi * 0.2 * t + 0.5) + 0.4 * np.cos(2 * np.pi * 0.35 * t)

    # Micro-jitter (involuntary)
    yaw_jitter = rng.normal(0, 0.25, n_frames)
    pitch_jitter = rng.normal(0, 0.2, n_frames)

    # A couple of small deliberate head movements
    for center, amp_y, amp_p in [(45, 2.5, 1.0), (105, -1.8, 2.0)]:
        width = 12
        for offset in range(-width, width + 1):
            idx = center + offset
            if 0 <= idx < n_frames:
                envelope = np.exp(-0.5 * (offset / (width * 0.35)) ** 2)
                yaw_drift[idx] += amp_y * envelope
                pitch_drift[idx] += amp_p * envelope

    yaw = yaw_drift + yaw_jitter
    pitch = pitch_drift + pitch_jitter

    return t, yaw, pitch


def _simulate_photo_pose(n_frames=150, fps=30, seed=88):
    """
    Simulate yaw/pitch for a static photo held in front of camera.
    Near-zero variation — only tiny sensor noise.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_frames) / fps

    # Photo has a fixed pose (maybe slightly tilted) with tiny noise
    yaw = 0.5 + rng.normal(0, 0.04, n_frames)
    pitch = -0.3 + rng.normal(0, 0.03, n_frames)

    return t, yaw, pitch


def generate_head_pose_over_time(save_path=None):
    """
    Generate a dual-panel plot:
      Top:    Yaw & Pitch over time for a LIVE person (natural movement)
      Bottom: Yaw & Pitch over time for a PHOTO (flat lines)
    With std threshold annotations, landmark info, and verdict badges.
    """
    print("  [3/3] Generating head pose over time plot...")

    n_frames = 150
    fps = 30
    t_live, yaw_live, pitch_live = _simulate_live_pose(n_frames, fps)
    t_photo, yaw_photo, pitch_photo = _simulate_photo_pose(n_frames, fps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor(BG_COLOR)

    # ── Suptitle ───────────────────────────────────────────
    fig.suptitle("Head Pose Estimation — Live Person vs. Static Photo",
                 fontsize=20, fontweight="bold", color=TEXT_COLOR, y=0.97)

    # ================ TOP PANEL: LIVE PERSON ================
    ax1.set_facecolor(PANEL_COLOR)

    ax1.plot(t_live, yaw_live, color=BLUE, linewidth=1.8, alpha=0.9,
             label="Yaw (left-right)")
    ax1.fill_between(t_live, yaw_live, alpha=0.06, color=BLUE)

    ax1.plot(t_live, pitch_live, color=ORANGE, linewidth=1.8, alpha=0.9,
             label="Pitch (up-down)")
    ax1.fill_between(t_live, pitch_live, alpha=0.06, color=ORANGE)

    # Zero reference
    ax1.axhline(y=0, color=DIM_TEXT, linewidth=0.8, linestyle=":", alpha=0.4)

    # Mark the deliberate head movements
    move_frames = [(45, "head turn"), (105, "head nod")]
    for frame_idx, label in move_frames:
        move_t = frame_idx / fps
        ax1.axvline(x=move_t, color=YELLOW, linewidth=1, linestyle="--", alpha=0.4)
        ax1.text(move_t, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else 4.0,
                 "", fontsize=1)  # placeholder for ylim calculation

    # Need to set ylim first, then place annotations
    ax1.set_ylim(-5.5, 5.5)

    for frame_idx, label in move_frames:
        move_t = frame_idx / fps
        ax1.axvline(x=move_t, color=YELLOW, linewidth=1, linestyle="--", alpha=0.4)
        y_val = yaw_live[frame_idx]
        ax1.annotate(label, xy=(move_t, y_val),
                     xytext=(move_t + 0.25, y_val + (1.5 if y_val > 0 else -1.5)),
                     fontsize=9, fontweight="bold", color=YELLOW,
                     arrowprops=dict(arrowstyle="-", color=YELLOW, lw=1, alpha=0.6))

    # Stats box
    yaw_std_live = np.std(yaw_live)
    pitch_std_live = np.std(pitch_live)
    stats_text = (f"Yaw std: {yaw_std_live:.2f}°  (threshold: {POSE_MIN_STD_YAW}°)    "
                  f"Pitch std: {pitch_std_live:.2f}°  (threshold: {POSE_MIN_STD_PITCH}°)")
    ax1.text(0.98, 0.92, stats_text, transform=ax1.transAxes,
             fontsize=9, color=TEXT_COLOR, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                       edgecolor=GREEN, alpha=0.9))

    # Verdict badge
    ax1.text(0.02, 0.92, "POSE CHECK: PASS", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", color="#1a1a2e", ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=GREEN,
                       edgecolor="white", linewidth=1.5))

    ax1.set_ylabel("Angle (degrees)", fontsize=12)
    ax1.set_title("Live Person — Natural Micro-Movements & Head Turns",
                  fontsize=13, fontweight="bold", color=GREEN, pad=8)
    ax1.legend(loc="lower right", framealpha=0.8,
               facecolor=BG_COLOR, edgecolor="#555")
    ax1.grid(True, alpha=0.15)

    # =============== BOTTOM PANEL: STATIC PHOTO ==============
    ax2.set_facecolor(PANEL_COLOR)

    ax2.plot(t_photo, yaw_photo, color=BLUE, linewidth=1.8, alpha=0.9,
             label="Yaw (left-right)")
    ax2.fill_between(t_photo, yaw_photo, alpha=0.06, color=BLUE)

    ax2.plot(t_photo, pitch_photo, color=ORANGE, linewidth=1.8, alpha=0.9,
             label="Pitch (up-down)")
    ax2.fill_between(t_photo, pitch_photo, alpha=0.06, color=ORANGE)

    # Zero reference
    ax2.axhline(y=0, color=DIM_TEXT, linewidth=0.8, linestyle=":", alpha=0.4)

    # Match y-axis range to live panel for visual contrast
    ax2.set_ylim(-5.5, 5.5)

    # Flat-line annotation
    mid_t = t_photo[n_frames // 2]
    ax2.annotate("Near-zero variation — no real movement\n"
                 "(fixed pose, only sensor noise)",
                 xy=(mid_t, yaw_photo[n_frames // 2]),
                 xytext=(mid_t + 0.8, 3.0),
                 fontsize=10, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5),
                 ha="center")

    # Stats box
    yaw_std_photo = np.std(yaw_photo)
    pitch_std_photo = np.std(pitch_photo)
    stats_text = (f"Yaw std: {yaw_std_photo:.2f}°  (threshold: {POSE_MIN_STD_YAW}°)    "
                  f"Pitch std: {pitch_std_photo:.2f}°  (threshold: {POSE_MIN_STD_PITCH}°)")
    ax2.text(0.98, 0.92, stats_text, transform=ax2.transAxes,
             fontsize=9, color=TEXT_COLOR, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                       edgecolor=RED, alpha=0.9))

    # Verdict badge
    ax2.text(0.02, 0.92, "POSE CHECK: FAIL", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", color="white", ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=RED,
                       edgecolor="white", linewidth=1.5))

    ax2.set_ylabel("Angle (degrees)", fontsize=12)
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_title("Static Photo — No Movement Detected (Spoof Attempt)",
                  fontsize=13, fontweight="bold", color=RED, pad=8)
    ax2.legend(loc="lower right", framealpha=0.8,
               facecolor=BG_COLOR, edgecolor="#555")
    ax2.grid(True, alpha=0.15)

    # ── Info footer ────────────────────────────────────────
    footer_items = [
        (f"Method:  solvePnP (6 facial landmarks)", CYAN),
        (f"Threshold:  std > {POSE_MIN_STD_YAW}° (yaw or pitch)", YELLOW),
        (f"Min frames:  {POSE_MIN_FRAMES}", DIM_TEXT),
        (f"Logic:  Yaw OR Pitch passes", GREEN),
    ]
    footer_text = "     |     ".join(item[0] for item in footer_items)
    fig.text(0.5, 0.01, footer_text,
             fontsize=9, color=DIM_TEXT, ha="center", va="bottom",
             style="italic")

    # ── Save ───────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    [OK] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SECURE GATEWAY — Liveness Detection Diagrams")
    print("=" * 60)

    ear_diagram_path = os.path.join(FIGURES_DIR, "8_ear_diagram.png")
    ear_time_path = os.path.join(FIGURES_DIR, "9_ear_over_time.png")
    head_pose_path = os.path.join(FIGURES_DIR, "10_head_pose_over_time.png")

    generate_ear_diagram(save_path=ear_diagram_path)
    generate_ear_over_time(save_path=ear_time_path)
    generate_head_pose_over_time(save_path=head_pose_path)

    print("\n  All liveness diagrams generated successfully.")
    print("=" * 60)
