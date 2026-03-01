"""
Animate 3D toe endpoint trajectories for a bipedal robot.

Data shape: (T, E, 2, 1, 3)
  T = num_timesteps
  E = num_envs
  2 = left / right foot  (0=left, 1=right)
  1 = redundant dim
  3 = x, y, z (world frame)

Two files are overlaid simultaneously:
  File 1  → red  colour family
  File 2  → blue colour family

Foot convention (both files):
  left  foot → dashed line  (--)
  right foot → dotted line  (:)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent

NPY_1 = (
    ROOT / "logs/rsl_rl/lab_02.24.2026/triple_specialists/expert_0"
    / "iter_2/tests"
    / "Feb28_20_31_17_fl0.480_tl0.443_hl0.150_gcr0.840_spcf0.0079_Ht27"
    / "toe_endpoints_world_hist.npy"
)
NPY_2 = (
    ROOT / "logs/rsl_rl/lab_02.24.2026/triple_specialists/expert_1"
    / "iter_2/tests"
    / "Feb28_20_18_05_fl0.480_tl0.443_hl0.150_gcr0.840_spcf0.0057_Ht27"
    / "toe_endpoints_world_hist.npy"
)
OUT_PATH = ROOT / "toe_trajectory_animation.mp4"

# ── render config ─────────────────────────────────────────────────────────────
FPS         = 30
TRAIL       = 60     # rolling trail length (frames)
GHOST_ALPHA = 0.45   # full-trajectory ghost opacity
TRAIL_LW    = 2.0    # live-trail line width

# ── camera ────────────────────────────────────────────────────────────────────
EYE    = np.array([-0.25, -53.0,  0.6])
TARGET = np.array([ 2.0,  -54.2,  0.0])

cam    = EYE - TARGET                         # vector from target to camera
horiz  = np.sqrt(cam[0]**2 + cam[1]**2)
ELEV   = float(np.degrees(np.arctan2(cam[2], horiz)))
AZIM   = float(np.degrees(np.arctan2(cam[1], cam[0])))

# ── colour palettes ───────────────────────────────────────────────────────────
#   sample Reds / Blues colormaps avoiding the very pale / very dark ends
def make_palette(cmap_name, n, lo=0.45, hi=0.90):
    cmap = cm.get_cmap(cmap_name)
    return [cmap(v) for v in np.linspace(lo, hi, n)]

# ── load data ─────────────────────────────────────────────────────────────────
d1 = np.load(NPY_1)[:, :, :, 0, :]   # (T, E1, 2, 3)
d2 = np.load(NPY_2)[:, :, :, 0, :]   # (T, E2, 2, 3)

T  = d1.shape[0]      # both files have same T
E1 = d1.shape[1]
E2 = d2.shape[1]

red_colors  = make_palette("Reds",  E1)
blue_colors = make_palette("Blues", E2)

# ── axis limits (global across both files) ────────────────────────────────────
all_pts = np.concatenate([d1.reshape(-1, 3), d2.reshape(-1, 3)], axis=0)
mg = 0.05
x_lim = (all_pts[:, 0].min() - mg, all_pts[:, 0].max() + mg)
y_lim = (all_pts[:, 1].min() - mg, all_pts[:, 1].max() + mg)
z_lim = (max(0.0, all_pts[:, 2].min() - mg), all_pts[:, 2].max() + mg)

# ── figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8))
ax  = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X (m)", labelpad=6)
ax.set_ylabel("Y (m)", labelpad=6)
ax.set_zlabel("Z (m)", labelpad=6)
ax.set_title("Toe Endpoints – World Frame  |  expert_0 (red) vs expert_1 (blue), all envs")
ax.set_xlim(*x_lim)
ax.set_ylim(*y_lim)
ax.set_zlim(*z_lim)
ax.view_init(elev=ELEV, azim=AZIM)

# ── ghost (full-trajectory) lines ─────────────────────────────────────────────
for e in range(E1):
    c = red_colors[e]
    ax.plot(d1[:, e, 0, 0], d1[:, e, 0, 1], d1[:, e, 0, 2],
            color=c, alpha=GHOST_ALPHA, lw=0.9, ls="--")   # left
    ax.plot(d1[:, e, 1, 0], d1[:, e, 1, 1], d1[:, e, 1, 2],
            color=c, alpha=GHOST_ALPHA, lw=0.9, ls=":")    # right

for e in range(E2):
    c = blue_colors[e]
    ax.plot(d2[:, e, 0, 0], d2[:, e, 0, 1], d2[:, e, 0, 2],
            color=c, alpha=GHOST_ALPHA, lw=0.9, ls="--")   # left
    ax.plot(d2[:, e, 1, 0], d2[:, e, 1, 1], d2[:, e, 1, 2],
            color=c, alpha=GHOST_ALPHA, lw=0.9, ls=":")    # right

# ── animated artists ──────────────────────────────────────────────────────────
# Structure: trails[file_idx][env_idx] = [trail_left, trail_right]
#            dots [file_idx][env_idx] = [dot_left,   dot_right]

def make_artists(data, colors):
    """Return (trails, dots) lists for one file."""
    n = data.shape[1]
    trails, dots = [], []
    for e in range(n):
        c = colors[e]
        tl, = ax.plot([], [], [], color=c, lw=TRAIL_LW, ls="--")
        tr, = ax.plot([], [], [], color=c, lw=TRAIL_LW, ls=":")
        dl, = ax.plot([], [], [], "o", color=c, ms=6, zorder=5)
        dr, = ax.plot([], [], [], "s", color=c, ms=6, zorder=5)
        trails.append([tl, tr])
        dots.append([dl, dr])
    return trails, dots

trails1, dots1 = make_artists(d1, red_colors)
trails2, dots2 = make_artists(d2, blue_colors)

time_txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=9)

# flat list of all animated artists
_all_trail_dot = (
    [a for pair in trails1 + dots1 + trails2 + dots2 for a in pair]
)
all_artists = _all_trail_dot + [time_txt]

# ── legend ────────────────────────────────────────────────────────────────────
legend_elems = []
for e in range(E1):
    legend_elems.append(Line2D([0],[0], color=red_colors[e],  lw=2, label=f"exp0 env{e}"))
for e in range(E2):
    legend_elems.append(Line2D([0],[0], color=blue_colors[e], lw=2, label=f"exp1 env{e}"))
legend_elems += [
    Line2D([0],[0], color="k", lw=2, ls="--", label="left foot"),
    Line2D([0],[0], color="k", lw=2, ls=":",  label="right foot"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=6.5, ncol=2)

# ── animation callbacks ───────────────────────────────────────────────────────
def init():
    for art in _all_trail_dot:
        art.set_data([], [])
        art.set_3d_properties([])
    time_txt.set_text("")
    ax.view_init(elev=ELEV, azim=AZIM)
    return all_artists


def _update_file(data, trails, dots, frame, start):
    n = data.shape[1]
    for e in range(n):
        for foot_idx, (trail, dot) in enumerate(zip(trails[e], dots[e])):
            seg = data[start:frame+1, e, foot_idx, :]
            trail.set_data(seg[:, 0], seg[:, 1])
            trail.set_3d_properties(seg[:, 2])
            cur = data[frame, e, foot_idx, :]
            dot.set_data([cur[0]], [cur[1]])
            dot.set_3d_properties([cur[2]])


def update(frame):
    start = max(0, frame - TRAIL)
    _update_file(d1, trails1, dots1, frame, start)
    _update_file(d2, trails2, dots2, frame, start)
    time_txt.set_text(f"step {frame:>4d} / {T-1}")
    return all_artists


ani = animation.FuncAnimation(
    fig, update, frames=T,
    init_func=init, interval=1000 / FPS, blit=False
)

writer = animation.FFMpegWriter(fps=FPS, bitrate=2800)
ani.save(str(OUT_PATH), writer=writer)
print(f"Saved → {OUT_PATH}")
print(f"Camera  elev={ELEV:.1f}°  azim={AZIM:.1f}°")
