"""
Interactive 3D toe trajectory visualisation using Plotly.

Output: a self-contained HTML file — open in any browser.
Controls:
  - Click + drag  → rotate
  - Scroll        → zoom
  - Play/Pause    → animate over timesteps
  - Slider        → scrub to any timestep

Data: two expert files overlaid.
  File 1 (expert_0) → red  colour family
  File 2 (expert_1) → blue colour family
  Left foot  → dashed line
  Right foot → dotted line
"""

import numpy as np
import plotly.graph_objects as go
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
OUT_HTML = ROOT / "toe_trajectory_interactive.html"

# ── config ────────────────────────────────────────────────────────────────────
TRAIL        = 60     # how many past steps to show as trail
FRAME_STEP   = 2      # animate every Nth frame (1 = all, higher = faster/smaller file)
FRAME_DUR_MS = 40     # ms per animation frame  (≈25 fps)
GHOST_ALPHA  = 0.20   # full-path ghost opacity

# ── camera (eye / target → Plotly camera dict) ───────────────────────────────
EYE_XYZ    = np.array([-0.25, -53.0,  0.6])
TARGET_XYZ = np.array([ 2.0,  -54.2,  0.0])

# Plotly uses normalised eye coords relative to scene centre.
# We pass the raw world coords — Plotly will normalise internally via `center`.
camera = dict(
    eye    = dict(x=float(EYE_XYZ[0]),    y=float(EYE_XYZ[1]),    z=float(EYE_XYZ[2])),
    center = dict(x=float(TARGET_XYZ[0]), y=float(TARGET_XYZ[1]), z=float(TARGET_XYZ[2])),
    up     = dict(x=0, y=0, z=1),
)

# ── colour palettes ───────────────────────────────────────────────────────────
def hex_palette(base_hex, n, lo=0.35, hi=0.90):
    """Interpolate n shades of a hue (expressed as a full hex colour)."""
    import matplotlib.colors as mc
    import matplotlib.cm as cm
    # use a named matplotlib colormap
    cmap = cm.get_cmap(base_hex)
    return [
        "rgba({},{},{},{})".format(
            *[int(c * 255) for c in cmap(v)[:3]], 1.0
        )
        for v in np.linspace(lo, hi, n)
    ]

# ── load data ─────────────────────────────────────────────────────────────────
d1 = np.load(NPY_1)[:, :, :, 0, :]   # (T, E1, 2, 3)
d2 = np.load(NPY_2)[:, :, :, 0, :]   # (T, E2, 2, 3)

T      = d1.shape[0]
E1, E2 = d1.shape[1], d2.shape[1]

red_colors  = hex_palette("Reds",  E1)
blue_colors = hex_palette("Blues", E2)

FOOT_DASH = ["dash", "dot"]   # 0=left→dash, 1=right→dot
FOOT_NAME = ["L", "R"]
DOT_SYM   = ["circle", "square"]

frames_idx = list(range(0, T, FRAME_STEP))

# ── helpers ───────────────────────────────────────────────────────────────────
def _trail_slice(data, e, foot, frame):
    start = max(0, frame - TRAIL)
    seg = data[start:frame+1, e, foot, :]
    return seg[:, 0].tolist(), seg[:, 1].tolist(), seg[:, 2].tolist()

def _cur(data, e, foot, frame):
    p = data[frame, e, foot, :]
    return [float(p[0])], [float(p[1])], [float(p[2])]

# ── build initial traces (frame 0) ───────────────────────────────────────────
# Ghost traces first (static, never updated in frames)
initial_traces = []

for e in range(E1):
    for foot in range(2):
        initial_traces.append(go.Scatter3d(
            x=d1[:, e, foot, 0], y=d1[:, e, foot, 1], z=d1[:, e, foot, 2],
            mode="lines",
            line=dict(color=red_colors[e], width=1, dash=FOOT_DASH[foot]),
            opacity=GHOST_ALPHA,
            showlegend=False, hoverinfo="skip",
        ))

for e in range(E2):
    for foot in range(2):
        initial_traces.append(go.Scatter3d(
            x=d2[:, e, foot, 0], y=d2[:, e, foot, 1], z=d2[:, e, foot, 2],
            mode="lines",
            line=dict(color=blue_colors[e], width=1, dash=FOOT_DASH[foot]),
            opacity=GHOST_ALPHA,
            showlegend=False, hoverinfo="skip",
        ))

n_ghost = len(initial_traces)  # number of static traces

# Live traces: trail lines + current-position dots
# Order: (file1 envs × 2 feet × trail+dot) then (file2 envs × 2 feet × trail+dot)
def add_live_traces(data, colors, file_label, frame=0):
    traces = []
    n = data.shape[1]
    for e in range(n):
        for foot in range(2):
            xs, ys, zs = _trail_slice(data, e, foot, frame)
            cx, cy, cz = _cur(data, e, foot, frame)
            label = f"{file_label} env{e} {FOOT_NAME[foot]}"
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=colors[e], width=3, dash=FOOT_DASH[foot]),
                name=label, legendgroup=label, showlegend=True,
            ))
            traces.append(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode="markers",
                marker=dict(color=colors[e], size=5, symbol=DOT_SYM[foot]),
                name=label, legendgroup=label, showlegend=False,
                hovertemplate=f"{label}<br>x=%{{x:.3f}} y=%{{y:.3f}} z=%{{z:.3f}}<extra></extra>",
            ))
    return traces

initial_traces += add_live_traces(d1, red_colors,  "exp0", frame=0)
initial_traces += add_live_traces(d2, blue_colors, "exp1", frame=0)

# ── build animation frames ───────────────────────────────────────────────────
print(f"Building {len(frames_idx)} animation frames …")

plotly_frames = []
for fi, frame in enumerate(frames_idx):
    frame_data = []

    # Ghost traces are static — include empty updates so indices stay aligned
    for _ in range(n_ghost):
        frame_data.append(go.Scatter3d(x=[], y=[], z=[]))  # no-op placeholder

    # Live traces
    for data, colors, file_label in [(d1, red_colors, "exp0"), (d2, blue_colors, "exp1")]:
        n = data.shape[1]
        for e in range(n):
            for foot in range(2):
                xs, ys, zs = _trail_slice(data, e, foot, frame)
                cx, cy, cz = _cur(data, e, foot, frame)
                frame_data.append(go.Scatter3d(x=xs, y=ys, z=zs))
                frame_data.append(go.Scatter3d(x=cx, y=cy, z=cz))

    plotly_frames.append(go.Frame(
        data=frame_data,
        name=str(frame),
        traces=list(range(len(initial_traces))),
    ))

    if fi % 50 == 0:
        print(f"  frame {frame}/{T-1}")

# ── slider + play/pause buttons ──────────────────────────────────────────────
sliders = [dict(
    active=0,
    currentvalue=dict(prefix="step: ", font=dict(size=12)),
    pad=dict(t=50),
    steps=[
        dict(
            method="animate",
            label=str(frame),
            args=[[str(frame)],
                  dict(mode="immediate", frame=dict(duration=FRAME_DUR_MS, redraw=True),
                       transition=dict(duration=0))],
        )
        for frame in frames_idx
    ],
)]

buttons = [
    dict(label="▶ Play",  method="animate",
         args=[None, dict(frame=dict(duration=FRAME_DUR_MS, redraw=True),
                          fromcurrent=True, transition=dict(duration=0))]),
    dict(label="⏸ Pause", method="animate",
         args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0))]),
]

# ── layout ────────────────────────────────────────────────────────────────────
layout = go.Layout(
    title="Toe Trajectories — expert_0 (red) vs expert_1 (blue) | all envs",
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        camera=camera,
        aspectmode="auto",
    ),
    updatemenus=[dict(
        type="buttons", showactive=False,
        y=1.05, x=0.1, xanchor="right",
        buttons=buttons,
    )],
    sliders=sliders,
    legend=dict(font=dict(size=9), tracegroupgap=2),
    margin=dict(l=0, r=0, b=60, t=60),
)

# ── assemble and export ───────────────────────────────────────────────────────
fig = go.Figure(data=initial_traces, layout=layout, frames=plotly_frames)
fig.write_html(str(OUT_HTML), include_plotlyjs=True, full_html=True)
print(f"\nSaved → {OUT_HTML}")
