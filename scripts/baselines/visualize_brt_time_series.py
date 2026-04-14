#!/usr/bin/env python3
"""
BRT time-series visualization: show BRT growth at T=1,2,3,4,5s
for both RoVer-CoRe and NNV, in a single θ=0 slice plot.

Each method is one color; each time horizon is a progressively lighter shade.

Usage:
  python scripts/baselines/visualize_brt_time_series.py
  python scripts/baselines/visualize_brt_time_series.py --nnv-config T5_ntheta100
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / 'baselines' / 'results'
CACHE_DIR = PROJECT_ROOT / '.cache'

PX_LIM = (12.0, 20.0)
PY_LIM = (-4.0, 4.0)

TIME_HORIZONS = [0.5, 1.0, 1.5, 2.0, 2.5]  # seconds


def _load_rovercore_t5():
    """Load RoVer-CoRe T=5s GridValue. Returns (axes, values_4d, times)."""
    cache_path = CACHE_DIR / 'grid_values' / 'RoverBaseline_BRT_T5.pkl'
    if not cache_path.exists():
        print(f'Warning: {cache_path} not found')
        return None

    with open(cache_path, 'rb') as f:
        payload = pickle.load(f)

    values = np.asarray(payload['values'])
    times = np.asarray(payload['times'])
    metadata = payload['metadata']
    axes = [np.asarray(ax) for ax in metadata['grid_coordinate_vectors']]

    if values.shape[0] == len(times):
        vals_tgrid = values  # [T, npx, npy, ntheta]
    else:
        vals_tgrid = np.moveaxis(values, -1, 0)

    return axes, vals_tgrid, times


def _get_rovercore_slice(axes, vals_tgrid, times, T_horizon, theta_slice=0.0):
    """Get V2d at the time corresponding to T_horizon seconds of backward reach."""
    # In HJ convention: time goes from T_max to 0. The BRT at horizon T
    # is the V<=0 set at time index closest to (T_max - T_horizon)...
    # Actually, times are stored as [0, ..., T_max]. The BRT at horizon T
    # is at times[i] closest to T_max - T_horizon? No — the value at t=0
    # is the full T_max BRT. The value at t=T_max is just the obstacle.
    # So for horizon T, we want the value at time index t = T_max - T.
    T_max = times[-1]
    target_t = T_max - T_horizon
    t_idx = int(np.argmin(np.abs(times - target_t)))

    V3d = vals_tgrid[t_idx]
    th_idx = int(np.argmin(np.abs(axes[2] - theta_slice)))
    V2d = V3d[:, :, th_idx]

    X, Y = np.meshgrid(axes[0], axes[1], indexing='ij')
    return X, Y, V2d


def _load_nnv_boxes_by_step(mat_path):
    """Load NNV boxes grouped by step. Returns (step_boxes_dict, ntheta, controlPeriod)."""
    import h5py
    with h5py.File(str(mat_path), 'r') as f:
        all_lb = np.array(f['all_lb_out']).T
        all_ub = np.array(f['all_ub_out']).T
        all_step = np.array(f['all_step_out']).flatten().astype(int)
        ntheta = int(f['ntheta'][0, 0])
        cp = float(f['controlPeriod'][0, 0])

    # Group by step
    step_boxes = {}
    for i in range(len(all_lb)):
        s = all_step[i]
        if s not in step_boxes:
            step_boxes[s] = ([], [])
        step_boxes[s][0].append(all_lb[i])
        step_boxes[s][1].append(all_ub[i])

    return step_boxes, ntheta, cp


def _get_nnv_boxes_at_horizon(step_boxes, control_period, T_horizon, theta_slice=0.0):
    """Get (px_lo, px_hi, py_lo, py_hi) boxes for the BRT at time horizon T."""
    max_step = int(round(T_horizon / control_period))
    TWO_PI = 2 * math.pi

    boxes_out = []
    for step_idx in range(max_step + 1):
        if step_idx not in step_boxes:
            continue
        lbs, ubs = step_boxes[step_idx]
        for lb, ub in zip(lbs, ubs):
            theta_lo, theta_hi = lb[2], ub[2]
            width = theta_hi - theta_lo
            lo = ((theta_lo + math.pi) % TWO_PI) - math.pi
            hi = lo + width
            if lo <= theta_slice <= hi or lo <= theta_slice + TWO_PI <= hi:
                boxes_out.append((lb[0], ub[0], lb[1], ub[1]))

    return boxes_out


def _rasterize_boxes(boxes, resolution=500, clip_px=PX_LIM, clip_py=PY_LIM):
    n = resolution
    px_edges = np.linspace(clip_px[0], clip_px[1], n + 1)
    py_edges = np.linspace(clip_py[0], clip_py[1], n + 1)
    dx = px_edges[1] - px_edges[0]
    dy = py_edges[1] - py_edges[0]
    occ = np.zeros((n, n), dtype=bool)

    for px_lo, px_hi, py_lo, py_hi in boxes:
        ix_lo = max(0, int(np.floor((px_lo - clip_px[0]) / dx)))
        ix_hi = min(n - 1, int(np.ceil((px_hi - clip_px[0]) / dx)))
        iy_lo = max(0, int(np.floor((py_lo - clip_py[0]) / dy)))
        iy_hi = min(n - 1, int(np.ceil((py_hi - clip_py[0]) / dy)))
        if ix_lo <= ix_hi and iy_lo <= iy_hi:
            occ[ix_lo:ix_hi + 1, iy_lo:iy_hi + 1] = True

    return occ, px_edges, py_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--theta-slice', type=float, default=0.0)
    ap.add_argument('--nnv-config', type=str, default='T5_ntheta100',
                    help='NNV config tag (default: T5_ntheta100)')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else RESULTS_DIR / 'brt_time_series.png'

    fig, ax = plt.subplots(figsize=(10, 7))
    legend_handles = []

    # Time color progression: darker range for better visibility
    time_cmap = plt.cm.Reds
    time_colors = [time_cmap(0.35 + 0.55 * i / (len(TIME_HORIZONS) - 1))
                   for i in range(len(TIME_HORIZONS))]
    n_T = len(TIME_HORIZONS)

    # ---- RoVer-CoRe (solid lines + fill, largest first) ----
    print('Loading RoVer-CoRe T=5s ...')
    rc = _load_rovercore_t5()
    if rc is not None:
        axes, vals_tgrid, times = rc
        for i in range(n_T - 1, -1, -1):  # largest T first (background)
            T = TIME_HORIZONS[i]
            X, Y, V2d = _get_rovercore_slice(axes, vals_tgrid, times, T,
                                              theta_slice=args.theta_slice)
            vmin = float(np.nanmin(V2d))
            ax.contour(X, Y, V2d, levels=[0.0], colors=[time_colors[i]],
                      linewidths=2.5, linestyles='solid', zorder=9 + i)

    # ---- NNV boxed (dashed lines + fill, largest first) ----
    nnv_mat = RESULTS_DIR / 'nnv' / f'reach_results_{args.nnv_config}.mat'
    if nnv_mat.exists():
        print(f'Loading NNV {args.nnv_config} ...')
        step_boxes, ntheta, cp = _load_nnv_boxes_by_step(str(nnv_mat))

        for i in range(n_T - 1, -1, -1):  # largest T first
            T = TIME_HORIZONS[i]
            boxes = _get_nnv_boxes_at_horizon(step_boxes, cp, T,
                                              theta_slice=args.theta_slice)
            if not boxes:
                continue
            occ, px_edges, py_edges = _rasterize_boxes(boxes, resolution=500)
            if not occ.any():
                continue

            px_c = 0.5 * (px_edges[:-1] + px_edges[1:])
            py_c = 0.5 * (py_edges[:-1] + py_edges[1:])
            X, Y = np.meshgrid(px_c, py_c, indexing='ij')

            ax.contour(X, Y, occ.astype(float), levels=[0.5],
                      colors=[time_colors[i]], linewidths=2.5,
                      linestyles='dashed', zorder=6 + i)
    else:
        print(f'  NNV config not found: {nnv_mat}')

    # ---- Legend: method style + time color ----
    legend_handles.append(Line2D([], [], color='gray', linewidth=2.5,
                                  linestyle='solid', label='RoVer-CoRe'))
    legend_handles.append(Line2D([], [], color='gray', linewidth=2.5,
                                  linestyle='dashed', label='NNV v2.0'))
    for i, T in enumerate(TIME_HORIZONS):
        legend_handles.append(Line2D([], [], color=time_colors[i], linewidth=2.5,
                                      linestyle='solid', label=f'T = {T}s'))

    # ---- Obstacle ----
    ax.add_patch(Circle((16.0, 0.0), 1.0, facecolor='red', alpha=0.3,
                         edgecolor='darkred', linewidth=1.5, zorder=10))
    legend_handles.append(mpatches.Patch(facecolor='red', alpha=0.3,
                                          edgecolor='darkred', label='Obstacle'))

    # ---- Format ----
    ax.set_xlim(*PX_LIM)
    ax.set_ylim(*PY_LIM)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$p_x$ (m)', fontsize=12)
    ax.set_ylabel(r'$p_y$ (m)', fontsize=12)
    ax.set_title(f'BRT Growth Over Time ($\\theta=0$ slice)\n'
                 f'T = {", ".join(str(t) + "s" for t in TIME_HORIZONS)}',
                 fontsize=12)

    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
