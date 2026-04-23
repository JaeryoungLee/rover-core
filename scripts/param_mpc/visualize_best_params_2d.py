#!/usr/bin/env python3
"""Visualize cached 2D safe-set volume heatmaps over (param_a, param_b),
with a companion BRT slice at the optimal parameters next to each heatmap.

Reads JSON files written by find_best_params_2d.py from
.cache/best_lambda/{TAG}__2d.json. The companion plot loads the original
GridValue cache and renders V(px, py, t=0) at the best (a, b) values, in the
same RdYlBu / TwoSlopeNorm style used by visualize_grid_value.py.

Usage:
    python scripts/param_mpc/visualize_best_params_2d.py --tags v_lambda_RoverParam_MPC_zero
    python scripts/param_mpc/visualize_best_params_2d.py --tags \
        v_lambda_RoverParam_MPC_zero v_lambda_RoverParam_MPC_small \
        v_lambda_RoverParam_MPC_moderate v_lambda_RoverParam_MPC_harsh \
        --shared-colorbar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from src.utils.cache_loaders import load_grid_value_by_tag, instantiate_system_by_name
from src.utils.grids import nearest_axis_indices, nearest_time_index


CACHE_DIR = PROJECT_ROOT / '.cache' / 'best_lambda'


# ── data loading ──────────────────────────────────────────────────────────────

def load_cache(tag: str) -> dict:
    path = CACHE_DIR / f'{tag}__2d.json'
    if not path.exists():
        raise SystemExit(
            f"No 2D cache for tag '{tag}' at {path}.\n"
            f"  Run: python scripts/param_mpc/find_best_params_2d.py --tag {tag}"
        )
    with open(path) as f:
        return json.load(f)


def _strip_unc_suffix(tag: str) -> str:
    for s in ('_zero', '_small', '_moderate', '_harsh'):
        if tag.endswith(s):
            return tag[: -len(s)]
    return tag


# ── plotting helpers ──────────────────────────────────────────────────────────

def _draw_heatmap(ax, r: dict, vmin: float, vmax: float, cmap='viridis',
                   show_xlabel=True, show_ylabel=True) -> 'AxesImage':
    a = np.asarray(r['a_axis'])
    b = np.asarray(r['b_axis'])
    pct = np.asarray(r['pct'])
    im = ax.imshow(
        pct, origin='lower', aspect='auto',
        extent=[b.min(), b.max(), a.min(), a.max()],
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    ax.plot(r['best_b'], r['best_a'], marker='*', color='red',
            markersize=14, markeredgecolor='white', markeredgewidth=1.0)
    if show_xlabel:
        ax.set_xlabel(r['b_label'])
    if show_ylabel:
        ax.set_ylabel(r['a_label'])
    title = (r.get('uncertainty_preset') or r['tag'])
    ax.set_title(f'{title}\nbest=({r["b_label"]}={r["best_b"]:.2f}, '
                 f'{r["a_label"]}={r["best_a"]:.2f}) → {r["best_pct"]:.1f}%')
    return im


def _draw_brt_slice_at_best(ax, r: dict, time: float = 0.0):
    """Render V(px, py) at (a, b) = (best_a, best_b) using RdYlBu, zero level set, obstacles."""
    tag = r['tag']
    vf = load_grid_value_by_tag(tag, interpolate=False)

    # Locate fixed-axis indices: best_a on axis param_dim_a, best_b on axis param_dim_b,
    # heading at the middle (or at exact 0.0 if present).
    da, db = int(r['param_dim_a']), int(r['param_dim_b'])
    # snap best_a / best_b to grid
    axa = vf._axes[da]
    axb = vf._axes[db]
    ia = int(nearest_axis_indices(axa, torch.tensor([float(r['best_a'])], dtype=axa.dtype))[0].item())
    ib = int(nearest_axis_indices(axb, torch.tensor([float(r['best_b'])], dtype=axb.dtype))[0].item())

    # We assume px=0, py=1, θ=2 are the spatial dims for slicing. Anything else (not 0,1,da,db)
    # gets fixed at its middle index.
    sl = []
    for d in range(vf.state_dim):
        if d in (0, 1):
            sl.append(slice(None))
        elif d == da:
            sl.append(ia)
        elif d == db:
            sl.append(ib)
        else:
            # other state dim (e.g. heading θ) — fix at middle
            sl.append(vf.grid_shape[d] // 2)

    t_idx = int(nearest_time_index(vf._times, float(time))[0].item())
    val_at_t = vf._values[..., t_idx]
    arr = val_at_t[tuple(sl)]
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)

    X, Y = np.meshgrid(
        vf._axes[0].detach().cpu().numpy(),
        vf._axes[1].detach().cpu().numpy(),
        indexing='ij',
    )
    vabs = max(abs(float(np.min(arr))), abs(float(np.max(arr))), 1e-9)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    levels = np.linspace(-vabs, vabs, 21)
    cf = ax.contourf(X, Y, arr, levels=levels, cmap='RdYlBu', norm=norm)
    ax.contour(X, Y, arr, levels=[0.0], colors='black', linewidths=2)

    # Overlay obstacles + goal if we can instantiate the system
    sys_name = (vf.metadata or {}).get('system_config', {}).get('class') \
        or (vf.metadata or {}).get('system')
    try:
        if sys_name:
            system = instantiate_system_by_name(sys_name)
            from src.utils.obstacles import draw_obstacles_2d, draw_goal_2d
            draw_obstacles_2d(ax, system, zorder=10)
            draw_goal_2d(ax, system, zorder=10)
    except Exception:
        pass

    ax.set_xlim(float(X.min()), float(X.max()))
    ax.set_ylim(float(Y.min()), float(Y.max()))
    ax.set_aspect('equal')
    ax.set_xlabel(r'$p_x$ (m)')
    ax.set_ylabel(r'$p_y$ (m)')
    ax.set_title(f'BRT @ best ({r["a_label"]}={r["best_a"]:.2f}, '
                 f'{r["b_label"]}={r["best_b"]:.2f})\nt = {float(vf._times[t_idx].item()):.2f}s')
    return cf


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tags', type=str, nargs='+', required=True)
    ap.add_argument('--save', type=str, default=None)
    ap.add_argument('--shared-colorbar', action='store_true',
                    help='Use the same color scale for the heatmaps across all panels')
    ap.add_argument('--time', type=float, default=0.0,
                    help='Time slice for the BRT companion plots (default: 0.0)')
    args = ap.parse_args()

    results = [load_cache(t) for t in args.tags]
    n = len(results)

    if args.shared_colorbar:
        vmin = min(np.min(np.asarray(r['pct'])) for r in results)
        vmax = max(np.max(np.asarray(r['pct'])) for r in results)
    else:
        vmin = vmax = None

    # Layout: 2 rows × n cols
    #   row 0: heatmap of safe-set % over (b, a)
    #   row 1: BRT V(px, py) at best (a, b)
    fig, axes = plt.subplots(
        2, n, figsize=(4.6 * n, 9.0), squeeze=False,
    )
    heat_axes = axes[0]
    brt_axes  = axes[1]

    last_heat_im = None
    for i, (ax, r) in enumerate(zip(heat_axes, results)):
        if args.shared_colorbar:
            last_heat_im = _draw_heatmap(ax, r, vmin, vmax,
                                          show_ylabel=(i == 0))
        else:
            pct = np.asarray(r['pct'])
            last_heat_im = _draw_heatmap(ax, r, float(pct.min()), float(pct.max()),
                                          show_ylabel=(i == 0))
            fig.colorbar(last_heat_im, ax=ax, label='safe %', shrink=0.85)
    if args.shared_colorbar and last_heat_im is not None:
        fig.colorbar(last_heat_im, ax=heat_axes.tolist(), label='safe %', shrink=0.85)

    # BRT row — each panel gets its own colorbar (V is on different scales per uncertainty)
    for ax, r in zip(brt_axes, results):
        cf = _draw_brt_slice_at_best(ax, r, time=args.time)
        fig.colorbar(cf, ax=ax, label='Value', shrink=0.85)

    a_lbl = results[0]['a_label']
    b_lbl = results[0]['b_label']
    plt.tight_layout()

    if args.save:
        out_path = Path(args.save)
    else:
        out_dir = PROJECT_ROOT / 'outputs' / 'visualizations' / 'best_lambda'
        out_dir.mkdir(parents=True, exist_ok=True)
        if n == 1:
            out_path = out_dir / f'{results[0]["tag"]}_2d_with_brt.png'
        else:
            common = _strip_unc_suffix(args.tags[0])
            out_path = out_dir / f'{common}_2d_with_brt.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')

    # Stdout summary — use actual parameter names for the column headers
    print()
    a_hdr = f'best {results[0]["a_label"]}'
    b_hdr = f'best {results[0]["b_label"]}'
    print(f"{'tag':<40}  {a_hdr:>10}  {b_hdr:>10}  {'best %':>8}")
    print('-' * (40 + 4 + 12 + 12 + 8))
    for r in results:
        label = (r.get('uncertainty_preset') or r['tag'])
        print(f"{label:<40}  {r['best_a']:10.3f}  {r['best_b']:10.3f}  {r['best_pct']:7.2f}%")


if __name__ == '__main__':
    main()
