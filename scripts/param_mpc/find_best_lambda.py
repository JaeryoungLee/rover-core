#!/usr/bin/env python3
"""Compute and cache the safe-set volume vs λ for a parameterized BRT.

For each λ value in the GridValue's parameter axis, count cells where the value
function indicates "safe" (V < 0 for reach-avoid, V > 0 for obstacle BRT) and
multiply by the cell volume of the non-parameter dimensions. Reports the
optimal λ and saves the per-λ volume table to .cache/best_lambda/{TAG}.json.

Use scripts/param_mpc/visualize_best_lambda.py to plot the cached results
(single tag or overlay across uncertainty categories).

Usage:
    python scripts/param_mpc/find_best_lambda.py --tag RoverParam_MPC_RA_moderate
    python scripts/param_mpc/find_best_lambda.py --tag ... --time 0.0 --param-dim 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.utils.cache_loaders import load_grid_value_by_tag


CACHE_DIR = PROJECT_ROOT / '.cache' / 'best_lambda'


def compute_lambda_volumes(tag: str, time: float = 0.0,
                           safe_is_negative: bool = True,
                           param_dim: int | None = None) -> dict:
    """Compute safe-set volume per λ for a parameterized GridValue.

    Returns a JSON-serializable dict with per-λ counts/volumes/percentages and
    metadata identifying the GridValue this was computed from.
    """
    vf = load_grid_value_by_tag(tag, interpolate=False)
    state_dim = int(vf.state_dim)
    if param_dim is None:
        param_dim = state_dim - 1
    if not (0 <= param_dim < state_dim):
        raise ValueError(f'param_dim {param_dim} out of range for state_dim={state_dim}')

    times = vf._times.detach().cpu().numpy()
    t_idx = int(np.argmin(np.abs(times - float(time))))
    t_val = float(times[t_idx])

    cell_dims = []
    total_cells_per_slab = 1
    for d in range(state_dim):
        if d == param_dim:
            continue
        axis = vf._axes[d].detach().cpu().numpy()
        if axis.size < 2:
            cell_dims.append(1.0)
        else:
            cell_dims.append(float(axis[1] - axis[0]))
        total_cells_per_slab *= int(axis.size)
    cell_volume = float(np.prod(cell_dims))
    total_volume = float(total_cells_per_slab) * cell_volume

    lam_axis = vf._axes[param_dim].detach().cpu().numpy()
    values_at_t = vf._values[..., t_idx]

    safe_counts = []
    for li in range(lam_axis.size):
        sl = [slice(None)] * state_dim
        sl[param_dim] = li
        v_slab = values_at_t[tuple(sl)]
        if isinstance(v_slab, torch.Tensor):
            v_slab = v_slab.detach().cpu().numpy()
        cond = v_slab < 0.0 if safe_is_negative else v_slab > 0.0
        safe_counts.append(int(np.count_nonzero(cond)))

    safe_counts_arr = np.asarray(safe_counts, dtype=int)
    volumes = safe_counts_arr.astype(float) * cell_volume
    pct = 100.0 * volumes / total_volume
    best_idx = int(np.argmax(volumes))

    # Pull uncertainty metadata if the GridValue was built from a GridSet
    meta = getattr(vf, 'metadata', {}) or {}
    uncertainty_preset = meta.get('uncertainty_preset')
    uncertainty_limits = meta.get('uncertainty_limits')

    return {
        'tag': tag,
        'time_requested': float(time),
        'time_snapped': t_val,
        'time_idx': t_idx,
        'safe_is_negative': bool(safe_is_negative),
        'param_dim': int(param_dim),
        'state_dim': state_dim,
        'cell_volume': cell_volume,
        'total_volume': total_volume,
        'lam_axis': [float(x) for x in lam_axis],
        'safe_counts': [int(x) for x in safe_counts_arr],
        'volumes': [float(x) for x in volumes],
        'pct': [float(x) for x in pct],
        'best_idx': best_idx,
        'best_lam': float(lam_axis[best_idx]),
        'best_pct': float(pct[best_idx]),
        'uncertainty_preset': uncertainty_preset,
        'uncertainty_limits': uncertainty_limits,
    }


def cache_path_for(tag: str) -> Path:
    return CACHE_DIR / f'{tag}.json'


def save_cache(result: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(result['tag'])
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    return path


def print_report(r: dict) -> None:
    safe_str = 'V < 0' if r['safe_is_negative'] else 'V > 0'
    print(f"Tag:            {r['tag']}")
    print(f"State dim:      {r['state_dim']}")
    print(f"Param dim:      {r['param_dim']}  ({len(r['lam_axis'])} grid values)")
    print(f"Time:           requested={r['time_requested']:.3f}, "
          f"snapped={r['time_snapped']:.3f} (idx {r['time_idx']})")
    print(f"Safe condition: {safe_str}")
    print(f"Uncertainty:    {r.get('uncertainty_preset', 'unknown')}")
    print(f"Cell volume (non-param dims):  {r['cell_volume']:.5g}")
    print(f"Total volume (non-param dims): {r['total_volume']:.5g}")
    print()
    header = f"{'λ':>8}  {'cells':>10}  {'volume':>14}  {'safe %':>8}"
    print(header)
    print('-' * len(header))
    for li, (lam, n, vol, p) in enumerate(zip(r['lam_axis'], r['safe_counts'],
                                               r['volumes'], r['pct'])):
        marker = '  ← best' if li == r['best_idx'] else ''
        print(f"{lam:8.4f}  {n:10d}  {vol:14.5g}  {p:7.2f}%{marker}")
    print()
    print(f"Best λ = {r['best_lam']:.4f}  →  {r['best_pct']:.2f}% of state space")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', type=str, required=True,
                    help='GridValue cache tag (4D parameterized BRT)')
    ap.add_argument('--time', type=float, default=0.0,
                    help='Time slice to evaluate (default: 0.0 — converged BRT)')
    ap.add_argument('--reach-avoid', action='store_true',
                    help='Reach-avoid convention: safe iff V < 0 (default)')
    ap.add_argument('--avoid', action='store_true',
                    help='Obstacle BRT convention: safe iff V > 0')
    ap.add_argument('--param-dim', type=int, default=None,
                    help='Index of the parameter dim (default: last state dim)')
    args = ap.parse_args()

    if args.reach_avoid and args.avoid:
        ap.error('--reach-avoid and --avoid are mutually exclusive')
    safe_is_negative = not args.avoid

    result = compute_lambda_volumes(
        tag=args.tag, time=args.time,
        safe_is_negative=safe_is_negative, param_dim=args.param_dim,
    )
    print_report(result)
    out_path = save_cache(result)
    print(f"\nCached → {out_path}")
    print(f"Visualize with: python scripts/param_mpc/visualize_best_lambda.py --tags {args.tag}")


if __name__ == '__main__':
    main()
