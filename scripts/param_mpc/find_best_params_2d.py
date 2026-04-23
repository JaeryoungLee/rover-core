#!/usr/bin/env python3
"""Compute and cache the safe-set volume on a 2D (param_a, param_b) grid.

For every (a, b) cell in the value function's parameter axes, count cells where
V indicates "safe" (V < 0 for reach-avoid; V > 0 for obstacle BRT) over the
non-parameter dimensions and multiply by their cell volume. Writes the full 2D
volume table to .cache/best_lambda/{TAG}__2d.json so a downstream visualizer
can plot it as a heatmap.

Defaults assume the 5D RoverParam layout:
    state = (px, py, θ, λ, v)   →  param_dim_a = 3 (λ), param_dim_b = 4 (v)

Usage:
    python scripts/param_mpc/find_best_params_2d.py --tag v_lambda_RoverParam_MPC_zero
    python scripts/param_mpc/find_best_params_2d.py --tag ... --param-dim-a 3 --param-dim-b 4
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


def compute_2d_param_volumes(tag: str, time: float = 0.0,
                             safe_is_negative: bool = True,
                             param_dim_a: int = 3,
                             param_dim_b: int = 4) -> dict:
    """Compute safe-set volumes over the (param_a, param_b) grid."""
    vf = load_grid_value_by_tag(tag, interpolate=False)
    state_dim = int(vf.state_dim)
    if param_dim_a == param_dim_b:
        raise ValueError("--param-dim-a and --param-dim-b must differ")
    for d in (param_dim_a, param_dim_b):
        if not (0 <= d < state_dim):
            raise ValueError(f'param dim {d} out of range for state_dim={state_dim}')

    # Snap time to grid
    times = vf._times.detach().cpu().numpy()
    t_idx = int(np.argmin(np.abs(times - float(time))))
    t_val = float(times[t_idx])

    # Cell volume in non-parameter dims
    cell_dims = []
    total_cells_per_slab = 1
    for d in range(state_dim):
        if d in (param_dim_a, param_dim_b):
            continue
        axis = vf._axes[d].detach().cpu().numpy()
        if axis.size < 2:
            cell_dims.append(1.0)
        else:
            cell_dims.append(float(axis[1] - axis[0]))
        total_cells_per_slab *= int(axis.size)
    cell_volume = float(np.prod(cell_dims))
    total_volume = float(total_cells_per_slab) * cell_volume

    a_axis = vf._axes[param_dim_a].detach().cpu().numpy()
    b_axis = vf._axes[param_dim_b].detach().cpu().numpy()
    values_at_t = vf._values[..., t_idx]  # shape == grid_shape

    counts = np.zeros((a_axis.size, b_axis.size), dtype=np.int64)

    # Iterate over the parameter grid; for each cell, slice and count
    for ia in range(a_axis.size):
        for ib in range(b_axis.size):
            sl = [slice(None)] * state_dim
            sl[param_dim_a] = ia
            sl[param_dim_b] = ib
            v_slab = values_at_t[tuple(sl)]
            if isinstance(v_slab, torch.Tensor):
                v_slab = v_slab.detach().cpu().numpy()
            cond = v_slab < 0.0 if safe_is_negative else v_slab > 0.0
            counts[ia, ib] = int(np.count_nonzero(cond))

    volumes = counts.astype(float) * cell_volume
    pct = 100.0 * volumes / total_volume

    flat_best = int(np.argmax(volumes))
    best_ia, best_ib = np.unravel_index(flat_best, volumes.shape)

    meta = getattr(vf, 'metadata', {}) or {}
    return {
        'tag': tag,
        'time_requested': float(time),
        'time_snapped': t_val,
        'time_idx': t_idx,
        'safe_is_negative': bool(safe_is_negative),
        'param_dim_a': int(param_dim_a),
        'param_dim_b': int(param_dim_b),
        'state_dim': state_dim,
        'cell_volume': cell_volume,
        'total_volume': total_volume,
        'a_axis': [float(x) for x in a_axis],
        'b_axis': [float(x) for x in b_axis],
        'a_label': _axis_label(vf, param_dim_a),
        'b_label': _axis_label(vf, param_dim_b),
        'counts': counts.tolist(),
        'volumes': volumes.tolist(),
        'pct': pct.tolist(),
        'best_ia': int(best_ia),
        'best_ib': int(best_ib),
        'best_a': float(a_axis[best_ia]),
        'best_b': float(b_axis[best_ib]),
        'best_pct': float(pct[best_ia, best_ib]),
        'uncertainty_preset': meta.get('uncertainty_preset'),
        'uncertainty_limits': meta.get('uncertainty_limits'),
    }


def _axis_label(vf, dim: int) -> str:
    """Best-effort axis label.

    Priority:
      1. `state_labels` from the snapshot stored in the GridValue's metadata
         (only present for caches built after the snapshot started recording labels).
      2. Live `state_labels` from instantiating the system by name.
      3. Generic 'dim_N' fallback.
    """
    meta = getattr(vf, 'metadata', {}) or {}
    sys_cfg = meta.get('system_config', {}) or {}
    labels = sys_cfg.get('state_labels')
    if labels and dim < len(labels):
        return str(labels[dim])

    # Fall back to instantiating the system to read its current state_labels
    sys_name = sys_cfg.get('class') or meta.get('system')
    if sys_name:
        try:
            from src.utils.cache_loaders import instantiate_system_by_name
            sys_inst = instantiate_system_by_name(sys_name)
            sys_labels = getattr(sys_inst, 'state_labels', None)
            if sys_labels and dim < len(sys_labels):
                return str(sys_labels[dim])
        except Exception:
            pass

    return f'dim_{dim}'


def cache_path_for(tag: str) -> Path:
    return CACHE_DIR / f'{tag}__2d.json'


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
    print(f"Param dims:     a={r['param_dim_a']} ({r['a_label']}, "
          f"{len(r['a_axis'])} pts), "
          f"b={r['param_dim_b']} ({r['b_label']}, {len(r['b_axis'])} pts)")
    print(f"Time:           requested={r['time_requested']:.3f}, "
          f"snapped={r['time_snapped']:.3f} (idx {r['time_idx']})")
    print(f"Safe condition: {safe_str}")
    print(f"Uncertainty:    {r.get('uncertainty_preset', 'unknown')}")
    print(f"Cell volume:    {r['cell_volume']:.5g}  total: {r['total_volume']:.5g}")
    print()
    a_axis = r['a_axis']
    b_axis = r['b_axis']
    pct = np.asarray(r['pct'])
    # Compact heatmap-style printout: rows = a, columns = b
    head = '         ' + ' '.join(f'{b:7.3f}' for b in b_axis)
    print(f'  a \\ b: {head}')
    for ia, a in enumerate(a_axis):
        row = ' '.join(f'{pct[ia, ib]:7.2f}' for ib in range(len(b_axis)))
        marker = '  ← row contains best' if ia == r['best_ia'] else ''
        print(f'{a:8.3f}: {row}{marker}')
    print()
    print(f"Best  a={r['best_a']:.4f}, b={r['best_b']:.4f}  →  {r['best_pct']:.2f}% safe")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', type=str, required=True,
                    help='GridValue cache tag (5D parameterized BRT)')
    ap.add_argument('--time', type=float, default=0.0,
                    help='Time slice to evaluate (default: 0.0 — converged BRT)')
    ap.add_argument('--reach-avoid', action='store_true',
                    help='Reach-avoid convention: safe iff V < 0 (default)')
    ap.add_argument('--avoid', action='store_true',
                    help='Obstacle BRT convention: safe iff V > 0')
    ap.add_argument('--param-dim-a', type=int, default=3,
                    help='First parameter dim (default: 3 = λ)')
    ap.add_argument('--param-dim-b', type=int, default=4,
                    help='Second parameter dim (default: 4 = v)')
    args = ap.parse_args()

    if args.reach_avoid and args.avoid:
        ap.error('--reach-avoid and --avoid are mutually exclusive')
    safe_is_negative = not args.avoid

    result = compute_2d_param_volumes(
        tag=args.tag, time=args.time,
        safe_is_negative=safe_is_negative,
        param_dim_a=args.param_dim_a,
        param_dim_b=args.param_dim_b,
    )
    print_report(result)
    out_path = save_cache(result)
    print(f"\nCached → {out_path}")
    print(f"Visualize with: python scripts/param_mpc/visualize_best_params_2d.py "
          f"--tags {args.tag}")


if __name__ == '__main__':
    main()
