#!/usr/bin/env python3
"""
Compute BRT volume for the RoVer-CoRe method from the RoverBaseline_BRT GridValue.

Loads the precomputed GridValue, counts grid cells where V(x, t=0) <= 0,
multiplies by cell volume (dx * dy * dtheta), and saves results to
baselines/results/rovercore_brt_volume.json and baselines/results/rovercore_runtime.json.

Usage:
  python scripts/baselines/compute_brt_volume.py
  python scripts/baselines/compute_brt_volume.py --grid-value-tag RoverBaseline_BRT
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_brt_volume(grid_value_tag: str) -> dict:
    """Load GridValue and compute BRT volume from the t=0 slice.

    Args:
        grid_value_tag: Tag of the precomputed GridValue cache.

    Returns:
        dict with keys: volume, cell_volume, brt_cell_count, total_cells,
                        brt_fraction, state_bounds, grid_resolution.
    """
    cache_path = PROJECT_ROOT / '.cache' / 'grid_values' / f'{grid_value_tag}.pkl'
    if not cache_path.exists():
        raise FileNotFoundError(f"GridValue cache not found: {cache_path}")

    print(f"Loading GridValue: {cache_path}")
    t_load_start = time.time()
    with open(cache_path, 'rb') as f:
        payload = pickle.load(f)
    t_load_end = time.time()
    print(f"  Loaded in {t_load_end - t_load_start:.2f}s")

    values = np.asarray(payload['values'])
    times = np.asarray(payload['times'])
    metadata = payload['metadata']
    grid_resolution = tuple(metadata['grid_resolution'])
    state_bounds = metadata['state_bounds']  # ([lo...], [hi...])
    bounds_lo = np.array(state_bounds[0], dtype=np.float64)
    bounds_hi = np.array(state_bounds[1], dtype=np.float64)

    # Detect time axis: first dim if it matches len(times), else last dim
    if values.shape[0] == len(times):
        vals_tgrid = values                        # (T, *grid_shape)
    else:
        vals_tgrid = np.moveaxis(values, -1, 0)   # (*grid_shape, T) → (T, *grid_shape)

    print(f"  Grid shape:  {grid_resolution}")
    print(f"  Time range:  [{float(times[0]):.3f}, {float(times[-1]):.3f}]")
    print(f"  State bounds lo: {bounds_lo}")
    print(f"  State bounds hi: {bounds_hi}")

    # t=0 is the initial time (first slice in ascending-time storage).
    # Times are saved ascending [0..T], so index 0 corresponds to t=0.
    t0_idx = int(np.argmin(np.abs(times - 0.0)))
    print(f"  t=0 index:   {t0_idx} (t={float(times[t0_idx]):.4f})")

    # Extract the t=0 value slice: shape (*grid_shape,)
    v0 = vals_tgrid[t0_idx]

    # Cell volume: product of cell sizes in each dimension
    # Each axis has `n` cells; cell size = (hi - lo) / (n - 1) is the spacing,
    # but the actual integration volume per cell uses (hi - lo) / n.
    cell_sizes = (bounds_hi - bounds_lo) / np.array(grid_resolution, dtype=np.float64)
    cell_volume = float(np.prod(cell_sizes))

    print(f"  Cell sizes:  {cell_sizes}")
    print(f"  Cell volume: {cell_volume:.6g} m^2·rad")

    # BRT = set of states where V(x, t=0) <= 0
    brt_mask = v0 <= 0.0
    brt_cell_count = int(np.sum(brt_mask))
    total_cells = int(v0.size)
    brt_fraction = brt_cell_count / total_cells

    volume = brt_cell_count * cell_volume

    print(f"\nBRT volume computation:")
    print(f"  BRT cells:   {brt_cell_count:,} / {total_cells:,} ({brt_fraction:.2%})")
    print(f"  BRT volume:  {volume:.4f} m^2·rad")

    return {
        'volume': volume,
        'cell_volume': cell_volume,
        'cell_sizes': cell_sizes.tolist(),
        'brt_cell_count': brt_cell_count,
        'total_cells': total_cells,
        'brt_fraction': brt_fraction,
        'state_bounds': state_bounds,
        'grid_resolution': list(grid_resolution),
        'grid_value_tag': grid_value_tag,
        't0_index': t0_idx,
        't0_actual': float(times[t0_idx]),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--grid-value-tag',
        type=str,
        default='RoverBaseline_BRT',
        help='GridValue cache tag to load (default: RoverBaseline_BRT)',
    )
    p.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for output JSON files (default: baselines/results/)',
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'baselines' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Timed computation (Steps 2–4 of the pipeline are already cached) -----
    # Wall-clock time covers: loading the GridSet/GridValue that was produced by
    # Steps 2 (CROWN bounds), 3 (clamp), and 4 (HJ PDE solve).  Since those
    # artifacts are already cached, we measure the load + volume extraction here
    # as the "analysis" runtime.  For reporting purposes, the runtime logged in
    # rovercore_runtime.json refers to the total HJ pipeline (build_grid_set +
    # constrain + build_grid_value), which was observed wall-clock separately.
    # Here we time the volume extraction step itself.
    t_start = time.time()
    result = compute_brt_volume(args.grid_value_tag)
    t_end = time.time()
    runtime_s = t_end - t_start

    print(f"\nRuntime (volume extraction): {runtime_s:.2f}s")

    # ----- Save BRT volume -----
    volume_path = out_dir / 'rovercore_brt_volume.json'
    volume_payload = {
        'method': 'RoVer-CoRe',
        'volume': round(result['volume'], 6),
        'unit': 'm^2*rad',
        'brt_cell_count': result['brt_cell_count'],
        'total_cells': result['total_cells'],
        'brt_fraction': round(result['brt_fraction'], 6),
        'grid_resolution': result['grid_resolution'],
        'grid_value_tag': result['grid_value_tag'],
    }
    with open(volume_path, 'w') as f:
        json.dump(volume_payload, f, indent=2)
    print(f"\nSaved BRT volume: {volume_path}")
    print(json.dumps(volume_payload, indent=2))

    # ----- Save runtime -----
    runtime_path = out_dir / 'rovercore_runtime.json'
    runtime_payload = {
        'method': 'RoVer-CoRe',
        'runtime_volume_extraction_s': round(runtime_s, 3),
        'note': (
            'Volume extraction runtime only. '
            'Full HJ pipeline (CROWN GridSet + constrain + HJ PDE solve) '
            'ran separately; see pipeline logs for wall-clock times.'
        ),
        'grid_value_tag': result['grid_value_tag'],
    }
    with open(runtime_path, 'w') as f:
        json.dump(runtime_payload, f, indent=2)
    print(f"\nSaved runtime:    {runtime_path}")
    print(json.dumps(runtime_payload, indent=2))


if __name__ == '__main__':
    main()
