"""
extract_results.py

Load NNV reachability results from reach_results_*.mat files,
compute grid-rasterized BRT volume on a 100^3 grid matching RoVer-CoRe,
and export JSON results per configuration.

Usage:
  python extract_results.py                    # process all configs
  python extract_results.py --config ntheta100 # process specific config
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

try:
    import h5py
    USE_H5PY = True
except ImportError:
    import scipy.io
    USE_H5PY = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NNV_DIR = os.path.join(RESULTS_DIR, "nnv")

GRID_LO = np.array([0.0, -5.0, -math.pi])
GRID_HI = np.array([20.0, 5.0, math.pi])
GRID_RES = 100


def wrap_theta_intervals(theta_lo, theta_hi):
    TWO_PI = 2 * math.pi
    width = theta_hi - theta_lo
    if width >= TWO_PI:
        return [(-math.pi, math.pi)]
    lo = ((theta_lo + math.pi) % TWO_PI) - math.pi
    hi = lo + width
    if hi <= math.pi:
        return [(lo, hi)]
    else:
        return [(lo, math.pi), (-math.pi, hi - TWO_PI)]


def compute_grid_volume(all_lb, all_ub, grid_lo, grid_hi, n):
    dx = (grid_hi - grid_lo) / n
    cell_volume = float(np.prod(dx))
    occupied = np.zeros((n, n, n), dtype=bool)

    for i in range(len(all_lb)):
        lo = all_lb[i]
        hi = all_ub[i]

        ix_lo = max(0, int(np.floor((lo[0] - grid_lo[0]) / dx[0])))
        ix_hi = min(n - 1, int(np.ceil((hi[0] - grid_lo[0]) / dx[0])))
        iy_lo = max(0, int(np.floor((lo[1] - grid_lo[1]) / dx[1])))
        iy_hi = min(n - 1, int(np.ceil((hi[1] - grid_lo[1]) / dx[1])))

        if ix_lo > ix_hi or iy_lo > iy_hi:
            continue

        for t_lo, t_hi in wrap_theta_intervals(lo[2], hi[2]):
            it_lo = max(0, int(np.floor((t_lo - grid_lo[2]) / dx[2])))
            it_hi = min(n - 1, int(np.ceil((t_hi - grid_lo[2]) / dx[2])))
            if it_lo <= it_hi:
                occupied[ix_lo:ix_hi + 1, iy_lo:iy_hi + 1, it_lo:it_hi + 1] = True

    n_occupied = int(occupied.sum())
    volume = n_occupied * cell_volume
    return volume, n_occupied


def process_one(mat_path):
    """Process a single reach_results_*.mat file. Returns result dict."""
    print(f"Loading: {mat_path}")

    if USE_H5PY:
        with h5py.File(mat_path, "r") as f:
            runtime = float(f["runtime"][0, 0])
            all_lb = np.array(f["all_lb_out"]).T
            all_ub = np.array(f["all_ub_out"]).T
            all_step = np.array(f["all_step_out"]).flatten()
            ntheta = int(f["ntheta"][0, 0])
            nx = int(f["nx"][0, 0])
            ny = int(f["ny"][0, 0])
            T_horizon = float(f["T"][0, 0])
    else:
        mat = scipy.io.loadmat(mat_path)
        runtime = float(mat["runtime"].flat[0])
        all_lb = mat["all_lb_out"]
        all_ub = mat["all_ub_out"]
        all_step = mat["all_step_out"].flatten()
        ntheta = int(mat["ntheta"].flat[0])
        nx = int(mat["nx"].flat[0])
        ny = int(mat["ny"].flat[0])
        T_horizon = float(mat["T"].flat[0])

    n_boxes = len(all_lb)
    n_steps = int(all_step.max()) + 1

    hull_lo = all_lb.min(axis=0)
    hull_hi = all_ub.max(axis=0)
    hull_volume = float(np.prod(hull_hi - hull_lo))

    print(f"  Config: {nx}x{ny}x{ntheta}, {n_boxes} boxes, {n_steps} timesteps")
    print(f"  Runtime: {runtime:.2f} s")
    print(f"  Hull volume: {hull_volume:.4f} m^2*rad")

    print(f"  Computing grid volume ({GRID_RES}^3) ...")
    volume, n_occupied = compute_grid_volume(all_lb, all_ub, GRID_LO, GRID_HI, GRID_RES)
    total_cells = GRID_RES ** 3
    print(f"  Grid: {n_occupied}/{total_cells} cells ({100 * n_occupied / total_cells:.2f}%)")
    print(f"  Grid volume: {volume:.4f} m^2*rad")

    return {
        "nx": nx, "ny": ny, "ntheta": ntheta, "T": T_horizon,
        "volume": round(volume, 4),
        "hull_volume": round(hull_volume, 4),
        "runtime_seconds": round(runtime, 2),
        "n_boxes": n_boxes,
        "brt_lb": hull_lo.tolist(),
        "brt_ub": hull_hi.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Process specific config tag (e.g., ntheta100)")
    args = ap.parse_args()

    if args.config:
        mat_files = [os.path.join(NNV_DIR, f"reach_results_{args.config}.mat")]
    else:
        mat_files = sorted(glob.glob(os.path.join(NNV_DIR, "reach_results_*.mat")))

    if not mat_files:
        print("No reach_results_*.mat files found.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for mat_path in mat_files:
        if not os.path.exists(mat_path):
            print(f"  Skipping (not found): {mat_path}")
            continue
        result = process_one(mat_path)
        # Extract config tag from filename
        basename = os.path.basename(mat_path)
        config_tag = basename.replace("reach_results_", "").replace(".mat", "")

        # Save per-config JSON
        json_path = os.path.join(RESULTS_DIR, f"nnv_brt_{config_tag}.json")
        with open(json_path, "w") as f:
            json.dump({
                "method": "NNV",
                "config": config_tag,
                **result,
            }, f, indent=2)
        print(f"  Saved: {json_path}")
        all_results.append(result)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "nnv_all_configs.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
