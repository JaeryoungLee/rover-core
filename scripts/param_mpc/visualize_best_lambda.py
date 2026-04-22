#!/usr/bin/env python3
"""Visualize cached safe-set volume vs λ curves.

Reads the JSON files written by find_best_lambda.py from .cache/best_lambda/
and produces a single PNG. With one tag, plots one curve. With multiple tags
(typically one per uncertainty preset), overlays them all on one figure so
you can see how the optimal λ shifts with uncertainty.

Usage:
    # Single tag
    python scripts/param_mpc/visualize_best_lambda.py --tags RoverParam_MPC_RA_moderate

    # Overlay across uncertainty categories
    python scripts/param_mpc/visualize_best_lambda.py --tags \
        RoverParam_MPC_RA_zero RoverParam_MPC_RA_small \
        RoverParam_MPC_RA_moderate RoverParam_MPC_RA_harsh

    # Custom output path
    python scripts/param_mpc/visualize_best_lambda.py --tags ... --save outputs/foo.png
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


CACHE_DIR = PROJECT_ROOT / '.cache' / 'best_lambda'


def load_cache(tag: str) -> dict:
    path = CACHE_DIR / f'{tag}.json'
    if not path.exists():
        raise SystemExit(
            f"No cache for tag '{tag}' at {path}.\n"
            f"  Run: python scripts/param_mpc/find_best_lambda.py --tag {tag}"
        )
    with open(path) as f:
        return json.load(f)


def label_for(result: dict) -> str:
    """Prefer the uncertainty preset name; fall back to the tag."""
    return result.get('uncertainty_preset') or result['tag']


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tags', type=str, nargs='+', required=True,
                    help='One or more GridValue tags whose best_lambda cache to plot')
    ap.add_argument('--save', type=str, default=None,
                    help='Output PNG path (default: outputs/visualizations/best_lambda/<auto>.png)')
    ap.add_argument('--title', type=str, default=None,
                    help='Override the figure title')
    args = ap.parse_args()

    results = [load_cache(tag) for tag in args.tags]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis if len(results) > 1 else None

    for i, r in enumerate(results):
        lam = np.asarray(r['lam_axis'])
        pct = np.asarray(r['pct'])
        color = cmap(i / max(1, len(results) - 1)) if cmap is not None else 'tab:blue'
        label = label_for(r)
        ax.plot(lam, pct, '-o', color=color, linewidth=1.6, markersize=4, label=label)
        # Mark best λ for each curve
        bi = r['best_idx']
        ax.plot([lam[bi]], [pct[bi]], 'o', color=color, markersize=10,
                fillstyle='none', markeredgewidth=2)

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Safe-set volume (% of state space)')
    if args.title:
        ax.set_title(args.title)
    elif len(results) == 1:
        r = results[0]
        ax.set_title(f"[{r['tag']}]\nSafe-set volume vs λ  (t = {r['time_snapped']:.2f}s)")
    else:
        ax.set_title('Safe-set volume vs λ'
                     f'(t = {results[0]["time_snapped"]:.2f}s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', title='uncertainty')
    plt.tight_layout()

    if args.save:
        out_path = Path(args.save)
    else:
        out_dir = PROJECT_ROOT / 'outputs' / 'visualizations' / 'best_lambda'
        # Strip known uncertainty-preset suffixes so the filename reflects the
        # experiment family (e.g. Dense_RoverParam_MPC_RA), not a specific run.
        UNC_SUFFIXES = ('_zero', '_small', '_moderate', '_harsh')
        def _strip_unc(tag: str) -> str:
            for s in UNC_SUFFIXES:
                if tag.endswith(s):
                    return tag[: -len(s)]
            return tag
        stripped = [_strip_unc(t) for t in args.tags]
        # Use the common prefix of stripped names (or the single stripped tag).
        common = stripped[0]
        for s in stripped[1:]:
            i = 0
            while i < len(common) and i < len(s) and common[i] == s[i]:
                i += 1
            common = common[:i].rstrip('_')
        out_path = out_dir / f'{common}_BestParam.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')

    # Brief summary
    print()
    print(f"{'label':<14}  {'best λ':>8}  {'best %':>8}")
    print('-' * 36)
    for r in results:
        print(f"{label_for(r):<14}  {r['best_lam']:8.4f}  {r['best_pct']:7.2f}%")


if __name__ == '__main__':
    main()
