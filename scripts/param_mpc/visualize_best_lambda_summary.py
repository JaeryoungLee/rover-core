#!/usr/bin/env python3
"""Plot best λ vs uncertainty, one line per scenario.

Reads the JSON files written by find_best_lambda.py (under .cache/best_lambda/)
for each (scenario, uncertainty) pair and produces one summary figure showing
how the optimal λ shifts with uncertainty for each obstacle scenario.

Tag pattern assumed:
    {SCENARIO}_{SYSTEM}_{CONTROLLER}_{UNCERTAINTY}
e.g.
    sparse_RoverParam_MPC_zero
    medium_RoverParam_MPC_moderate
    dense_RoverParam_MPC_harsh

Usage:
    python scripts/param_mpc/visualize_best_lambda_summary.py
    python scripts/param_mpc/visualize_best_lambda_summary.py --scenarios sparse medium dense
    python scripts/param_mpc/visualize_best_lambda_summary.py --uncertainties zero small moderate harsh
    python scripts/param_mpc/visualize_best_lambda_summary.py --system RoverParam --controller MPC
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenarios', nargs='+', default=['sparse', 'medium', 'dense'])
    ap.add_argument('--uncertainties', nargs='+',
                    default=['zero', 'small', 'moderate', 'harsh'])
    ap.add_argument('--system', type=str, default='RoverParam')
    ap.add_argument('--controller', type=str, default='MPC')
    ap.add_argument('--save', type=str, default=None,
                    help='Output PNG path (default: outputs/visualizations/best_lambda/<system>_<controller>_summary.png)')
    args = ap.parse_args()

    # Pull (best_lam, best_pct) for each (scenario, uncertainty) pair, skipping any missing files
    table: dict[str, dict[str, dict]] = {s: {} for s in args.scenarios}
    missing: list[str] = []
    for s in args.scenarios:
        for u in args.uncertainties:
            tag = f'{s}_{args.system}_{args.controller}_{u}'
            path = CACHE_DIR / f'{tag}.json'
            if not path.exists():
                missing.append(tag)
                continue
            with open(path) as f:
                table[s][u] = json.load(f)

    if missing:
        print('⚠ Missing caches (skipped):')
        for tag in missing:
            print(f'    {tag}')
        print('  Run: python scripts/param_mpc/find_best_lambda.py --tag <missing tag>')
        print()

    # Filter scenarios that have at least one data point
    scenarios = [s for s in args.scenarios if table[s]]
    if not scenarios:
        raise SystemExit('No data found — check --scenarios / --uncertainties / cache paths.')

    # ── One figure, two side-by-side subplots: best λ and best safe-set % ──
    cmap = plt.cm.tab10
    x_idx = list(range(len(args.uncertainties)))

    fig, (ax_lam, ax_pct) = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)

    for i, s in enumerate(scenarios):
        color = cmap(i % 10)
        lam_ys = [table[s].get(u, {}).get('best_lam', np.nan) for u in args.uncertainties]
        pct_ys = [table[s].get(u, {}).get('best_pct', np.nan) for u in args.uncertainties]

        ax_lam.plot(x_idx, lam_ys, '-o', color=color, linewidth=2, markersize=8, label=s)
        for xi, y in zip(x_idx, lam_ys):
            if not np.isnan(y):
                ax_lam.annotate(f'{y:.2f}', xy=(xi, y), xytext=(0, 8),
                                textcoords='offset points', ha='center',
                                fontsize=8, color=color)

        ax_pct.plot(x_idx, pct_ys, '-o', color=color, linewidth=2, markersize=8, label=s)
        for xi, y in zip(x_idx, pct_ys):
            if not np.isnan(y):
                ax_pct.annotate(f'{y:.1f}%', xy=(xi, y), xytext=(0, 8),
                                textcoords='offset points', ha='center',
                                fontsize=8, color=color)

    for ax in (ax_lam, ax_pct):
        ax.set_xticks(x_idx)
        ax.set_xticklabels(args.uncertainties)
        ax.set_xlabel('Uncertainty preset')
        ax.grid(True, alpha=0.3)
        ax.legend(title='scenario', loc='best')

    ax_lam.set_ylabel(r'Optimal $\lambda^*$')
    ax_lam.set_ylim(-0.05, 1.05)
    ax_lam.set_title('Optimal controller weight')

    ax_pct.set_ylabel('Best safe-set volume (% of state space)')
    ax_pct.set_title('Achievable safe-set volume')

    fig.suptitle(f'{args.system} — {args.controller}', fontsize=13)
    plt.tight_layout()

    if args.save:
        out_path = Path(args.save)
    else:
        out_dir = PROJECT_ROOT / 'outputs' / 'visualizations' / 'best_lambda'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{args.system}_{args.controller}_summary.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')

    # ── Tiny stdout summary table ───────────────────────────────────────
    print()
    head = f"{'scenario':<10}  " + '  '.join(f'{u:>10}' for u in args.uncertainties)
    print(head)
    print('-' * len(head))
    for s in scenarios:
        cells = []
        for u in args.uncertainties:
            r = table[s].get(u)
            cells.append(f'{r["best_lam"]:.2f} ({r["best_pct"]:.1f}%)' if r else '   —')
        print(f"{s:<10}  " + '  '.join(f'{c:>10}' for c in cells))


if __name__ == '__main__':
    main()
