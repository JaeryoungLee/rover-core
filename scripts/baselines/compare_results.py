#!/usr/bin/env python3
"""
Compile comparison results for RoVer-CoRe vs NNV baseline.

Loads BRT volume JSONs from baselines/results/, computes volume ratios,
prints a formatted comparison table, and saves results as
comparison_table.json and comparison_table.md.

Usage:
  python scripts/baselines/compare_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'baselines' / 'results'

METHOD_ORDER = [
    'RoVer-CoRe',
    'NNV',
]

METHOD_FILES = {
    'RoVer-CoRe':    'rovercore_brt_volume.json',
    'NNV':           'nnv_brt_T5_ntheta100.json',
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_results(results_dir: Path) -> list[dict]:
    """Load volume for all methods and return a list of dicts."""
    rows = []
    for method in METHOD_ORDER:
        vol_file = METHOD_FILES[method]
        vol_path = results_dir / vol_file

        if not vol_path.exists():
            print(f'Warning: volume file not found: {vol_path}')
            continue

        vol_data = _load_json(vol_path)
        volume = float(vol_data['volume'])
        rows.append({
            'method': method,
            'volume': volume,
        })

    return rows


def compute_ratios(rows: list[dict]) -> list[dict]:
    """Add ratio vs RoVer-CoRe to each row."""
    rover_vol = None
    for row in rows:
        if row['method'] == 'RoVer-CoRe':
            rover_vol = row['volume']
            break
    if rover_vol is None or rover_vol == 0:
        raise ValueError('RoVer-CoRe volume not found or zero.')

    for row in rows:
        row['ratio'] = row['volume'] / rover_vol

    return rows


def print_table(rows: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Method':<20} {'BRT Volume (m2*rad)':>22} {'Ratio vs RoVer-CoRe':>22}"
    sep = '-' * len(header)
    print()
    print('Baseline Verification Comparison')
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(f"{row['method']:<20} {row['volume']:>22.2f} {row['ratio']:>22.2f}x")
    print(sep)
    print()


def build_markdown_table(rows: list[dict]) -> str:
    """Return a Markdown comparison table string."""
    lines = []
    lines.append('| Method | BRT Volume (m2*rad) | Ratio vs RoVer-CoRe |')
    lines.append('|--------|--------------------:|--------------------:|')
    for row in rows:
        volume = f"{row['volume']:.2f}"
        ratio = f"{row['ratio']:.2f}x"
        lines.append(f"| {row['method']} | {volume} | {ratio} |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(
        description='Compile BRT comparison results.'
    )
    ap.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Path to baselines/results/ directory (default: auto-detected)',
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    if not results_dir.exists():
        ap.error(f'Results directory not found: {results_dir}')

    rows = load_all_results(results_dir)
    rows = compute_ratios(rows)
    print_table(rows)

    comparison = {
        'description': 'BRT volume comparison for RoVer-CoRe vs NNV v2.0 baseline.',
        'unit_volume': 'm^2*rad',
        'reference_method': 'RoVer-CoRe',
        'methods': [
            {
                'method': row['method'],
                'volume': row['volume'],
                'ratio_vs_rovercore': round(row['ratio'], 4),
            }
            for row in rows
        ],
    }

    json_out = results_dir / 'comparison_table.json'
    with open(json_out, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'Saved JSON: {json_out}')

    md_lines = [
        '# Baseline Comparison Results',
        '',
        'BRT volume comparison for RoVer-CoRe vs NNV v2.0 baseline.',
        '',
        '**Unit:** m2*rad  |  **Reference:** RoVer-CoRe (optimal HJ BRT)',
        '',
        build_markdown_table(rows),
        '',
        '_Volume ratio = method volume / RoVer-CoRe volume. '
        'Ratios > 1.0 indicate conservatism (over-approximation)._',
        '_All volumes computed via grid rasterization on a 100^3 grid '
        'matching RoVer-CoRe\'s state-space discretization._',
    ]
    md_out = results_dir / 'comparison_table.md'
    md_out.write_text('\n'.join(md_lines) + '\n')
    print(f'Saved Markdown: {md_out}')


if __name__ == '__main__':
    main()
