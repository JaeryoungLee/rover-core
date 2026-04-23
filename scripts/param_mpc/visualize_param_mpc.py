#!/usr/bin/env python3
"""Visualize closed-loop trajectories for different (λ, v) values to verify RoverParam_MPC.

Rolls out the Dubins car under RoverParam_MPC over the cartesian product of the
provided λ and v values. Trajectories are color-coded by whichever parameter is
varying (or both if both are multi-valued).

Usage:
    # Default: sweep λ at fixed v=4.0
    python scripts/param_mpc/visualize_param_mpc.py --lambdas 0.0 0.5 1.0 --velocities 4.0

    # Sweep v at fixed λ=0.5 (your "speed effect at fixed weight" check)
    python scripts/param_mpc/visualize_param_mpc.py --lambdas 0.5 --velocities 2.0 3.0 4.0 5.0 6.0

    # Single initial state from CLI
    python scripts/param_mpc/visualize_param_mpc.py --x0 2.5 1.0 0.0 --velocities 3.0 5.0

    # Use ICs from simulations.yaml
    python scripts/param_mpc/visualize_param_mpc.py --use-config-states
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.impl.systems.rover_param import RoverParam
from src.impl.inputs.standalone.controls.rover_param.mpc import RoverParam_MPC


# ── helpers ───────────────────────────────────────────────────────────────────

def load_baseline_initial_states() -> list[np.ndarray]:
    cfg_path = PROJECT_ROOT / 'config' / 'simulations.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    states = cfg['RoverParam']['default']['initial_states']
    # Keep only the first 3 dims (px, py, heading); we set λ, v ourselves.
    return [np.array(s[:3], dtype=np.float32) for s in states]


# ── simulation ────────────────────────────────────────────────────────────────

def rollout(
    mpc: RoverParam_MPC,
    x0_physical: np.ndarray,   # shape (3,): px, py, heading
    lam: float,
    v: float,
    dt: float,
    T: float,
) -> np.ndarray:
    """Closed-loop Euler rollout. Cold-start before each MPC solve to avoid
    warm-start local-minimum lock-in. Returns trajectory of shape (N+1, 3)."""
    import casadi as ca
    steps = int(round(T / dt))
    traj = np.empty((steps + 1, 3), dtype=np.float32)
    state = x0_physical.copy().astype(np.float32)
    traj[0] = state

    for k in range(steps):
        mpc.mpc._u0.master = ca.DM.zeros(mpc.mpc.model.n_u, 1)
        mpc._initialised = False

        # 5D state expected by RoverParam_MPC: (px, py, θ, λ, v)
        s5 = torch.tensor(
            [[state[0], state[1], state[2], lam, v]],
            dtype=torch.float32,
        )
        u = mpc.input(s5, float(k * dt)).item()

        px, py, th = state
        state = np.array([
            px + dt * v * np.cos(th),
            py + dt * v * np.sin(th),
            th + dt * u,
        ], dtype=np.float32)
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        traj[k + 1] = state

    return traj


# ── plotting ──────────────────────────────────────────────────────────────────

def _color_for(value: float, vmin: float, vmax: float, cmap) -> tuple:
    rng = vmax - vmin if vmax > vmin else 1.0
    return cmap((value - vmin) / rng)


def plot_trajectories(
    system: RoverParam,
    trajs: list[tuple[float, float, np.ndarray]],   # (lam, v, traj)
    lam_values: list[float],
    v_values: list[float],
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Obstacles
    for obs in system.obstacles:
        c = obs.center.cpu().numpy()
        ax.add_patch(plt.Circle(c, obs.radius, color='red', alpha=0.30, zorder=3))
        ax.plot(*c, 'r+', markersize=8, zorder=4)

    # Goal
    gx, gy = system.goal_state[:2].cpu().numpy()
    ax.add_patch(plt.Circle((gx, gy), system.goal_radius,
                             color='green', alpha=0.25, zorder=2))
    ax.plot(gx, gy, 'g*', markersize=12, zorder=5)

    # Decide what's the primary varying parameter for coloring/colorbar
    lam_varies = len(set(lam_values)) > 1
    v_varies = len(set(v_values)) > 1

    if v_varies and not lam_varies:
        cmap = plt.cm.viridis
        vmin, vmax = min(v_values), max(v_values)
        color_label = r'$v$ (m/s)'
        get_color = lambda lam, v: _color_for(v, vmin, vmax, cmap)
        get_label = lambda lam, v: rf'$v$={v:.1f} m/s (λ={lam:.2f})'
    elif lam_varies and not v_varies:
        cmap = plt.cm.cool
        vmin, vmax = min(lam_values), max(lam_values)
        color_label = r'$\lambda$ (obstacle weight)'
        get_color = lambda lam, v: _color_for(lam, vmin, vmax, cmap)
        get_label = lambda lam, v: rf'$\lambda$={lam:.2f} (v={v:.1f})'
    else:
        # Both vary → color by v, linestyle by λ; or single point → arbitrary color
        cmap = plt.cm.viridis
        vmin, vmax = min(v_values), max(v_values)
        color_label = r'$v$ (m/s)'
        get_color = lambda lam, v: _color_for(v, vmin, vmax, cmap)
        get_label = lambda lam, v: rf'$v$={v:.1f}, $\lambda$={lam:.2f}'

    seen = set()
    for lam, v, traj in trajs:
        color = get_color(lam, v)
        key = (round(lam, 4), round(v, 4))
        label = get_label(lam, v) if key not in seen else None
        seen.add(key)
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5,
                alpha=0.85, zorder=6, label=label)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=5, zorder=7)
        ax.plot(traj[-1, 0], traj[-1, 1], 'x', color=color, markersize=5, zorder=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=color_label, pad=0.02)
    ax.legend(loc='upper left', fontsize=10, ncol=2)

    lo = system.state_limits[0].cpu().numpy()
    hi = system.state_limits[1].cpu().numpy()
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel(r'$p_x$ (m)')
    ax.set_ylabel(r'$p_y$ (m)')
    ax.set_title(f'RoverParam MPC  (scenario={system.SCENARIO})\n(o = start, x = end)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    # Filename reflects which sweep
    if v_varies and not lam_varies:
        fname = f'mpc_v_sweep_lam{lam_values[0]:.2f}.png'
    elif lam_varies and not v_varies:
        fname = f'mpc_lam_sweep_v{v_values[0]:.1f}.png'
    else:
        fname = 'mpc_trajectories.png'
    save_path = save_dir / fname
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {save_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lambdas', nargs='+', type=float, default=[0.5],
                        help='λ values to simulate (default: 0.5)')
    parser.add_argument('--velocities', nargs='+', type=float, default=[4.0],
                        help='v values to simulate (default: 4.0)')
    parser.add_argument('--x0', nargs=3, type=float, default=None,
                        metavar=('PX', 'PY', 'HEADING'),
                        help='Single initial state (px, py, heading). '
                             'Defaults to (2.5, 1.0, 0.0).')
    parser.add_argument('--use-config-states', action='store_true',
                        help='Load initial states from config/simulations.yaml '
                             '(takes only the first 3 dims; λ and v are set from CLI)')
    parser.add_argument('--dt', type=float, default=0.1, help='integration step (s)')
    parser.add_argument('--T', type=float, default=5.0, help='simulation horizon (s)')
    parser.add_argument('--horizon', type=int, default=30, help='MPC prediction horizon steps')
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.save_dir) if args.save_dir else \
                 Path('outputs') / 'visualizations' / 'param_mpc'

    # Initial conditions
    if args.use_config_states:
        initial_states = load_baseline_initial_states()
        print(f'Loaded {len(initial_states)} initial states from simulations.yaml')
    elif args.x0 is not None:
        initial_states = [np.array(args.x0, dtype=np.float32)]
    else:
        initial_states = [np.array([2.5, 1.0, 0.0], dtype=np.float32)]

    system = RoverParam()
    mpc = RoverParam_MPC(num_workers=1, horizon=args.horizon)
    mpc.bind(system)

    print(f'Scenario: {system.SCENARIO}  ({len(system.obstacles)} obstacles)')
    print(f'Sweeping λ × v: {len(args.lambdas)} × {len(args.velocities)} '
          f'= {len(args.lambdas) * len(args.velocities)} combinations × '
          f'{len(initial_states)} ICs')

    trajs: list[tuple[float, float, np.ndarray]] = []
    total = len(initial_states) * len(args.lambdas) * len(args.velocities)
    n = 0
    for i, x0 in enumerate(initial_states):
        for lam in args.lambdas:
            for v in args.velocities:
                n += 1
                print(f'  [{n}/{total}] IC{i+1}, λ={lam:.2f}, v={v:.1f} ...',
                      end=' ', flush=True)
                traj = rollout(mpc, x0, lam, v, args.dt, args.T)
                trajs.append((lam, v, traj))
                print(f'done  (final: ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f}))')

    plot_trajectories(system, trajs, args.lambdas, args.velocities, output_dir)


if __name__ == '__main__':
    main()
