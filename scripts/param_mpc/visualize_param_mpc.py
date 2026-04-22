#!/usr/bin/env python3
"""Visualize closed-loop trajectories for different λ values to verify RoverParam_MPC.

Rolls out the Dubins car under RoverParam_MPC for several λ values.  By default
uses the initial conditions registered in config/simulations.yaml (RoverBaseline
preset), so results are directly comparable to the baseline simulation plots.

Usage:
    python scripts/param_mpc/visualize_param_mpc.py
    python scripts/param_mpc/visualize_param_mpc.py --lambdas 0.0 0.5 1.0
    python scripts/param_mpc/visualize_param_mpc.py --x0 2.0 0.0 0.0   # single IC
    python scripts/param_mpc/visualize_param_mpc.py --save-dir outputs/my_dir
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
    """Load RoverBaseline initial states from config/simulations.yaml."""
    cfg_path = PROJECT_ROOT / 'config' / 'simulations.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    states = cfg['RoverParam']['default']['initial_states']
    return [np.array(s, dtype=np.float32) for s in states]


# ── simulation ────────────────────────────────────────────────────────────────

def rollout(
    system: RoverParam,
    mpc: RoverParam_MPC,
    x0_physical: np.ndarray,   # shape (3,): px, py, heading
    lam: float,
    dt: float,
    T: float,
) -> np.ndarray:
    """Roll out closed-loop Euler integration; return trajectory array (N+1, 3).

    Uses cold-start (u0→0) before every solve to prevent warm-start from
    locking the NLP into a bad local minimum.
    """
    import casadi as ca
    steps = int(round(T / dt))
    traj = np.empty((steps + 1, 3), dtype=np.float32)
    state = x0_physical.copy().astype(np.float32)
    traj[0] = state

    for k in range(steps):
        mpc.mpc._u0.master = ca.DM.zeros(mpc.mpc.model.n_u, 1)
        mpc._initialised = False

        s4 = torch.tensor([[state[0], state[1], state[2], lam]], dtype=torch.float32)
        u = mpc.input(s4, float(k * dt)).item()

        v = system.v
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

def plot_trajectories(
    system: RoverParam,
    # list of (lam, traj) — may come from multiple initial conditions
    trajs: list[tuple[float, np.ndarray]],
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

    cmap = plt.cm.cool
    # norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # Draw trajectories; only label each λ value once for the legend
    seen_lams: set[float] = set()
    for lam, traj in trajs:
        # color = cmap(norm(lam))
        color = cmap(lam)
        label = rf'$\lambda$={lam:.2f}' if lam not in seen_lams else None
        seen_lams.add(lam)
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5,
                alpha=0.85, zorder=6, label=label)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=5, zorder=7)
        ax.plot(traj[-1, 0], traj[-1, 1], 'x', color=color, markersize=5, zorder=7)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r'$\lambda$ (obstacle weight)', pad=0.02)

    ax.legend(loc='upper left', fontsize=12)

    lo = system.state_limits[0].cpu().numpy()
    hi = system.state_limits[1].cpu().numpy()
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    # ax.set_xlim(12.5, 20.5)
    # ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(r'$p_x$ (m)')
    ax.set_ylabel(r'$p_y$ (m)')
    ax.set_title('RoverParam MPC — baseline initial conditions\n(o = start, x = end)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'mpc.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {save_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--lambdas', nargs='+', type=float,
        default=[0.0, 0.5, 1.0],
        help='λ values to simulate (default: 0.0 0.5 1.0)',
    )
    parser.add_argument(
        '--x0', nargs=3, type=float, default=None,
        metavar=('PX', 'PY', 'HEADING'),
        help='single initial state; if omitted, uses all states from simulations.yaml',
    )
    parser.add_argument('--dt',      type=float, default=0.1, help='integration step (s)')
    parser.add_argument('--T',       type=float, default=10.0, help='simulation horizon (s)')
    parser.add_argument('--horizon', type=int,   default=30,  help='MPC prediction horizon steps')
    parser.add_argument('--save-dir', type=str,  default=None)
    args = parser.parse_args()

    output_dir = Path(args.save_dir) if args.save_dir else \
                 Path('outputs') / 'visualizations' / 'param_mpc'

    # Initial conditions
    if args.x0 is not None:
        initial_states = [np.array(args.x0, dtype=np.float32)]
    else:
        # initial_states = load_baseline_initial_states()
        initial_states = [np.array([2.5, 1.00, 0.00])]
        # print(f'Loaded {len(initial_states)} initial states from simulations.yaml')

    system = RoverParam()
    mpc    = RoverParam_MPC(num_workers=1, horizon=args.horizon)
    mpc.bind(system)

    trajs: list[tuple[float, np.ndarray]] = []
    for i, x0 in enumerate(initial_states):
        for lam in args.lambdas:
            print(f'  IC {i+1}/{len(initial_states)}, λ={lam:.2f} ...', end=' ', flush=True)
            traj = rollout(system, mpc, x0, lam, args.dt, args.T)
            trajs.append((lam, traj))
            print(f'done  (final: {traj[-1, :2]})')

    plot_trajectories(system, trajs, output_dir)


if __name__ == '__main__':
    main()
