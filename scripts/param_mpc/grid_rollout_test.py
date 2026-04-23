#!/usr/bin/env python3
"""Sweep a 2D grid of initial states, run worst-case rollouts, and compare against the BRT.

For each (px, py) in the requested rectangle (with fixed θ, λ, v, and uncertainty
channel from --value-tag), this script:

  1. Computes V(IC, t=0) — the BRT's verdict
  2. Rolls out a worst-case-perception trajectory
  3. Classifies the outcome (success / collision / missed_goal)
  4. Reports:
        - failure rate
        - soundness violations  (V<0 but failed) — should be 0
        - conservatism cushion (V≥0 but succeeded) — quantifies over-conservatism
        - a saved figure: grid coloured by outcome × BRT verdict

Usage:
    python scripts/param_mpc/grid_rollout_test.py \\
        --value-tag v_lambda_RoverParam_MPC_moderate \\
        --control-tag v_lambda_RoverParam_MPC \\
        --x-range 7.5 10.0 --y-range 2.0 4.0 \\
        --nx 6 --ny 5 \\
        --lam 1.0 --v 3.0 --theta 0.0 \\
        --T 5.0 --dt 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.cache_loaders import (
    load_grid_value_by_tag,
    load_grid_input_by_tag,
)
from src.impl.systems.rover_param import RoverParam
from src.impl.inputs.derived.optimal_input_from_value import OptimalInputFromValue
from src.impl.inputs.standalone.common.zero import ZeroInput
from src.core.simulators import simulate_euler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--value-tag', required=True, help='GridValue tag (BRT)')
    ap.add_argument('--control-tag', required=True, help='GridInput tag for MPC controls')
    ap.add_argument('--x-range', type=float, nargs=2, default=[7.5, 10.0])
    ap.add_argument('--y-range', type=float, nargs=2, default=[2.0, 4.0])
    ap.add_argument('--nx', type=int, default=6)
    ap.add_argument('--ny', type=int, default=5)
    ap.add_argument('--theta', type=float, default=0.0)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--v', type=float, default=3.0)
    ap.add_argument('--T', type=float, default=5.0, help='Sim horizon (s)')
    ap.add_argument('--dt', type=float, default=0.05)
    ap.add_argument('--save', type=str, default=None)
    args = ap.parse_args()

    print(f'Loading BRT (value tag): {args.value_tag}')
    vf = load_grid_value_by_tag(args.value_tag, interpolate=True)

    print(f'Loading control GridInput: {args.control_tag}')
    system = RoverParam()
    control = load_grid_input_by_tag(args.control_tag, system, interpolate=True)
    uncertainty = OptimalInputFromValue(vf)
    uncertainty.bind(system)
    disturbance = ZeroInput()
    disturbance.type = 'disturbance'
    disturbance.bind(system)

    # Build initial state grid
    xs = np.linspace(args.x_range[0], args.x_range[1], args.nx)
    ys = np.linspace(args.y_range[0], args.y_range[1], args.ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    n = XX.size
    init_states = torch.zeros((n, 5), dtype=torch.float32)
    init_states[:, 0] = torch.from_numpy(XX.flatten().astype(np.float32))
    init_states[:, 1] = torch.from_numpy(YY.flatten().astype(np.float32))
    init_states[:, 2] = float(args.theta)
    init_states[:, 3] = float(args.lam)
    init_states[:, 4] = float(args.v)

    # V at each IC
    with torch.no_grad():
        v_ic = vf.value(init_states, 0.0).cpu().numpy()

    # times at which we'll later evaluate V along the trajectory
    times_np = None  # filled in after rollout (depends on result.times)

    # Roll out
    num_steps = int(round(args.T / args.dt))
    print(f'\nSimulating {n} trajectories  (T={args.T}, dt={args.dt}, steps={num_steps})')
    result = simulate_euler(
        system=system,
        control=control,
        disturbance=disturbance,
        uncertainty=uncertainty,
        dt=args.dt,
        num_steps=num_steps,
        initial_state=init_states,
        show_progress=True,
        leave_progress=False,
        enforce_system_constraints=True,
    )
    states = result.states.cpu().numpy()  # [n, T+1, 5]
    times_np = result.times.cpu().numpy() if isinstance(result.times, torch.Tensor) else np.asarray(result.times)

    # ── V along trajectory: min and final value per trajectory ──
    # Under HJ DP, V should be non-increasing along the worst-case rollout.
    v_min_traj = np.zeros(n, dtype=np.float64)
    v_final_traj = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for i in range(n):
            traj_t = torch.from_numpy(states[i]).float()  # [T+1, 5]
            v_seq = vf.value(traj_t, torch.from_numpy(times_np).float()).cpu().numpy()
            v_min_traj[i] = float(v_seq.min())
            v_final_traj[i] = float(v_seq[-1])

    # Classify outcomes
    outcomes = []
    for i in range(n):
        traj = torch.from_numpy(states[i]).float()
        f_vals = system.failure_function(traj).numpy()
        t_vals = system.target_function(traj).numpy()
        collided = bool((f_vals <= 0.0).any())
        reached = bool((t_vals <= 0.0).any())
        if reached:
            if collided:
                t_idx = int(np.argmax(t_vals <= 0.0))
                c_idx = int(np.argmax(f_vals <= 0.0))
                outcomes.append('success' if t_idx < c_idx else 'collision')
            else:
                outcomes.append('success')
        elif collided:
            outcomes.append('collision')
        else:
            outcomes.append('missed_goal')

    # Aggregate stats
    n_success = outcomes.count('success')
    n_collision = outcomes.count('collision')
    n_missed = outcomes.count('missed_goal')
    n_brt_safe = int((v_ic < 0).sum())
    n_brt_unsafe = int((v_ic >= 0).sum())

    soundness_violations = sum(1 for v, o in zip(v_ic, outcomes) if v < 0 and o != 'success')
    cushion_count = sum(1 for v, o in zip(v_ic, outcomes) if v >= 0 and o == 'success')

    # Per-cell breakdown
    print('\n── Per-trajectory ──')
    print(f"  {'#':>3}  {'px':>5}  {'py':>5}  {'V(IC)':>8}  {'V_min':>8}  {'V_final':>8}  "
          f"{'verdict':<11}  {'outcome':<12}  {'flag':<22}")
    for i in range(n):
        flag = ''
        if v_ic[i] < 0 and outcomes[i] != 'success':
            flag = '⚠ SOUNDNESS BUG'
        elif v_ic[i] >= 0 and outcomes[i] == 'success':
            flag = '⛌ conservative cushion'
        verdict = 'safe (V<0)' if v_ic[i] < 0 else 'unsafe (V≥0)'
        print(f"  {i:>3}  {init_states[i,0]:>5.2f}  {init_states[i,1]:>5.2f}  "
              f"{v_ic[i]:>8.4f}  {v_min_traj[i]:>8.4f}  {v_final_traj[i]:>8.4f}  "
              f"{verdict:<11}  {outcomes[i]:<12}  {flag:<22}")

    print('\n── Summary ──')
    print(f"  total              : {n}")
    print(f"  outcomes:")
    print(f"    success          : {n_success}/{n}  ({100.0*n_success/n:.1f}%)")
    print(f"    collision        : {n_collision}/{n}  ({100.0*n_collision/n:.1f}%)")
    print(f"    missed_goal      : {n_missed}/{n}  ({100.0*n_missed/n:.1f}%)")
    print(f"  BRT verdict at IC:")
    print(f"    V<0  (safe)      : {n_brt_safe}/{n}")
    print(f"    V≥0  (unsafe)    : {n_brt_unsafe}/{n}")
    print(f"  soundness violations (V<0 but failed): {soundness_violations}  "
          f"{'⚠ SHOULD BE 0' if soundness_violations > 0 else '✓'}")
    print(f"  conservatism cushion (V≥0 but succeeded): {cushion_count}/{n_brt_unsafe} "
          f"({100.0*cushion_count/max(1,n_brt_unsafe):.1f}% of BRT-unsafe)")

    # Aggregate stats over V along trajectory.
    # Under HJ DP, worst-case rollouts should give V_min ≈ V(IC) (V is non-increasing
    # along the worst trajectory; min is at the *end*). If V_min << V(IC), the trajectory
    # entered a less-safe region than its IC — possible when sim drifts off-grid.
    inc_count = int(np.sum((v_min_traj > v_ic + 1e-3)))   # V actually went UP
    print(f"  trajectories where V_min > V(IC) (V increased): {inc_count}/{n}  "
          f"{'⚠ adversary not at worst case' if inc_count > 0 else '✓'}")
    print(f"  mean V drop along traj  (V(IC) - V_min):  "
          f"{float(np.mean(v_ic - v_min_traj)):.4f}  "
          f"(if huge, sim drifted into low-V region; if ~0, V stayed flat)")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    # Obstacles
    for obs in system.obstacles:
        c = obs.center.cpu().numpy()
        ax.add_patch(plt.Circle(c, obs.radius, color='red', alpha=0.3, zorder=2))
    # Goal
    gx, gy = system.goal_state[:2].cpu().numpy()
    ax.add_patch(plt.Circle((gx, gy), system.goal_radius, color='green', alpha=0.3, zorder=2))

    # Trajectories — colored by outcome
    color_for = {'success': 'tab:green', 'collision': 'tab:red',
                 'missed_goal': 'tab:gray'}
    for i in range(n):
        c = color_for.get(outcomes[i], 'k')
        ax.plot(states[i, :, 0], states[i, :, 1], color=c, linewidth=0.6, alpha=0.6, zorder=3)

    # IC markers — symbol by BRT verdict, edge by outcome
    for i in range(n):
        marker = 'o' if v_ic[i] < 0 else 's'   # circle = BRT-safe, square = BRT-unsafe
        edge = color_for.get(outcomes[i], 'k')
        face = 'white'
        # Highlight cushion + soundness bugs
        if v_ic[i] < 0 and outcomes[i] != 'success':
            face = 'red'  # SOUNDNESS bug — should never happen
        elif v_ic[i] >= 0 and outcomes[i] == 'success':
            face = 'gold'  # conservatism cushion
        ax.plot(init_states[i, 0], init_states[i, 1], marker=marker,
                markerfacecolor=face, markeredgecolor=edge,
                markeredgewidth=1.5, markersize=10, zorder=5)

    ax.set_xlim(0, 20)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$p_x$ (m)')
    ax.set_ylabel(r'$p_y$ (m)')
    ax.set_title(
        f'Region rollout test  | λ={args.lam}, v={args.v}, θ={args.theta}, T={args.T}s\n'
        f'○ = V<0 (BRT-safe), □ = V≥0 (BRT-unsafe);  '
        f'gold-fill = conservatism cushion, red-fill = soundness bug'
    )
    ax.grid(True, alpha=0.3)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([], [], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='tab:green', markersize=10, label='V<0, success ✓'),
        Line2D([], [], marker='s', color='w', markerfacecolor='gold',
               markeredgecolor='tab:green', markersize=10, label='V≥0, success ⛌ (cushion)'),
        Line2D([], [], marker='s', color='w', markerfacecolor='white',
               markeredgecolor='tab:red', markersize=10, label='V≥0, collision ✓'),
        Line2D([], [], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='tab:red', markersize=10, label='V<0, failed ⚠ SOUNDNESS BUG'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=9)

    plt.tight_layout()
    out_path = Path(args.save) if args.save else \
        PROJECT_ROOT / 'outputs' / 'visualizations' / 'best_lambda' / \
        f'grid_rollout_{args.value_tag}_lam{args.lam}_v{args.v}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
