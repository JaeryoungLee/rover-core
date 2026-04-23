"""Parameterized MPC controller for the RoverParam system.

Reads (λ, v) from state[..., 3:5] and passes them as CasADi _p parameters so
the NLP is compiled only once and (λ, v) are substituted numerically at every
solve step.

  obstacle_weight = λ * RoverParam.OBSTACLE_WEIGHT_MAX
  vehicle_speed   = v   (operator-set cruise speed, m/s)
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import List, Tuple

import casadi as ca
import do_mpc
import numpy as np
import torch

from src.core.inputs import Input
from src.impl.systems.rover_param import RoverParam
from src.utils.obstacles import Circle2D

__all__ = ["RoverParam_MPC"]


# ── model / MPC builders ──────────────────────────────────────────────────────

def _build_dubins_model() -> do_mpc.model.Model:
    model = do_mpc.model.Model('continuous')
    px    = model.set_variable('_x', 'px')
    py    = model.set_variable('_x', 'py')
    theta = model.set_variable('_x', 'theta')
    omega = model.set_variable('_u', 'omega')
    # Static parameters substituted at every solve step
    lam   = model.set_variable('_p', 'lam')
    v     = model.set_variable('_p', 'v')
    model.set_rhs('px',    v * ca.cos(theta))
    model.set_rhs('py',    v * ca.sin(theta))
    model.set_rhs('theta', omega)
    model.setup()
    return model, lam, v


def _softplus(value: ca.MX, beta: float = 10.0) -> ca.MX:
    return ca.log1p(ca.exp(beta * value)) / beta


def _circle_signed_distance(px, py, params: Tuple[float, ...]) -> ca.MX:
    cx, cy, radius = params
    return ca.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-8) - radius


def _build_dubins_mpc(
    model: do_mpc.model.Model,
    lam_sym: ca.MX,
    v_sym: ca.MX,            # accepted for symmetry; not used in cost (only via dynamics)
    dt: float,
    horizon: int,
    control_weight: float,
    control_lower: float,
    control_upper: float,
    goal: Tuple[float, float],
    obstacles: List[Tuple[str, Tuple[float, ...]]],
    obstacle_weight_max: float,
    obstacle_margin: float,
) -> do_mpc.controller.MPC:
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=horizon,
        t_step=dt,
        n_robust=0,
        store_full_solution=False,
        nlpsol_opts={
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
        },
    )

    px, py = model.x['px'], model.x['py']
    omega  = model.u['omega']
    goal_x, goal_y = goal

    # Collision penalty weighted by λ * obstacle_weight_max
    collision = 0  # scalar; CasADi promotes to match variable type (SX or MX)
    for kind, params in obstacles:
        if kind == 'circle':
            dist = _circle_signed_distance(px, py, params)
        else:
            raise ValueError(f'Unknown obstacle type: {kind}')
        collision += _softplus(obstacle_margin - dist)
    collision *= lam_sym * obstacle_weight_max   # ← λ enters the cost here

    goal_cost = (px - goal_x) ** 2 + (py - goal_y) ** 2
    stage_cost     = goal_cost + collision
    mpc.set_objective(lterm=stage_cost, mterm=stage_cost)
    mpc.set_rterm(omega=control_weight)
    mpc.bounds['lower', '_u', 'omega'] = control_lower
    mpc.bounds['upper', '_u', 'omega'] = control_upper
    # Provide a default p_fun so setup() doesn't complain; updated before each solve
    p_template = mpc.get_p_template(1)
    p_template['_p', 0, 'lam'] = 0.0
    p_template['_p', 0, 'v']   = 1.0   # arbitrary nonzero default
    mpc.set_p_fun(lambda _: p_template)
    mpc.setup()
    return mpc


# ── worker helpers ────────────────────────────────────────────────────────────

_W_MPC   = None
_W_PINIT = False
_W_PTEMPL = None


def _worker_init(dt, horizon, control_weight, obstacle_weight_max, obstacle_margin,
                 control_lower, control_upper, goal, obstacles):
    global _W_MPC, _W_PINIT, _W_PTEMPL
    model, lam_sym, v_sym = _build_dubins_model()
    _W_MPC = _build_dubins_mpc(
        model, lam_sym, v_sym, dt, horizon, control_weight,
        control_lower, control_upper, goal, obstacles,
        obstacle_weight_max, obstacle_margin,
    )
    _W_PTEMPL = _W_MPC.get_p_template(1)
    _W_PINIT  = False


def _worker_compute(args):
    """Compute MPC control for one (state_row, lam_val, v_val) tuple."""
    global _W_MPC, _W_PINIT, _W_PTEMPL
    state_row, lam_val, v_val = args
    col = np.asarray(state_row, dtype=float).reshape(-1, 1)

    # Update the parameter template with current (λ, v)
    _W_PTEMPL['_p', 0, 'lam'] = float(lam_val)
    _W_PTEMPL['_p', 0, 'v']   = float(v_val)
    _W_MPC.set_p_fun(lambda _: _W_PTEMPL)

    _W_MPC.x0 = col
    if not _W_PINIT:
        _W_MPC.set_initial_guess()
        _W_PINIT = True
    return np.asarray(_W_MPC.make_step(col)).reshape(-1)


# ── main class ────────────────────────────────────────────────────────────────

class RoverParam_MPC(Input):
    """MPC controller for RoverParam.

    Reads λ = state[..., 3] at each call and passes it to the solver as a
    CasADi _p parameter.  The NLP is compiled once at bind() time.
    """

    type = 'control'
    system_class = RoverParam
    dim = 1
    time_invariant = True

    _use_gpu = True
    _batch_size = 100_000

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 5,
        control_weight: float = 1e-2,
        obstacle_margin: float = 0.5,
        num_workers: int = -1,
        parallel_threshold: int = 10,
    ) -> None:
        self.dt               = float(dt)
        self.horizon          = int(horizon)
        self.control_weight   = float(control_weight)
        self.obstacle_margin  = float(obstacle_margin)
        self.num_workers      = max(1, mp.cpu_count() - 1) if num_workers == -1 else max(1, int(num_workers))
        self.parallel_threshold = max(1, int(parallel_threshold))
        self._pool            = None
        self._initialised     = False
        if self.num_workers > 1:
            atexit.register(self._cleanup_pool)

    # ── binding ──────────────────────────────────────────────────────────

    def bind(self, system: RoverParam) -> None:
        if not isinstance(system, RoverParam):
            raise TypeError(f'RoverParam_MPC requires RoverParam, got {type(system).__name__}')

        self._physical_dim     = 3  # px, py, heading
        self._obstacle_wt_max  = float(system.OBSTACLE_WEIGHT_MAX)

        init = system.initial_state[:3]
        t    = torch.tensor(0.0)
        lo, hi = system.control_limits(init, t)
        self._ctrl_lo = float(lo[0].item())
        self._ctrl_hi = float(hi[0].item())

        self._goal      = system.goal_state[:2].cpu().numpy().astype(float)
        self._obstacles = self._extract_obstacles(system)

        # Build sequential (fallback) MPC
        model, lam_sym, v_sym = _build_dubins_model()
        self.model   = model
        self._lam_sym = lam_sym
        self._v_sym   = v_sym
        self.mpc     = _build_dubins_mpc(
            model, lam_sym, v_sym, self.dt, self.horizon, self.control_weight,
            self._ctrl_lo, self._ctrl_hi, tuple(self._goal),
            self._obstacles, self._obstacle_wt_max, self.obstacle_margin,
        )
        self._p_template = self.mpc.get_p_template(1)
        self._initialised = False

        # Build worker pool
        if self.num_workers > 1:
            self._start_pool()

    def _extract_obstacles(self, system: RoverParam) -> List[Tuple[str, Tuple[float, ...]]]:
        specs = []
        for obs in system.obstacles:
            if isinstance(obs, Circle2D):
                c = obs.center.cpu().numpy().astype(float)
                specs.append(('circle', (float(c[0]), float(c[1]), float(obs.radius))))
            else:
                raise TypeError(f'Unsupported obstacle: {type(obs)!r}')
        return specs

    def _start_pool(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
        self._pool = mp.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(
                self.dt, self.horizon, self.control_weight,
                self._obstacle_wt_max, self.obstacle_margin,
                self._ctrl_lo, self._ctrl_hi,
                tuple(self._goal), self._obstacles,
            ),
        )

    # ── single-state solve ────────────────────────────────────────────────

    def _solve_one(self, state_row: np.ndarray, lam_val: float, v_val: float) -> torch.Tensor:
        col = state_row[:self._physical_dim].reshape(-1, 1)
        self._p_template['_p', 0, 'lam'] = lam_val
        self._p_template['_p', 0, 'v']   = v_val
        self.mpc.set_p_fun(lambda _: self._p_template)
        self.mpc.x0 = col
        if not self._initialised:
            self.mpc.set_initial_guess()
            self._initialised = True
        return torch.from_numpy(np.asarray(self.mpc.make_step(col)).reshape(-1)).float()

    # ── batched input ──────────────────────────────────────────────────────

    def input(self, state, time):
        del time
        if not hasattr(self, 'mpc'):
            raise RuntimeError('RoverParam_MPC must be bound before use.')

        x = torch.as_tensor(state)
        if x.shape[-1] != 5:
            raise ValueError(f'Expected 5D state (px,py,heading,λ,v), got shape {tuple(x.shape)}')

        batch_shape = x.shape[:-1]
        dtype, device = x.dtype, x.device
        flat = x.detach().cpu().to(torch.float64).reshape(-1, 5).numpy()
        B = flat.shape[0]

        use_par = self.num_workers > 1 and B >= self.parallel_threshold and self._pool is not None

        if use_par:
            args = [(row[:3], float(row[3]), float(row[4])) for row in flat]
            results = self._pool.map(_worker_compute, args)
            ctrl = torch.from_numpy(np.array(results, dtype=np.float32))
        else:
            ctrl = torch.stack([self._solve_one(row, float(row[3]), float(row[4])) for row in flat])

        return ctrl.reshape(*batch_shape, self.dim).to(dtype=dtype, device=device)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._initialised = False
        if hasattr(self, 'mpc') and hasattr(self.mpc, 'reset_history'):
            self.mpc.reset_history()
        if self.num_workers > 1 and hasattr(self, '_ctrl_lo'):
            self._start_pool()

    def _cleanup_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
                self._pool = None
            except Exception:
                pass

    def __del__(self):
        self._cleanup_pool()
