"""Model Predictive Controller for the :class:`RoverBaseline` system."""

from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import List, Tuple

import casadi as ca
import do_mpc
import numpy as np
import torch

from src.core.inputs import Input
from src.impl.systems.rover_baseline import RoverBaseline
from src.utils.obstacles import Circle2D

__all__ = ["RoverBaseline_MPC"]


def _build_dubins_model(speed: float) -> do_mpc.model.Model:
    """Build the do-mpc model for Dubins dynamics."""
    model = do_mpc.model.Model('continuous')
    px = model.set_variable(var_type='_x', var_name='px')
    py = model.set_variable(var_type='_x', var_name='py')
    theta = model.set_variable(var_type='_x', var_name='theta')
    omega = model.set_variable(var_type='_u', var_name='omega')
    model.set_rhs('px', speed * ca.cos(theta))
    model.set_rhs('py', speed * ca.sin(theta))
    model.set_rhs('theta', omega)
    model.setup()
    return model


def _softplus(value: ca.MX, beta: float = 10.0) -> ca.MX:
    """Smoothed relu used to softly penalize constraint violations."""
    return ca.log1p(ca.exp(beta * value)) / beta


def _circle_signed_distance(px: ca.MX, py: ca.MX, params: Tuple[float, ...]) -> ca.MX:
    """Compute signed distance to circle obstacle."""
    cx, cy, radius = params
    eps = 1e-8  # Prevent gradient singularity at obstacle center
    return ca.sqrt((px - cx) ** 2 + (py - cy) ** 2 + eps) - radius


def _build_collision_penalty(
    px: ca.MX,
    py: ca.MX,
    obstacles: List[Tuple[str, Tuple[float, ...]]],
    obstacle_weight: float,
    obstacle_margin: float,
) -> ca.MX:
    """Build collision penalty term for MPC objective."""
    if not obstacles or obstacle_weight <= 0:
        return ca.MX.zeros(1)
    penalties = []
    for kind, params in obstacles:
        if kind == 'circle':
            distance = _circle_signed_distance(px, py, params)
        else:
            raise ValueError(f'Unknown obstacle type: {kind}')
        violation = obstacle_margin - distance
        penalties.append(_softplus(violation))
    return obstacle_weight * sum(penalties)


def _build_dubins_mpc(
    model: do_mpc.model.Model,
    dt: float,
    horizon: int,
    control_weight: float,
    control_lower: float,
    control_upper: float,
    goal: Tuple[float, float],
    obstacles: List[Tuple[str, Tuple[float, ...]]],
    obstacle_weight: float,
    obstacle_margin: float,
) -> do_mpc.controller.MPC:
    """Build and configure the MPC controller."""
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
    px = model.x['px']
    py = model.x['py']
    omega = model.u['omega']
    goal_x, goal_y = goal[0], goal[1]
    base_stage_cost = (px - goal_x) ** 2 + (py - goal_y) ** 2
    collision_penalty = _build_collision_penalty(px, py, obstacles, obstacle_weight, obstacle_margin)
    stage_cost = base_stage_cost + collision_penalty
    terminal_cost = base_stage_cost + collision_penalty
    mpc.set_objective(lterm=stage_cost, mterm=terminal_cost)
    mpc.set_rterm(omega=control_weight)
    mpc.bounds['lower', '_u', 'omega'] = control_lower
    mpc.bounds['upper', '_u', 'omega'] = control_upper
    mpc.setup()
    return mpc


def _worker_init(dt, horizon, control_weight, obstacle_weight, obstacle_margin,
                 state_dim, speed, control_lower, control_upper, goal, obstacles):
    """Initialize worker process with its own MPC instance."""
    global _worker_mpc, _worker_initialized
    model = _build_dubins_model(speed)
    _worker_mpc = _build_dubins_mpc(
        model, dt, horizon, control_weight, control_lower, control_upper,
        goal, obstacles, obstacle_weight, obstacle_margin,
    )
    _worker_initialized = False


def _worker_compute(state_row):
    """Worker function to compute control for a single state."""
    global _worker_mpc, _worker_initialized
    column_state = np.asarray(state_row, dtype=float).reshape(-1, 1)
    _worker_mpc.x0 = column_state
    if not _worker_initialized:
        _worker_mpc.set_initial_guess()
        _worker_initialized = True
    control = np.asarray(_worker_mpc.make_step(column_state)).reshape(-1)
    return control


class RoverBaseline_MPC(Input):
    """MPC controller for the RoverBaseline system (single obstacle)."""

    type = 'control'
    system_class = RoverBaseline
    dim = 1
    time_invariant = True

    _use_gpu = False
    _batch_size = 100000

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 5,
        control_weight: float = 1e-2,
        obstacle_weight: float = 20.0,
        obstacle_margin: float = 0.5,
        num_workers: int = -1,
        parallel_threshold: int = 10,
    ) -> None:
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.control_weight = float(control_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.obstacle_margin = float(obstacle_margin)

        if num_workers == -1:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = max(1, int(num_workers))
        self.parallel_threshold = max(1, int(parallel_threshold))

        self._initialised = False
        self._state_dim = RoverBaseline.state_dim
        self._pool = None

        if self.num_workers > 1:
            atexit.register(self._cleanup_pool)

    def bind(self, system: RoverBaseline) -> None:
        if not isinstance(system, RoverBaseline):
            raise TypeError(
                f"RoverBaseline_MPC requires RoverBaseline system, got {type(system).__name__}"
            )
        self._state_dim = system.state_dim
        self._speed = float(system.v)

        initial_state = system.initial_state
        time_tensor = torch.tensor(0.0, dtype=initial_state.dtype, device=initial_state.device)
        lower, upper = system.control_limits(initial_state, time_tensor)
        self._control_lower = lower.detach().cpu().numpy().astype(float).reshape(-1)
        self._control_upper = upper.detach().cpu().numpy().astype(float).reshape(-1)

        self._goal = system.goal_state.detach().cpu().numpy().astype(float)
        self._obstacles = self._extract_obstacle_descriptions(system)
        self._effective_margin = self.obstacle_margin

        # Build sequential MPC
        self.model = _build_dubins_model(self._speed)
        self.mpc = _build_dubins_mpc(
            self.model, self.dt, self.horizon, self.control_weight,
            self._control_lower, self._control_upper, self._goal,
            self._obstacles, self.obstacle_weight, self._effective_margin,
        )
        self._initialised = False

        # Build worker pool
        if self.num_workers > 1:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
            self._pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(
                    self.dt, self.horizon, self.control_weight, self.obstacle_weight,
                    self._effective_margin, self._state_dim, self._speed,
                    self._control_lower, self._control_upper, self._goal, self._obstacles,
                ),
            )

    def _extract_obstacle_descriptions(
        self, system: RoverBaseline
    ) -> List[Tuple[str, Tuple[float, ...]]]:
        """Convert obstacle objects into lightweight tuples for symbolic use."""
        specs: List[Tuple[str, Tuple[float, ...]]] = []
        for obstacle in system.obstacles:
            if isinstance(obstacle, Circle2D):
                center = obstacle.center.detach().cpu().numpy().astype(float)
                radius = float(obstacle.radius)
                specs.append(('circle', (center[0], center[1], radius)))
            else:
                raise TypeError(f'Unsupported obstacle type: {type(obstacle)!r}')
        return specs

    def _compute_control(self, state_row: np.ndarray) -> torch.Tensor:
        column_state = np.asarray(state_row, dtype=float).reshape(self._state_dim, 1)
        self.mpc.x0 = column_state
        if not self._initialised:
            self.mpc.set_initial_guess()
            self._initialised = True
        control = np.asarray(self.mpc.make_step(column_state)).reshape(-1)
        return torch.from_numpy(control).to(torch.float32)

    def input(self, state, time):  # type: ignore[override]
        """Compute the optimal control action for the provided state(s)."""
        del time
        if not hasattr(self, 'mpc'):
            raise RuntimeError('Controller must be bound to a system before use.')

        state_tensor = torch.as_tensor(state)
        if state_tensor.ndim == 0:
            raise ValueError('State input must include the state dimension.')
        if state_tensor.shape[-1] != self._state_dim:
            raise ValueError(
                f'Expected states with last dim {self._state_dim}, got {tuple(state_tensor.shape)}'
            )
        batch_shape = tuple(state_tensor.shape[:-1])
        dtype = state_tensor.dtype
        device = state_tensor.device
        flat_states = state_tensor.detach().to(dtype=torch.float64, device='cpu').reshape(-1, self._state_dim).numpy().astype(float)
        batch_size = flat_states.shape[0]

        use_parallel = (
            self.num_workers > 1
            and batch_size >= self.parallel_threshold
            and self._pool is not None
        )

        if use_parallel:
            controls = self._pool.map(_worker_compute, flat_states)
            controls_tensor = torch.from_numpy(np.array(controls)).to(torch.float32)
        else:
            controls = [self._compute_control(row) for row in flat_states]
            controls_tensor = torch.stack(controls, dim=0)

        return controls_tensor.reshape(*batch_shape, self.dim).to(dtype=dtype, device=device)

    def reset(self) -> None:
        """Reset the internal warm-start state used by the MPC solver."""
        self._initialised = False
        if hasattr(self, 'mpc') and hasattr(self.mpc, 'reset_history'):
            self.mpc.reset_history()

        # Recreate parallel pool if it exists
        if self.num_workers > 1 and hasattr(self, '_pool') and self._pool is not None:
            self._pool.close()
            self._pool.join()
            if hasattr(self, '_speed'):  # Only recreate if already bound
                self._pool = mp.Pool(
                    processes=self.num_workers,
                    initializer=_worker_init,
                    initargs=(
                        self.dt, self.horizon, self.control_weight, self.obstacle_weight,
                        self._effective_margin, self._state_dim, self._speed,
                        self._control_lower, self._control_upper, self._goal, self._obstacles,
                    ),
                )

    def _cleanup_pool(self):
        """Clean up the worker pool safely."""
        if hasattr(self, '_pool') and self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
                self._pool = None
            except Exception:
                pass

    def __del__(self):
        """Clean up the worker pool on deletion."""
        self._cleanup_pool()
