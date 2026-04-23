"""RoverParam: RoverBaseline augmented with TWO controller-parameter virtual states.

State: (px, py, heading, λ, v)
  px, py, heading : physical state — identical to RoverBaseline
  λ ∈ [0, 1]      : normalized MPC obstacle-weight (× OBSTACLE_WEIGHT_MAX)
  v ∈ [V_MIN, V_MAX] (m/s) : vehicle cruise speed (operator-set "speed cap")

Both λ and v have zero dynamics and zero uncertainty — they are virtual states
that index the family of (controller, speed) combinations so that a single BRT
covers the whole parameter grid.

Pick the obstacle scenario via the ROVER_PARAM_SCENARIO environment variable:
    ROVER_PARAM_SCENARIO=sparse   (default — 2 obstacles)
    ROVER_PARAM_SCENARIO=medium   (3 obstacles)
    ROVER_PARAM_SCENARIO=dense    (5 obstacles)

OBSTACLE_WEIGHT_MAX is set per scenario so that λ=1.0 corresponds to "just
enough avoidance to be clean at zero uncertainty" — i.e. λ ∈ [0,1] is a
normalized parameter, comparable across scenarios.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch

from .rover_baseline import RoverBaseline
from ...utils.obstacles import Circle2D, signed_distance_to_obstacles

__all__ = ["RoverParam", "OBSTACLE_SCENARIOS"]


OBSTACLE_SCENARIOS: dict = {
    'sparse': {
        'obstacles': (
            Circle2D(center=(8.0,  1.0), radius=1.0),
            Circle2D(center=(14.0, -1.0), radius=1.0),
        ),
        'obstacle_weight_max': 40.0,
    },
    'medium': {
        'obstacles': (
            Circle2D(center=(8.0,  1.0), radius=1.0),
            Circle2D(center=(12.0, -1.5), radius=1.0),
            Circle2D(center=(16.0,  0.5), radius=0.5),
        ),
        'obstacle_weight_max': 40.0,
    },
    'dense': {
        'obstacles': (
            Circle2D(center=(5.0,  1.5), radius=0.8),
            Circle2D(center=(8.0, -1.0), radius=0.8),
            Circle2D(center=(11.0, 1.5), radius=0.8),
            Circle2D(center=(14.0,-1.5), radius=0.8),
            Circle2D(center=(17.0, 0.5), radius=0.8),
        ),
        'obstacle_weight_max': 40.0,
    },
}

_DEFAULT_SCENARIO = 'sparse'
_SELECTED_SCENARIO = os.environ.get('ROVER_PARAM_SCENARIO', _DEFAULT_SCENARIO).lower()
if _SELECTED_SCENARIO not in OBSTACLE_SCENARIOS:
    raise ValueError(
        f"Unknown ROVER_PARAM_SCENARIO={_SELECTED_SCENARIO!r}. "
        f"Choices: {sorted(OBSTACLE_SCENARIOS.keys())}"
    )
_CFG = OBSTACLE_SCENARIOS[_SELECTED_SCENARIO]


class RoverParam(RoverBaseline):
    """RoverBaseline with (λ, v) appended as 4th and 5th virtual state dimensions."""

    SCENARIO: str = _SELECTED_SCENARIO
    OBSTACLE_WEIGHT_MAX: float = _CFG['obstacle_weight_max']
    obstacles = _CFG['obstacles']

    # Speed range (operator-set cap). v=2.0 was the RoverBaseline default; we span
    # [1.0, 5.0] so the BRT spans walking pace through highway-relative speeds.
    V_MIN: float = 3.0
    V_MAX: float = 7.0

    # ── augmented state ──────────────────────────────────────────────────
    state_dim = 5
    state_limits = torch.tensor(
        [
            [0.0, -5.0, -math.pi, 0.0, V_MIN],   # lo: px, py, θ, λ, v
            [20.0,  5.0,  math.pi, 1.0, V_MAX],  # hi
        ],
        dtype=torch.float32,
    )
    state_periodic = [False, False, True, False, False]
    state_labels = (r'$p_x$ (m)', r'$p_y$ (m)', r'$\theta$ (rad)',
                    r'$\lambda$', r'$v$ (m/s)')

    # ── uncertainty presets: pad RoverBaseline presets with (λ, v) = 0 ───
    terminal_uncertainty_limits: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (-0.5, -0.5, -0.1, 0.0, 0.0),
        ( 0.5,  0.5,  0.1, 0.0, 0.0),
    )
    # uncertainty_presets: dict = {
    #     "zero":     (( 0.0,  0.0,  0.00, 0.0, 0.0), ( 0.0,  0.0,  0.00, 0.0, 0.0)),
    #     "small":    ((-0.2, -0.2, -0.04, 0.0, 0.0), ( 0.2,  0.2,  0.04, 0.0, 0.0)),
    #     "moderate": ((-0.5, -0.5, -0.10, 0.0, 0.0), ( 0.5,  0.5,  0.10, 0.0, 0.0)),
    #     "harsh":    ((-0.8, -0.8, -0.20, 0.0, 0.0), ( 0.8,  0.8,  0.20, 0.0, 0.0)),
    # }
    
    # uncertainty_presets: dict = {
    #     "zero":     (( 0.0,  0.0,  0.00, 0.0, 0.0), ( 0.0,  0.0,  0.00, 0.0, 0.0)),
    #     "small":    ((-0.2, -0.2, -0.15, 0.0, 0.0), ( 0.2,  0.2,  0.15, 0.0, 0.0)),
    #     "moderate": ((-0.5, -0.5, -0.20, 0.0, 0.0), ( 0.5,  0.5,  0.20, 0.0, 0.0)),
    #     "harsh":    ((-0.8, -0.8, -0.30, 4.0, 0.0), ( 0.8,  0.8,  0.30, 0.0, 0.0)),
    # }

    uncertainty_presets: dict = {
        "zero":     (( 0.0,  0.0,  0.00, 0.0, 0.0), ( 0.0,  0.0,  0.00, 0.0, 0.0)),
        "small":    ((-0.2, -0.2, -0.15, 0.0, 0.0), ( 0.2,  0.2,  0.15, 0.0, 0.0)),
        "moderate": ((-0.5, -0.5, -0.20, 0.0, 0.0), ( 0.5,  0.5,  0.20, 0.0, 0.0)),
        "harsh":    ((-0.8, -0.8, -0.30, 4.0, 0.0), ( 0.8,  0.8,  0.30, 0.0, 0.0)),
    }

    # ── helpers ──────────────────────────────────────────────────────────
    @classmethod
    def param_to_obstacle_weight(cls, lam: float) -> float:
        """Map normalized λ ∈ [0, 1] → obstacle_weight ∈ [0, OBSTACLE_WEIGHT_MAX]."""
        return float(lam) * cls.OBSTACLE_WEIGHT_MAX

    @classmethod
    def obstacle_weight_to_param(cls, weight: float) -> float:
        """Map obstacle_weight → normalized λ ∈ [0, 1]."""
        return float(weight) / cls.OBSTACLE_WEIGHT_MAX

    # ── constraints ──────────────────────────────────────────────────────
    def control_limits(self, state, time):
        base = torch.tensor([[-1.0], [1.0]], dtype=state.dtype, device=state.device)
        lo = torch.broadcast_to(base[0], state.shape[:-1] + (self.control_dim,))
        hi = torch.broadcast_to(base[1], state.shape[:-1] + (self.control_dim,))
        return lo, hi

    def uncertainty_limits(self, state, time):
        lims = torch.tensor(
            self.terminal_uncertainty_limits, dtype=state.dtype, device=state.device
        )
        lo = torch.broadcast_to(lims[0], state.shape[:-1] + (self.state_dim,))
        hi = torch.broadcast_to(lims[1], state.shape[:-1] + (self.state_dim,))
        return lo, hi

    # ── objective functions act on physical dims only ────────────────────
    def failure_function(self, state, time=None):
        return signed_distance_to_obstacles(
            self.obstacles, state[..., :2].reshape(-1, 2)
        ).reshape(*state.shape[:-1])

    def goal_function(self, state, time=None):
        dx = state[..., 0] - self.goal_state[0]
        dy = state[..., 1] - self.goal_state[1]
        return torch.hypot(dx, dy)

    def target_function(self, state, time=None):
        dx = state[..., 0] - self.goal_state[0]
        dy = state[..., 1] - self.goal_state[1]
        return torch.hypot(dx, dy) - self.goal_radius

    # ── dynamics: append λ̇ = 0, v̇ = 0 ──────────────────────────────────
    def dynamics(self, state, control, disturbance, time):
        x, y, heading, _lam, v = state.unbind(-1)
        (omega,) = control.unbind(-1)
        return torch.stack(
            (
                v * torch.cos(heading),
                v * torch.sin(heading),
                omega,
                torch.zeros_like(omega),  # λ̇ = 0
                torch.zeros_like(omega),  # v̇ = 0  (frozen — operator-set cruise)
            ),
            dim=-1,
        )
