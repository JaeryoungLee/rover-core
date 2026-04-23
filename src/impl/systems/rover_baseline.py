"""Standalone Dubins car system with a single obstacle for baseline comparisons.

RoverBaseline: Single obstacle at (16, 0) r=1, time-invariant uncertainty,
designed for fair comparison between RoVer-CoRe and baseline NNCS verification tools.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from ...utils.obstacles import Circle2D, signed_distance_to_obstacles
from .rover_dark import RoverBase

__all__ = ["RoverBaseline"]


class RoverBaseline(RoverBase):
    """Dubins car with a single obstacle and time-invariant perception uncertainty.

    State: (x, y, theta) - position and heading.
    Control: omega - angular velocity.

    Uncertainty is fixed (time-invariant): e in [-0.5, 0.5] x [-0.5, 0.5] x [-0.1, 0.1].
    """

    state_dim = 3
    state_limits = torch.tensor(
        [
            [0.0, -5.0, -math.pi],   # lower bounds: x, y, theta
            [20.0, 5.0, math.pi],     # upper bounds: x, y, theta
        ],
        dtype=torch.float32,
    )
    state_periodic = [False, False, True]  # theta is periodic
    state_labels = (r'$p_x$ (m)', r'$p_y$ (m)', r'$\theta$ (rad)')

    control_dim = 1
    control_labels = (r'$\omega$ (rad/s)',)

    disturbance_dim = 0
    disturbance_labels = ()

    initial_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    # Computational configuration
    _use_gpu = True
    _batch_size = 100000

    # System parameters
    v = 5.0  # constant speed (m/s)
    time_horizon = 5.0  # s

    # Time-invariant uncertainty bounds (fixed, not scaled by time)
    terminal_uncertainty_limits: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (-0.5, -0.5, -0.1),  # lower bounds
        (0.5, 0.5, 0.1),     # upper bounds
    )
    uncertainty_growth_rate: float = 0.0  # time-invariant

    uncertainty_presets: dict = {
        "zero":     (( 0.0,  0.0,  0.0 ), ( 0.0,  0.0,  0.0 )),
        "small":    ((-0.2, -0.2, -0.04), ( 0.2,  0.2,  0.04)),
        "moderate": ((-0.5, -0.5, -0.1 ), ( 0.5,  0.5,  0.1 )),
        "harsh":    ((-0.8, -0.8, -0.2 ), ( 0.8,  0.8,  0.2 )),
    }

    @property
    def time_invariant_uncertainty_limits(self) -> bool:
        """Always True — uncertainty is fixed for this system."""
        return True

    obstacles = (
        Circle2D(center=(16.0, 0.0), radius=1.0),
    )
    goal_state = torch.tensor([18.0, 0.0, 0.0], dtype=torch.float32)
    goal_radius: float = 0.5

    _render_title = "RoverBaseline"

    def __init__(self) -> None:
        self._render_cache: Dict[int, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def control_limits(self, state, time):
        base_limits = torch.tensor(
            [[-1.0], [1.0]],
            dtype=state.dtype,
            device=state.device,
        )
        limits = torch.broadcast_to(base_limits, state.shape[:-1] + (2, self.control_dim))
        return limits[..., 0, :], limits[..., 1, :]

    def disturbance_limits(self, state, time):
        base_limits = torch.empty(
            (2, self.disturbance_dim),
            dtype=state.dtype,
            device=state.device,
        )
        limits = torch.broadcast_to(base_limits, state.shape[:-1] + (2, self.disturbance_dim))
        return limits[..., 0, :], limits[..., 1, :]

    def uncertainty_limits(self, state, time):
        """Return fixed (time-invariant) uncertainty bounds."""
        terminal_limits = torch.tensor(
            self.terminal_uncertainty_limits,
            dtype=state.dtype,
            device=state.device,
        )
        limits = torch.broadcast_to(terminal_limits, state.shape[:-1] + (2, self.state_dim))
        return limits[..., 0, :], limits[..., 1, :]

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------
    def failure_function(self, state, time=None):
        return signed_distance_to_obstacles(
            self.obstacles, state[..., :2].view(-1, 2)
        ).view(*state.shape[:-1])

    def goal_function(self, state, time=None):
        dx = state[..., 0] - self.goal_state[0]
        dy = state[..., 1] - self.goal_state[1]
        return torch.hypot(dx, dy)

    def target_function(self, state, time=None):
        """Signed distance to goal region: negative inside, positive outside."""
        dx = state[..., 0] - self.goal_state[0]
        dy = state[..., 1] - self.goal_state[1]
        return torch.hypot(dx, dy) - self.goal_radius

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def dynamics(self, state, control, disturbance, time):
        x, y, heading = state.unbind(-1)
        (omega,) = control.unbind(-1)
        x_dot = self.v * torch.cos(heading)
        y_dot = self.v * torch.sin(heading)
        heading_dot = omega
        return torch.stack((x_dot, y_dot, heading_dot), dim=-1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(
        self,
        state,
        control,
        disturbance,
        uncertainty,
        time,
        ax,
        *,
        artists: Optional[Dict[str, object]] = None,
        history: Optional[torch.Tensor] = None,
        frame: Optional[int] = None,
    ):
        cache = self._get_render_cache(ax)

        if artists is None:
            artists = self._create_base_artists(ax, cache, include_point=True)

        self._update_trajectory_lines(artists, history)
        if state is not None:
            state_tensor = torch.as_tensor(state).detach().cpu()
            x, y = float(state_tensor[0]), float(state_tensor[1])
            if 'point' in artists:
                artists['point'].set_data([x], [y])
            self._update_heading_arrow(artists, state)

        self._update_estimated_heading(artists, history)

        return artists
