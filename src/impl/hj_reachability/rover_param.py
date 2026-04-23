from typing import Tuple, Any

import jax.numpy as jnp
import torch

from ...core.hj_reachability import ChannelConfig, ChannelMode, HJReachabilityDynamics
from ...core.inputs import Input
from ...core.sets import Set
from ..inputs.jax_grid_input import JaxGridInput
from ..sets.jax_grid_set import JaxGridSet
from ..systems.rover_param import RoverParam as RoverParamSystem
from .base import HJSolverDynamics

__all__ = ["RoverParam", "RoverParamNominal", "RuntimeRoverParam"]


class RuntimeRoverParam(HJReachabilityDynamics):
    """RoverParam runtime dynamics using PyTorch GridSet for uncertainty optimization."""

    def optimal_uncertainty_from_grad(self, state, time, grad):
        if self.control.given_kind != "set" or self.control.given_set is None:
            raise RuntimeError("Control set must be bound before extracting optimal uncertainty.")
        state_t = torch.as_tensor(state).to(torch.float32)
        grad_t = torch.as_tensor(grad, dtype=state_t.dtype, device=state_t.device)
        flat_state = state_t.reshape(-1, state_t.shape[-1])
        direction = -grad_t.reshape(-1, grad_t.shape[-1])[:, 2:3]  # dV/dtheta
        control_set = self.control.given_set.to(state_t.device)
        _, xhat = control_set.argmax_support_with_state_est(direction, flat_state, float(time))
        if xhat is None:
            return torch.zeros_like(state_t)
        return (xhat.reshape_as(flat_state) - flat_state).reshape_as(state_t)


class _RoverParamBase(HJSolverDynamics):
    """Shared base for RoverParam HJ dynamics variants.

    Unicycle dynamics augmented with TWO frozen virtual states:
      - λ_obs ∈ [0,1]                       controller obstacle weight   (λ̇ = 0)
      - λ_unc ∈ [LAM_UNC_MIN, LAM_UNC_MAX]  perception-uncertainty scale (λ̇ = 0)
    Both contribute zero to the Hamiltonian. v is constant from the system.
    """

    def __init__(self) -> None:
        self.system = RoverParamSystem()
        self._v = float(self.system.v)

    def __call__(self, state, control, disturbance, time):
        raise NotImplementedError("Use hamiltonian() method instead")

    def optimal_control_and_disturbance(self, state, time, grad_value):
        raise NotImplementedError("Use hamiltonian() method instead")

    def _get_control_bounds(self, state: jnp.ndarray, time: float) -> Tuple[float, float]:
        raise NotImplementedError("Subclass must implement _get_control_bounds")

    def partial_max_magnitudes(
        self,
        state: jnp.ndarray,
        time: float,
        value: jnp.ndarray,
        grad_value_box: Any,
    ) -> jnp.ndarray:
        """Max magnitudes of Hamiltonian partials for the 5D augmented state.

        H = dVdx·v·cos(θ) + dVdy·v·sin(θ) + dVdθ·ω + dVdλ_obs·0 + dVdλ_unc·0
            ∂H/∂(dVdx)     = v·cos(θ)
            ∂H/∂(dVdy)     = v·sin(θ)
            ∂H/∂(dVdθ)     = ω  (bounded by control limits)
            ∂H/∂(dVdλ_obs) = 0  (λ_obs frozen)
            ∂H/∂(dVdλ_unc) = 0  (λ_unc frozen)
        """
        theta = state[2]
        v     = self._v
        partial_x_mag = jnp.abs(v * jnp.cos(theta))
        partial_y_mag = jnp.abs(v * jnp.sin(theta))
        omega_min, omega_max = self._get_control_bounds(state, time)
        partial_theta_mag = jnp.max(jnp.abs(jnp.array([omega_min, omega_max])))
        return jnp.array([partial_x_mag, partial_y_mag, partial_theta_mag, 0.0, 0.0])


class RoverParam(_RoverParamBase):
    """HJ reachability dynamics for RoverParam (4D: px, py, θ, λ).

    reach_avoid controls the optimization direction:
      False (default): adversary minimizes H → obstacle BRT (V < 0 = unsafe)
      True:            adversary maximizes H → reach-avoid  (V < 0 = safe)
    """

    @classmethod
    def runtime_class(cls) -> type:
        return RuntimeRoverParam

    def __init__(self, reach_avoid: bool = False) -> None:
        super().__init__()
        self._adversary_sign = +1 if reach_avoid else -1
        self.control = ChannelConfig(mode=ChannelMode.GIVEN)
        self.disturbance = ChannelConfig(mode=ChannelMode.ZERO)
        self.uncertainty = ChannelConfig(mode=ChannelMode.OPTIMIZE)
        self.control.given_kind = 'set'

    def _get_control_bounds(self, state: jnp.ndarray, time: float) -> Tuple[float, float]:
        lower_ctrl, upper_ctrl = self.jax_grid_set.as_box(state, time)
        return lower_ctrl[0], upper_ctrl[0]

    def hamiltonian(self, state: jnp.ndarray, time: float, value: jnp.ndarray, grad_value: jnp.ndarray) -> jnp.ndarray:
        """H = dVdx·v·cos(θ) + dVdy·v·sin(θ) + dVdθ·ω*  (dVdλ·0, dVdv·0 omitted)."""
        dVdx, dVdy, dVdtheta = grad_value[0], grad_value[1], grad_value[2]
        theta = state[2]
        v     = self._v
        optimal_omega = self.jax_grid_set.argmax_support(
            jnp.array([self._adversary_sign * dVdtheta]), state, time)[0]
        return dVdx * v * jnp.cos(theta) + dVdy * v * jnp.sin(theta) + dVdtheta * optimal_omega

    def bind_control_set(self, given_set: Set) -> None:
        self.control.given_set = given_set
        self.jax_grid_set = JaxGridSet(given_set)

    def optimal_uncertainty_from_grad(self, state, time, grad_value):
        if not hasattr(self, "jax_grid_set"):
            raise RuntimeError("Control set must be bound before extracting optimal uncertainty.")

        import numpy as np

        def _to_numpy(arr):
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

        state_np = _to_numpy(state)
        grad_np = _to_numpy(grad_value)

        original_shape = state_np.shape
        state_flat = state_np.reshape(-1, state_np.shape[-1])
        grad_flat = grad_np.reshape(-1, grad_np.shape[-1])

        direction = self._adversary_sign * grad_flat[:, 2:3]  # dV/dtheta drives control

        best_u, best_xhat, has_state = self.jax_grid_set.argmax_support_with_state_est(
            jnp.asarray(direction), jnp.asarray(state_flat), float(time),
        )

        best_xhat = np.asarray(best_xhat)
        has_state = np.asarray(has_state)

        uncertainty = np.zeros_like(state_flat)
        if best_xhat.ndim == 1:
            has = bool(has_state) if np.ndim(has_state) == 0 else bool(has_state[0])
            if has:
                uncertainty = (best_xhat - state_flat).reshape(uncertainty.shape)
        else:
            mask = has_state.astype(bool)
            if mask.ndim == 0:
                mask = np.array([bool(mask)])
            if mask.any():
                uncertainty[mask] = (best_xhat - state_flat)[mask]

        # Zero out λ and v components — they have no uncertainty by design
        uncertainty[:, 3] = 0.0
        uncertainty[:, 4] = 0.0

        return uncertainty.reshape(original_shape)


class RoverParamNominal(_RoverParamBase):
    """Nominal HJ dynamics for RoverParam with deterministic control (no uncertainty)."""

    def __init__(self, system=None, reach_avoid: bool = False) -> None:
        super().__init__()
        self._adversary_sign = +1 if reach_avoid else -1
        self.control = ChannelConfig(mode=ChannelMode.GIVEN)
        self.disturbance = ChannelConfig(mode=ChannelMode.ZERO)
        self.uncertainty = ChannelConfig(mode=ChannelMode.ZERO)
        self.control.given_kind = 'input'
        self._jax_grid_input = None

    def _get_control_bounds(self, state: jnp.ndarray, time: float) -> Tuple[float, float]:
        return -1.0, 1.0

    def bind_control_input(self, given_input: Input) -> None:
        self.control.given_input = given_input
        self._jax_grid_input = JaxGridInput(given_input)

    def hamiltonian(self, state: jnp.ndarray, time: float, value: jnp.ndarray, grad_value: jnp.ndarray) -> jnp.ndarray:
        dVdx, dVdy, dVdtheta = grad_value[0], grad_value[1], grad_value[2]
        theta = state[2]
        v     = self._v
        omega = self._jax_grid_input.value(state, time)[0]
        return dVdx * v * jnp.cos(theta) + dVdy * v * jnp.sin(theta) + dVdtheta * omega
