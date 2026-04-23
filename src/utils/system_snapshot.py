"""Capture a JSON-serializable snapshot of a System instance.

Used by build_*.py and simulate.py to record the exact system definition
(obstacles, goal, parameters, uncertainty preset, etc.) alongside the cache
or simulation output, so experiments are reproducible after the fact.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def _tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


def snapshot_system(system: Any) -> Dict[str, Any]:
    """Return a JSON-serializable dict capturing the system's salient definition.

    Fields are best-effort — only attributes the system actually has are included.
    """
    out: Dict[str, Any] = {'class': type(system).__name__}

    # Obstacles (Circle2D / Rectangle / etc.)
    obstacles = getattr(system, 'obstacles', None)
    if obstacles is not None:
        obs_list: List[Dict[str, Any]] = []
        for ob in obstacles:
            entry: Dict[str, Any] = {'type': type(ob).__name__}
            for attr in ('center', 'radius', 'corners', 'half_extents'):
                if hasattr(ob, attr):
                    entry[attr] = _tensor_to_list(getattr(ob, attr))
            obs_list.append(entry)
        out['obstacles'] = obs_list

    # Goal
    if hasattr(system, 'goal_state'):
        out['goal_state'] = _tensor_to_list(system.goal_state)
    if hasattr(system, 'goal_radius'):
        out['goal_radius'] = float(system.goal_radius)

    # State / time geometry
    if hasattr(system, 'state_limits'):
        out['state_limits'] = _tensor_to_list(system.state_limits)
    if hasattr(system, 'state_dim'):
        out['state_dim'] = int(system.state_dim)
    if hasattr(system, 'state_periodic'):
        out['state_periodic'] = list(system.state_periodic)
    if hasattr(system, 'state_labels'):
        out['state_labels'] = list(system.state_labels)
    if hasattr(system, 'time_horizon'):
        out['time_horizon'] = float(system.time_horizon)

    # System-specific scalars
    for attr in ('v', 'OBSTACLE_WEIGHT_MAX'):
        if hasattr(system, attr):
            try:
                out[attr] = float(getattr(system, attr))
            except (TypeError, ValueError):
                pass

    # Uncertainty bounds + presets (if defined)
    if hasattr(system, 'terminal_uncertainty_limits'):
        out['terminal_uncertainty_limits'] = _tensor_to_list(system.terminal_uncertainty_limits)
    if hasattr(system, 'uncertainty_growth_rate'):
        try:
            out['uncertainty_growth_rate'] = float(system.uncertainty_growth_rate)
        except (TypeError, ValueError):
            pass
    if hasattr(system, 'uncertainty_presets'):
        try:
            presets = {k: _tensor_to_list(v) for k, v in dict(system.uncertainty_presets).items()}
            out['uncertainty_presets'] = presets
        except Exception:
            pass

    return out
