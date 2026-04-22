#!/usr/bin/env python3
"""
Visualize GridValue caches by tag.

This script generates visualizations of the HJ reachability grid value (value function)
using 2D slices and optional presets defined in config/visualizations.yaml.

Usage:
  python scripts/grid_value/visualize_grid_value.py \
      --tag {TAG} \
      [--preset {PRESET}] \
      [--save-dir {SAVE_DIR}] \
      [--interpolate]

If --preset is not provided, a simple default visualization will be produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.systems import System
from src.impl.values.grid_value import GridValue
from src.utils.cache_loaders import (
    get_grid_value_metadata,
    instantiate_system_by_name,
    load_grid_value_by_tag,
)
from src.utils.config import load_visualization_presets
from src.utils.grids import nearest_axis_indices, nearest_time_index
from src.utils.interactive_viz import InteractiveVisualizer, SliderSpec, create_time_slider


# ── slicing helpers (work for any state_dim) ───────────────────────────────────

def _axis_label(system: Optional[System], dim: int) -> str:
    if system is not None and hasattr(system, 'state_labels') and dim < len(system.state_labels):
        return system.state_labels[dim]
    return f'dim_{dim}'


def _resolve_fixed_indices(vf: GridValue, dims: List[int],
                            fixed_dims: Dict[int, float]) -> Dict[int, int]:
    """Snap fixed values to nearest grid indices; default missing dims to middle."""
    out: Dict[int, int] = {}
    for d in range(vf.state_dim):
        if d in dims:
            continue
        if d in fixed_dims and fixed_dims[d] is not None:
            axis_t = vf._axes[d]
            idx = int(nearest_axis_indices(
                axis_t,
                torch.tensor([float(fixed_dims[d])], dtype=axis_t.dtype, device=axis_t.device),
            )[0].item())
        else:
            idx = vf.grid_shape[d] // 2
        out[d] = idx
    return out


def _extract_2d_slice(values_at_time, vf: GridValue, dims: List[int],
                      fixed_indices: Dict[int, int]) -> np.ndarray:
    """Index N-D values down to a 2-D array with axes in the order given by `dims`."""
    sl = [fixed_indices[d] if d in fixed_indices else slice(None) for d in range(vf.state_dim)]
    arr = values_at_time[tuple(sl)]
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    # Numpy/torch indexing preserves original axis order; transpose if user asked otherwise.
    sorted_dims = sorted(dims)
    if list(dims) != sorted_dims:
        perm = [sorted_dims.index(d) for d in dims]
        arr = arr.transpose(perm)
    return arr


def _format_fixed_str(system: Optional[System], fixed_indices: Dict[int, int],
                      vf: GridValue) -> str:
    parts = []
    for d, i in sorted(fixed_indices.items()):
        v = float(vf._axes[d][i].item())
        parts.append(f"{_axis_label(system, d)}={v:.2f}")
    return ', '.join(parts)


def visualize_value_slice_2d(
    vf: GridValue,
    system: System,
    time_val: float,
    slice_dim: int = 2,
    slice_value: Optional[float] = None,
    ax=None,
    fig=None
):
    """
    Visualize a single 2D slice of the grid value at a specific time.
    
    Args:
        vf: GridValue instance
        system: System instance (for obstacles)
        time_val: Time value to visualize
        slice_dim: Dimension to slice (for 3D+ state spaces)
        slice_value: Value at which to slice (None = middle)
        ax: Matplotlib axis to plot into (for interactive mode)
        fig: Matplotlib figure (for interactive mode)
    
    Returns:
        fig: Matplotlib figure
    """
    
    if vf.state_dim < 2:
        print("Cannot create 2D slices for 1D state space")
        return None
    
    # Find nearest time index
    # NOTE: nearest_time_index uses searchsorted which assumes ascending order.
    # For HJ reachability (backward in time), times are typically descending.
    # Use robust argmin approach instead.
    time_diffs = torch.abs(vf._times - float(time_val))
    time_idx = int(torch.argmin(time_diffs).item())
    actual_time = float(vf._times[time_idx].item())
    
    # Determine slice index for 3D state spaces
    if vf.state_dim > 2:
        if slice_value is None:
            slice_idx = vf.grid_shape[slice_dim] // 2
        else:
            coord_t = vf._axes[slice_dim]
            slice_idx = int(nearest_axis_indices(coord_t, torch.tensor([float(slice_value)], dtype=coord_t.dtype, device=coord_t.device))[0].item())
        slice_val = float(vf._axes[slice_dim][slice_idx].item())
    
    # Setup figure and axis
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        else:
            ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.figure
        ax.clear()
    
    # Get grid value at this time
    value_slice = vf._values[..., time_idx]
    
    # Extract 2D slice
    if vf.state_dim == 2:
        value_2d = value_slice
        X, Y = np.meshgrid(
            vf._axes[0].detach().cpu().numpy(),
            vf._axes[1].detach().cpu().numpy(),
            indexing='ij'
        )
        xlabel, ylabel = 'x', 'y'
    elif vf.state_dim == 3:
        if slice_dim == 2:
            value_2d = value_slice[:, :, slice_idx]
            X, Y = np.meshgrid(
                vf._axes[0].detach().cpu().numpy(),
                vf._axes[1].detach().cpu().numpy(),
                indexing='ij'
            )
            xlabel, ylabel = 'x', 'y'
        elif slice_dim == 1:
            value_2d = value_slice[:, slice_idx, :]
            X, Y = np.meshgrid(
                vf._axes[0].detach().cpu().numpy(),
                vf._axes[2].detach().cpu().numpy(),
                indexing='ij'
            )
            xlabel, ylabel = 'x', 'θ'
        else:  # slice_dim == 0
            value_2d = value_slice[slice_idx, :, :]
            X, Y = np.meshgrid(
                vf._axes[1].detach().cpu().numpy(),
                vf._axes[2].detach().cpu().numpy(),
                indexing='ij'
            )
            xlabel, ylabel = 'y', 'θ'
    else:
        print(f"Visualization not supported for {vf.state_dim}D state spaces")
        return None
    
    # Convert to numpy
    if isinstance(value_2d, torch.Tensor):
        value_2d_np = value_2d.detach().cpu().numpy()
    else:
        value_2d_np = np.asarray(value_2d)
    
    # Determine color normalization
    from matplotlib.colors import TwoSlopeNorm
    vabs = max(abs(float(np.min(value_2d_np))), abs(float(np.max(value_2d_np))))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    
    # Plot filled contours
    levels = np.linspace(-vabs, vabs, 21)
    cf = ax.contourf(X, Y, value_2d_np, levels=levels, cmap='RdYlBu', norm=norm)
    
    # Zero level set (boundary of reachable set)
    ax.contour(X, Y, value_2d_np, levels=[0.0], colors='black', linewidths=2, linestyles='-')
    
    # Add obstacles (if 2D spatial)
    if vf.state_dim >= 2 and slice_dim >= 2:
        from src.utils.obstacles import draw_obstacles_2d
        draw_obstacles_2d(ax, system, zorder=10)
    
    # Handle colorbar
    if ax is not None:
        # Interactive mode: check for existing colorbar
        existing_cbar_ax = None
        for cbar_ax in fig.get_axes():
            if cbar_ax.get_label() == '<colorbar>' and cbar_ax != ax:
                existing_cbar_ax = cbar_ax
                break
        
        if existing_cbar_ax is not None:
            existing_cbar_ax.clear()
            sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
            sm.set_array([])
            plt.colorbar(sm, cax=existing_cbar_ax, label='Value')
            existing_cbar_ax.set_label('<colorbar>')
        else:
            sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Value')
            if fig is not None:
                cbar.ax.set_label('<colorbar>')
    
    # Format
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(X.min()), float(X.max()))
    ax.set_ylim(float(Y.min()), float(Y.max()))
    # Only set title in static mode
    if ax is None:
        if vf.state_dim > 2:
            dim_names = ['x', 'y', 'θ']
            slice_name = dim_names[slice_dim] if slice_dim < len(dim_names) else f'dim{slice_dim}'
            ax.set_title(f'Backward Reachable Tube at t={actual_time:.2f}s ({slice_name}={slice_val:.2f})')
        else:
            ax.set_title(f'Backward Reachable Tube at t={actual_time:.2f}s')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if ax is None:
        plt.tight_layout()
    
    return fig


def visualize_2d_slices(
    vf: GridValue,
    system: System,
    time_indices: List[int],
    save_dir: Path,
    dims: Optional[List[int]] = None,
    fixed_dims: Optional[Dict[int, float]] = None,
    title_prefix: Optional[str] = None,
    filename: str = 'value_function_slices.png',
    tag: Optional[str] = None,
):
    """
    Visualize 2D slices of the value function at the requested times.

    Args:
        vf: GridValue instance
        system: System instance (for obstacles / labels)
        time_indices: Time indices to visualize
        output_dir: Output directory for plots
        dims: The two state-dim indices to plot on (x, y) of each subplot.
              Defaults to [0, 1].
        fixed_dims: {dim_idx: value} for the remaining state dims. Missing dims
                    default to the middle of their axis.
        title_prefix: Optional string prepended to the suptitle.
    """

    print(f"\nGenerating 2D slice visualizations...")
    output_dir =  save_dir/'value_function_slices'
    output_dir.mkdir(parents=True, exist_ok=True)

    if vf.state_dim < 2:
        print("Cannot create 2D slices for 1D state space")
        return

    if dims is None:
        dims = [0, 1]
    if fixed_dims is None:
        fixed_dims = {}
    if len(dims) != 2:
        print(f"`dims` must be exactly 2 entries, got {dims}")
        return

    fixed_indices = _resolve_fixed_indices(vf, dims, fixed_dims)
    fixed_str = _format_fixed_str(system, fixed_indices, vf)
    if fixed_indices:
        print(f"  Plot dims={dims} | fixed: {fixed_str}")
    
    # Create figure
    n_times = len(time_indices)
    fig, axes = plt.subplots(1, n_times, figsize=(5*n_times, 5))
    
    if n_times == 1:
        axes = [axes]
    
    # Determine symmetric color normalization centered at 0 across all requested time slices
    from matplotlib.colors import TwoSlopeNorm
    vmin_all, vmax_all = None, None
    for time_idx in time_indices:
        value_at_t = vf._values[..., time_idx]
        value_2d_arr = _extract_2d_slice(value_at_t, vf, dims, fixed_indices)
        vmin_cur = float(np.min(value_2d_arr))
        vmax_cur = float(np.max(value_2d_arr))
        vmin_all = vmin_cur if vmin_all is None else min(vmin_all, vmin_cur)
        vmax_all = vmax_cur if vmax_all is None else max(vmax_all, vmax_cur)
    vabs = max(abs(vmin_all or 0.0), abs(vmax_all or 0.0), 1e-9)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    # Pre-build meshgrid for the chosen dims
    X, Y = np.meshgrid(
        vf._axes[dims[0]].detach().cpu().numpy(),
        vf._axes[dims[1]].detach().cpu().numpy(),
        indexing='ij',
    )
    xlabel = _axis_label(system, dims[0])
    ylabel = _axis_label(system, dims[1])

    spatial_xy = (sorted(dims) == [0, 1])

    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        value_at_t = vf._values[..., time_idx]
        value_2d_np = _extract_2d_slice(value_at_t, vf, dims, fixed_indices)

        # Plot filled contours with colorbar centered at 0
        levels = np.linspace(-vabs, vabs, 21)
        ax.contourf(X, Y, value_2d_np, levels=levels, cmap='RdYlBu', norm=norm)

        # Zero level set (boundary of reachable set)
        ax.contour(X, Y, value_2d_np, levels=[0.0], colors='black', linewidths=2, linestyles='-')

        # Obstacles + goal only when both spatial dims are visible
        if spatial_xy and system is not None:
            from src.utils.obstacles import draw_obstacles_2d, draw_goal_2d
            draw_obstacles_2d(ax, system, zorder=10)
            draw_goal_2d(ax, system, zorder=10)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(float(X.min()), float(X.max()))
        ax.set_ylim(float(Y.min()), float(Y.max()))
        ax.set_title(f't = {float(vf._times[time_idx].item()):.2f}s')
        if spatial_xy:
            ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Single shared colorbar for consistency across subplots
    sm = plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu')
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='Value', shrink=0.9)

    # Overall title (prefix with tag so uncertainty/system are visible at a glance)
    head = title_prefix or 'Backward Reachable Tube'
    if tag:
        head = f'[{tag}]\n{head}'
    if fixed_indices:
        plt.suptitle(f'{head}  ({fixed_str})', fontsize=14)
    else:
        plt.suptitle(head, fontsize=14)
    
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ {output_file.name}")
    
    plt.close()


def visualize_param_sweep(
    vf: GridValue,
    system: Optional[System],
    output_dir: Path,
    dims: List[int],
    sweep_dim: int,
    sweep_values: List[float],
    fixed_dims: Optional[Dict[int, float]] = None,
    time: float = 0.0,
    title_prefix: Optional[str] = None,
    filename: str = 'param_sweep.png',
    tag: Optional[str] = None,
):
    """One figure, N subplots — same 2D slice but with `sweep_dim` taking each value.

    Useful for showing e.g. how the initial BRT changes with λ.
    """

    print(f"\nGenerating parameter sweep visualization (dim={sweep_dim})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if fixed_dims is None:
        fixed_dims = {}
    if len(dims) != 2:
        print(f"`dims` must be exactly 2 entries, got {dims}")
        return
    if sweep_dim in dims:
        print(f"sweep_dim {sweep_dim} cannot be one of the plotted dims {dims}")
        return

    # Snap time to grid
    t_idx = int(nearest_time_index(vf._times, float(time))[0].item())
    t_val = float(vf._times[t_idx].item())
    value_at_t = vf._values[..., t_idx]

    # Pre-compute each subplot's 2D slice and a shared vabs
    slices_2d = []
    fixed_per_sub = []
    vabs = 1e-9
    for sv in sweep_values:
        fd = {**fixed_dims, sweep_dim: float(sv)}
        fixed_indices = _resolve_fixed_indices(vf, dims, fd)
        arr = _extract_2d_slice(value_at_t, vf, dims, fixed_indices)
        slices_2d.append(arr)
        fixed_per_sub.append(fixed_indices)
        vabs = max(vabs, float(np.max(np.abs(arr))))

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    X, Y = np.meshgrid(
        vf._axes[dims[0]].detach().cpu().numpy(),
        vf._axes[dims[1]].detach().cpu().numpy(),
        indexing='ij',
    )
    xlabel = _axis_label(system, dims[0])
    ylabel = _axis_label(system, dims[1])
    spatial_xy = (sorted(dims) == [0, 1])
    sweep_label = _axis_label(system, sweep_dim)

    # Color each zero level set by its sweep value
    cmap = plt.cm.viridis
    sw_min, sw_max = float(min(sweep_values)), float(max(sweep_values))
    sw_range = sw_max - sw_min if sw_max > sw_min else 1.0
    proxies = []
    proxy_labels = []
    for sv, arr, fi in zip(sweep_values, slices_2d, fixed_per_sub):
        color = cmap((float(sv) - sw_min) / sw_range)
        ax.contour(X, Y, arr, levels=[0.0], colors=[color], linewidths=2, linestyles='-')
        snapped = float(vf._axes[sweep_dim][fi[sweep_dim]].item())
        proxies.append(plt.Line2D([0], [0], color=color, linewidth=2))
        proxy_labels.append(f'{sweep_label} = {snapped:.2f}')

    if spatial_xy and system is not None:
        from src.utils.obstacles import draw_obstacles_2d, draw_goal_2d
        draw_obstacles_2d(ax, system, zorder=10)
        draw_goal_2d(ax, system, zorder=10)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(X.min()), float(X.max()))
    ax.set_ylim(float(Y.min()), float(Y.max()))
    if spatial_xy:
        ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(proxies, proxy_labels, loc='best', fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=sw_min, vmax=sw_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=sweep_label, shrink=0.85)

    head = title_prefix or f'BRT zero level vs {sweep_label} (t = {t_val:.2f}s)'
    if tag:
        head = f'[{tag}]\n{head}'
    ax.set_title(head)

    output_file = output_dir / filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ {output_file.name}")
    plt.close()


def visualize_reachable_set_evolution(
    vf: GridValue,
    save_dir: Path,
    system: Optional[System] = None,
    dims: Optional[List[int]] = None,
    fixed_dims: Optional[Dict[int, float]] = None,
    num_frames: int = 10,
    title_prefix: Optional[str] = None,
    filename: str = 'reachable_set_evolution.png',
    tag: Optional[str] = None,
):
    """Plot zero level sets at multiple times on a single 2D slice."""

    print(f"\nGenerating reachable set evolution visualization...")
    output_dir = save_dir/'reachable_set_evolution'
    output_dir.mkdir(parents=True, exist_ok=True)

    if vf.state_dim < 2:
        print("Cannot visualize evolution for 1D state space")
        return

    if dims is None:
        dims = [0, 1]
    if fixed_dims is None:
        fixed_dims = {}
    if len(dims) != 2:
        print(f"`dims` must be exactly 2 entries, got {dims}")
        return

    fixed_indices = _resolve_fixed_indices(vf, dims, fixed_dims)
    fixed_str = _format_fixed_str(system, fixed_indices, vf)

    time_indices = np.linspace(0, int(vf._times.shape[0]) - 1, num_frames, dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    X, Y = np.meshgrid(
        vf._axes[dims[0]].detach().cpu().numpy(),
        vf._axes[dims[1]].detach().cpu().numpy(),
        indexing='ij',
    )
    xlabel = _axis_label(system, dims[0])
    ylabel = _axis_label(system, dims[1])
    spatial_xy = (sorted(dims) == [0, 1])

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, num_frames - 1)) for i in range(num_frames)]

    for i, time_idx in enumerate(time_indices):
        value_at_t = vf._values[..., time_idx]
        value_2d_np = _extract_2d_slice(value_at_t, vf, dims, fixed_indices)

        cs = ax.contour(X, Y, value_2d_np, levels=[0.0],
                        colors=[colors[i]], linewidths=2, linestyles='-')
        if i % max(1, num_frames // 5) == 0:
            ax.clabel(cs, fmt=f't={float(vf._times[time_idx].item()):.2f}s',
                      inline=True, fontsize=8)

    if spatial_xy and system is not None:
        from src.utils.obstacles import draw_obstacles_2d, draw_goal_2d
        draw_obstacles_2d(ax, system, zorder=10)
        draw_goal_2d(ax, system, zorder=10)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    head = title_prefix or 'Reachable Set Evolution'
    if tag:
        head = f'[{tag}]\n{head}'
    ax.set_title(f'{head}  ({fixed_str})' if fixed_str else head)
    if spatial_xy:
        ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=Normalize(vmin=float(vf._times[0].item()), vmax=float(vf._times[-1].item())),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time (s)')

    output_file = output_dir / filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ {output_file.name}")
    plt.close()
    
    plt.close()


def run_interactive(vf, system, args, tag=None):
    """
    Run interactive visualization mode with sliders.
    
    Creates an interactive window where users can adjust:
    - Time (to scrub through the backward reachable tube evolution)
    - Slice dimension value (for 3D+ state spaces)
    
    Args:
        vf: GridValue instance
        system: System instance
        args: Command line arguments
        tag: GridValue cache tag
    """
    
    # Build sliders
    sliders = []
    
    # Time slider - always available for grid values
    time_points = vf._times.cpu().numpy()
    sliders.append(create_time_slider(time_points, description='Time (s)'))
    
    # Slice value slider for 3D+ state spaces
    slice_dim = args.slice_dim if hasattr(args, 'slice_dim') else 2
    if vf.state_dim > 2:
        slice_axis = vf._axes[slice_dim].cpu().numpy()
        dim_names = ['x', 'y', 'θ']
        slice_label = dim_names[slice_dim] if slice_dim < len(dim_names) else f'dim_{slice_dim}'
        
        initial_slice_val = args.slice_value if hasattr(args, 'slice_value') and args.slice_value is not None else slice_axis[len(slice_axis)//2]
        
        sliders.append(SliderSpec(
            name='slice_value',
            min_val=float(slice_axis.min()),
            max_val=float(slice_axis.max()),
            initial_val=float(initial_slice_val),
            description=f'{slice_label} slice'
        ))
    
    # Update function
    def update_visualization(*slider_values, ax=None):
        time_val = slider_values[0]
        if vf.state_dim > 2:
            slice_value = slider_values[1]
        else:
            slice_value = None
        
        # Call visualization with provided axis
        visualize_value_slice_2d(vf, system, time_val, slice_dim, slice_value, ax=ax)
    
    # Create title
    if tag:
        title = f"{system.__class__.__name__} - {tag} - Reachable Tube (Interactive)"
    else:
        title = f"{system.__class__.__name__} - Reachable Tube (Interactive)"
    
    viz = InteractiveVisualizer(sliders, update_visualization, title=title, direct_plot=True)
    
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print('='*60)
    print(f"State space: {vf.state_dim}D")
    print(f"Time range: [{time_points[0]:.2f}, {time_points[-1]:.2f}] s")
    if vf.state_dim > 2:
        print(f"Slice dimension: {slice_dim}")
    print("\nAdjust sliders to explore the backward reachable tube.")
    print("The black contour shows the boundary of the reachable set.")
    print("Close the window when done.\n")
    
    viz.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HJ reachability grid value (value function)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--tag', type=str, required=True,
                       help='GridValue cache tag')
    
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Output directory (default: outputs/visualizations/grid_values/{tag}/{preset})')
    # Backward-compat alias
    parser.add_argument('--output', type=str, default=None,
                       help=argparse.SUPPRESS)

    parser.add_argument('--preset', type=str, default=None,
                       help='Visualization preset name from config/visualizations.yaml')

    parser.add_argument('--interpolate', action='store_true',
                       help='Use nearest-neighbor for off-grid times/values')
    
    parser.add_argument('--time-indices', type=int, nargs='+', default=None,
                       help='Time indices to visualize (default: 0, middle, -1)')
    
    parser.add_argument('--slice-dim', type=int, default=2,
                       help='[legacy] Dimension to slice for 3D state spaces (default: 2)')
    parser.add_argument('--slice-value', type=float, default=None,
                       help='[legacy] Value at which to slice (default: middle)')
    parser.add_argument('--dims', type=int, nargs=2, default=None, metavar=('D1', 'D2'),
                       help='The two state dims to plot (x, y) of the figure. Default: [0, 1]')
    parser.add_argument('--fixed', type=str, nargs='+', default=None, metavar='DIM=VALUE',
                       help='Fix one or more non-plotted dims, e.g. --fixed 2=0.0 3=0.5. '
                            'Missing dims default to middle of their axis.')
    
    parser.add_argument('--evolution-frames', type=int, default=10,
                       help='Number of frames for evolution plot (default: 10)')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive visualization with sliders')
    
    args = parser.parse_args()
    
    # Set matplotlib backend based on mode
    if args.interactive:
        # Interactive mode needs TkAgg or similar
        try:
            matplotlib.use('TkAgg')
        except:
            try:
                matplotlib.use('Qt5Agg')
            except:
                print("Warning: Could not set interactive backend")
    else:
        # Static mode uses Agg for saving files
        matplotlib.use('Agg')
    
    # Load value function
    print("=" * 60)
    print("GridValue Visualization")
    print("=" * 60)
    
    try:
        vf = load_grid_value_by_tag(args.tag, interpolate=args.interpolate)
        if args.interpolate:
            print(f"Interpolation enabled: GridValue can evaluate at arbitrary resolutions")
        else:
            print(f"Interpolation disabled: GridValue will use cached grid points directly")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return

    # Load system for obstacle rendering using cache metadata
    try:
        meta = get_grid_value_metadata(args.tag)
        sys_name = meta.get('system_name', 'UnknownSystem')
        system = instantiate_system_by_name(sys_name)
    except Exception:
        print(f"Warning: Could not instantiate system from metadata")
        system = None
    
    # Interactive mode
    if args.interactive:
        if system is None:
            print("Error: System must be available for interactive visualization")
            return
        run_interactive(vf, system, args, tag=args.tag)
        return
    
    # Determine preset and output dir
    preset = args.preset or 'default'
    if args.save_dir is None:
        output_dir = Path('outputs') / 'visualizations' / 'grid_values' / f'{args.tag}' / preset
    else:
        output_dir = Path(args.save_dir)

    if args.output is not None and args.save_dir is None:
        # Back-compat: treat --output as --save-dir
        output_dir = Path(args.output)

    # Preset-driven visualization: explicit --preset OR auto-discover `default` from YAML
    sys_name = meta.get('system_name', 'UnknownSystem')
    preset_node = load_visualization_presets(sys_name, 'default', preset)
    has_preset = isinstance(preset_node, dict) and bool(preset_node.get('slices'))

    if args.preset and not has_preset:
        print(f"✗ Preset not found for system={sys_name}, input=default, preset={preset}")
        return

    if has_preset:
        if not args.preset:
            print(f"ℹ Using preset '{preset}' from visualizations.yaml (pass --preset to pick another)")
        slices: List[Dict[str, Any]] = preset_node.get('slices', []) or []
        if not slices:
            print("⚠ Warning: Preset contains no slices, using empty list")

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving figures to {output_dir}...")

        for idx, s_cfg in enumerate(slices):
            dims = s_cfg.get('dims', [0, 1])
            if 'dims' not in s_cfg:
                print(f"⚠ Warning: Slice missing 'dims', using default {dims}")
            fixed: Dict[int, float] = {int(k): float(v) for k, v in (s_cfg.get('fixed', {}) or {}).items()}
            # GridValue is inherently time-varying: ignore preset `times` (which exists for
            # GridInput/Set snapshots) and always show BRT evolution. Override with --time-indices.
            if args.time_indices is not None:
                times_list = [float(vf._times[i].item()) for i in args.time_indices]
            else:
                tN = int(vf._times.shape[0])
                times_list = [float(vf._times[i].item()) for i in (0, max(0, tN // 2), -1)]
            title = s_cfg.get('title', None)
            if len(dims) != 2:
                print(f"Skipping slice with non-2D dims: {dims}")
                continue

            # Resolve all requested times to grid indices (one figure with subplots over time)
            time_indices: List[int] = []
            for tv in times_list:
                cand = int(nearest_time_index(vf._times, float(tv))[0].item())
                if not args.interpolate and abs(float(vf._times[cand].item()) - float(tv)) > 1e-6:
                    raise ValueError(f"Time {tv} not on grid and --interpolate not set")
                time_indices.append(cand)

            # Filename convention mirrors visualize_grid_set
            fixed_str = '_'.join(f"{k}{v}" for k, v in sorted(fixed.items()))
            base = f"{args.tag}_{preset}_slice{idx}_dims{''.join(map(str, dims))}"
            if fixed_str:
                base += f"_fix{fixed_str}"

            visualize_2d_slices(
                vf, system, time_indices, output_dir,
                dims=list(dims),
                fixed_dims=fixed,
                title_prefix=title,
                filename=f"{base}.png",
                tag=args.tag,
            )

            # Reachable set evolution on the same slice (zero level sets over time)
            visualize_reachable_set_evolution(
                vf, output_dir,
                system=system,
                dims=list(dims),
                fixed_dims=fixed,
                num_frames=args.evolution_frames,
                title_prefix=title,
                filename=f"{base}_evolution.png",
                tag=args.tag,
            )

        # Parameter sweeps: one figure with N subplots, one per value of sweep_dim
        sweeps: List[Dict[str, Any]] = preset_node.get('sweeps', []) or []
        for sidx, sw_cfg in enumerate(sweeps):
            sw_dims = sw_cfg.get('dims', [0, 1])
            sw_sweep_dim = int(sw_cfg['sweep_dim'])
            sw_values = [float(v) for v in sw_cfg['sweep_values']]
            sw_fixed = {int(k): float(v) for k, v in (sw_cfg.get('fixed', {}) or {}).items()}
            sw_time = float(sw_cfg.get('time', 0.0))
            sw_title = sw_cfg.get('title', None)

            fixed_tag = '_'.join(f"{k}{v}" for k, v in sorted(sw_fixed.items()))
            sw_base = f"sweep{sidx}_dims{''.join(map(str, sw_dims))}_sw{sw_sweep_dim}"
            if fixed_tag:
                sw_base += f"_fix{fixed_tag}"
            sw_base += f"_t{sw_time:.1f}"

            visualize_param_sweep(
                vf, system, output_dir,
                dims=list(sw_dims),
                sweep_dim=sw_sweep_dim,
                sweep_values=sw_values,
                fixed_dims=sw_fixed,
                time=sw_time,
                title_prefix=sw_title,
                filename=f"{sw_base}.png",
                tag=args.tag,
            )
    else:
        # No preset in YAML → fall back to CLI args / sensible defaults
        if args.time_indices is None:
            tN = int(vf._times.shape[0]) if getattr(vf, '_times', None) is not None else 1
            time_indices = [0, max(0, tN // 2), -1]
        else:
            time_indices = args.time_indices

        # Parse --fixed "dim=value" pairs (multi)
        fixed_from_cli: Dict[int, float] = {}
        for item in (args.fixed or []):
            if '=' not in item:
                raise ValueError(f"--fixed expects DIM=VALUE, got {item!r}")
            k, v = item.split('=', 1)
            fixed_from_cli[int(k)] = float(v)

        # Resolve `dims`: prefer explicit --dims; else derive from legacy --slice-dim.
        if args.dims is not None:
            dims_arg = args.dims
        else:
            # Legacy behavior: the 2 plotted dims are all dims except slice_dim (plus any --fixed),
            # picked as the two lowest-index dims not in the fixed set.
            excluded = set(fixed_from_cli.keys()) | {args.slice_dim}
            dims_arg = [d for d in range(vf.state_dim) if d not in excluded][:2]
            if len(dims_arg) < 2:
                dims_arg = [0, 1]
            # Honor legacy --slice-value by adding slice_dim to fixed_from_cli
            if args.slice_value is not None:
                fixed_from_cli.setdefault(args.slice_dim, float(args.slice_value))

        visualize_2d_slices(
            vf, system, time_indices, output_dir,
            dims=dims_arg,
            fixed_dims=fixed_from_cli,
            tag=args.tag,
        )
        visualize_reachable_set_evolution(
            vf, output_dir,
            system=system,
            dims=dims_arg,
            fixed_dims=fixed_from_cli,
            num_frames=args.evolution_frames,
            tag=args.tag,
        )
    
    print("\n" + "=" * 60)
    print("✓ Visualization Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
