#!/usr/bin/env python3
"""Export the trained pure ReLU MLP to ONNX and MATLAB .mat formats.

Normalization layers (Normalize/Unnormalize) are folded into the first/last linear
layers so the exported models take raw (x, y, theta) and output raw omega.

Usage:
  source env/bin/activate
  python scripts/baselines/export_nn.py [--tag RoverBaseline_MPC_NN]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.nn import MLP


def load_model(tag: str, device: torch.device) -> tuple[MLP, dict]:
    """Load trained model and metadata."""
    cache_dir = PROJECT_ROOT / '.cache' / 'nn_inputs'
    meta_path = cache_dir / f'{tag}.meta.json'
    pth_path = cache_dir / f'{tag}.pth'

    with open(meta_path) as f:
        meta = json.load(f)

    model = MLP(
        meta['sizes'],
        input_min=meta['input_min'],
        input_max=meta['input_max'],
        output_min=meta['output_min'],
        output_max=meta['output_max'],
        periodic_input_dims=meta['periodic_input_dims'],
        device=device,
    ).to(device)
    pth_data = torch.load(pth_path, map_location=device, weights_only=False)
    state_dict = pth_data['state_dict'] if isinstance(pth_data, dict) and 'state_dict' in pth_data else pth_data
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta


def extract_folded_weights(model: MLP) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract linear layer weights with normalization folded in.

    Returns list of (W, b) tuples for each layer, where:
    - First layer has input normalization baked in
    - Last layer has output unnormalization baked in
    - Intermediate layers are unchanged
    """
    # Get normalization parameters
    input_mean = model.input_normalizer.mean.detach().cpu().numpy()
    input_hw = model.input_normalizer.halfwidth.detach().cpu().numpy()
    output_mean = model.output_unnormalizer.mean.detach().cpu().numpy()
    output_hw = model.output_unnormalizer.halfwidth.detach().cpu().numpy()

    # Extract raw linear layers from model.nn (Sequential of Linear+ReLU)
    linear_layers = [m for m in model.nn.modules() if isinstance(m, torch.nn.Linear)]

    layers = []
    for i, layer in enumerate(linear_layers):
        W = layer.weight.detach().cpu().numpy()  # [out, in]
        b = layer.bias.detach().cpu().numpy()     # [out]

        if i == 0:
            # Fold input normalization: x_norm = (x - mean) / hw
            # Linear(x_norm) = W @ ((x - mean) / hw) + b
            #                 = (W / hw) @ x + (b - W @ (mean / hw))
            scale = 1.0 / input_hw  # [in]
            W = W * scale[np.newaxis, :]  # broadcast across output dim
            b = b - W @ input_mean

        if i == len(linear_layers) - 1:
            # Fold output unnormalization: y = y_norm * hw + mean
            # unnorm(Linear(x)) = hw * (W @ x + b) + mean
            #                    = (hw * W) @ x + (hw * b + mean)
            W = output_hw[:, np.newaxis] * W
            b = output_hw * b + output_mean

        layers.append((W.copy(), b.copy()))

    return layers


def verify_export(model: MLP, layers: list[tuple[np.ndarray, np.ndarray]], device: torch.device):
    """Verify folded weights match model output on random inputs."""
    test_inputs = torch.tensor([
        [10.0, 0.0, 0.0],
        [16.0, 0.0, 0.5],
        [5.0, -2.0, -1.0],
        [18.0, 3.0, 2.5],
        [0.5, -4.0, -3.0],
    ], dtype=torch.float32, device=device)

    with torch.no_grad():
        expected = model(test_inputs).cpu().numpy()

    # Forward pass through folded layers
    x = test_inputs.cpu().numpy()
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:  # ReLU on all but last
            x = np.maximum(x, 0)

    max_err = np.max(np.abs(x - expected))
    print(f"  Verification max error: {max_err:.2e}")
    if max_err > 1e-4:
        print(f"  WARNING: Large verification error! Check folding logic.")
        print(f"  Expected:\n{expected}")
        print(f"  Got:\n{x}")
    return max_err


def export_onnx(model: MLP, layers: list[tuple[np.ndarray, np.ndarray]], out_path: Path):
    """Export to ONNX with normalization folded in."""
    # Build a clean Sequential model with folded weights
    modules = []
    for i, (W, b) in enumerate(layers):
        linear = torch.nn.Linear(W.shape[1], W.shape[0])
        linear.weight.data = torch.from_numpy(W).float()
        linear.bias.data = torch.from_numpy(b).float()
        modules.append(linear)
        if i < len(layers) - 1:
            modules.append(torch.nn.ReLU())

    folded_model = torch.nn.Sequential(*modules)
    folded_model.eval()

    dummy = torch.randn(1, 3)
    torch.onnx.export(
        folded_model,
        dummy,
        str(out_path),
        input_names=['state'],
        output_names=['control'],
        dynamic_axes={'state': {0: 'batch'}, 'control': {0: 'batch'}},
        opset_version=17,
    )
    print(f"  ONNX saved: {out_path}")


def export_matlab(layers: list[tuple[np.ndarray, np.ndarray]], out_path: Path):
    """Export to MATLAB .mat format with cell arrays of weights and biases."""
    from scipy.io import savemat

    weights = np.empty(len(layers), dtype=object)
    biases = np.empty(len(layers), dtype=object)
    for i, (W, b) in enumerate(layers):
        weights[i] = W.astype(np.float64)
        biases[i] = b.reshape(-1, 1).astype(np.float64)

    savemat(str(out_path), {
        'weights': weights,
        'bias': biases,
        'num_layers': len(layers),
        'input_dim': layers[0][0].shape[1],
        'output_dim': layers[-1][0].shape[0],
    })
    print(f"  MATLAB .mat saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Export trained ReLU MLP to multiple formats")
    p.add_argument('--tag', default='RoverBaseline_MPC_NN', help='Model tag')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = PROJECT_ROOT / 'baselines' / 'nn'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{args.tag}'...")
    model, meta = load_model(args.tag, device)
    print(f"  Architecture: {meta['sizes']}")
    print(f"  periodic_input_dims: {meta['periodic_input_dims']}")

    print("Extracting folded weights...")
    layers = extract_folded_weights(model)

    print("Verifying export accuracy...")
    verify_export(model, layers, device)

    print("Exporting ONNX...")
    export_onnx(model, layers, out_dir / f'{args.tag}.onnx')

    print("Exporting MATLAB .mat...")
    export_matlab(layers, out_dir / f'{args.tag}.mat')

    print(f"\nAll exports complete in {out_dir}/")


if __name__ == '__main__':
    main()
