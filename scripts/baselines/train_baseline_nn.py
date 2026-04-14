#!/usr/bin/env python3
"""Train a pure ReLU MLP to imitate the RoverBaseline MPC controller.

No PeriodicEmbedding — the NN is pure Linear+ReLU for baseline tool compatibility.
Uses periodic_input_dims=[] so the existing MLP class skips the sin/cos embedding.

Usage:
  source env/bin/activate
  python scripts/baselines/train_baseline_nn.py [--num-samples 50000] [--epochs 50000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impl.systems.rover_baseline import RoverBaseline
from src.impl.inputs.standalone.controls.rover_baseline.mpc import RoverBaseline_MPC
from src.utils.nn import MLP


def generate_mpc_data(
    system: RoverBaseline,
    mpc: RoverBaseline_MPC,
    num_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (state, control) training pairs from MPC."""
    print(f"Generating {num_samples} MPC training samples...")
    low = system.state_limits[0]
    high = system.state_limits[1]
    u = torch.rand(num_samples, system.state_dim)
    states = low + u * (high - low)

    t0 = time.time()
    with torch.no_grad():
        controls = mpc.input(states, 0.0)
    elapsed = time.time() - t0
    print(f"  MPC evaluation: {elapsed:.1f}s ({num_samples / elapsed:.0f} samples/s)")

    return states.to(device), controls.to(device)


def main():
    p = argparse.ArgumentParser(description="Train pure ReLU MLP for RoverBaseline")
    p.add_argument('--num-samples', type=int, default=50000, help='Number of MPC training samples')
    p.add_argument('--epochs', type=int, default=50000, help='Training epochs')
    p.add_argument('--batch-size', type=int, default=100000, help='Batch size')
    p.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    p.add_argument('--hidden', type=int, default=128, help='Hidden layer width')
    p.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    p.add_argument('--tag', type=str, default='RoverBaseline_MPC_NN', help='Output tag')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint-freq', type=int, default=5000, help='Checkpoint every N epochs')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Instantiate system and MPC
    system = RoverBaseline()
    mpc = RoverBaseline_MPC()
    mpc.bind(system)

    # Generate training data
    states, controls = generate_mpc_data(system, mpc, args.num_samples, device)

    # Compute normalization ranges
    x_min = states.min(dim=0).values.tolist()
    x_max = states.max(dim=0).values.tolist()
    y_min = controls.min(dim=0).values.tolist()
    y_max = controls.max(dim=0).values.tolist()

    # Build pure ReLU MLP (periodic_input_dims=[] means no PeriodicEmbedding)
    input_dim = system.state_dim
    output_dim = 1
    sizes = [input_dim, *([args.hidden] * args.layers), output_dim]
    model = MLP(
        sizes,
        input_min=x_min,
        input_max=x_max,
        output_min=y_min,
        output_max=y_max,
        periodic_input_dims=[],  # NO periodic embedding — pure ReLU
        device=device,
    ).to(device)

    print(f"Model architecture: {sizes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  periodic_input_dims=[] (pure ReLU, no PeriodicEmbedding)")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mse = float('inf')
    cache_dir = PROJECT_ROOT / '.cache' / 'nn_inputs'
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cache_dir / 'checkpoints' / 'RoverBaseline' / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume
    last_path = ckpt_dir / 'last.pth'
    start_epoch = 0
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_mse = ckpt.get('best_mse', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best MSE: {best_mse:.6f}")

    print(f"\nTraining for {args.epochs - start_epoch} epochs (batch_size={args.batch_size}, lr={args.lr})...")

    # GPU-fast training: keep all data on device, random permutation per epoch
    N = states.shape[0]
    bs = min(args.batch_size, N)

    pbar = tqdm(range(start_epoch, args.epochs), desc="Training", initial=start_epoch, total=args.epochs)
    for epoch in pbar:
        # Random permutation
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            x_batch = states[idx]
            y_batch = controls[idx]
            pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_mse = epoch_loss / n_batches

        if avg_mse < best_mse:
            best_mse = avg_mse
            torch.save(model.state_dict(), cache_dir / f'{args.tag}.pth')

        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_mse': best_mse,
            }, last_path)
            pbar.set_postfix(mse=f"{avg_mse:.6f}", best=f"{best_mse:.6f}")

    # Save final model
    torch.save(model.state_dict(), cache_dir / f'{args.tag}.pth')

    # Save metadata
    meta = {
        'sizes': sizes,
        'input_min': x_min,
        'input_max': x_max,
        'output_min': y_min,
        'output_max': y_max,
        'periodic_input_dims': [],
        'time_invariant': True,
        'system_name': 'RoverBaseline',
        'input_type': 'control',
        'input_class': 'RoverBaseline_MPC',
        'checkpoint': {
            'epoch': args.epochs,
            'best_mse': best_mse,
        },
    }
    meta_path = cache_dir / f'{args.tag}.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Best MSE: {best_mse:.6f}")
    print(f"  Model saved: {cache_dir / f'{args.tag}.pth'}")
    print(f"  Metadata saved: {meta_path}")


if __name__ == '__main__':
    main()
