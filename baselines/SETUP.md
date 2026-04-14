# Baseline Experiments — Setup Guide

This document walks through setting up all dependencies for the RoVer-CoRe
baseline comparison experiments. Follow these instructions to reproduce the
comparison of RoVer-CoRe's HJ reachability approach against NNV 2.0.

## Overview

| Method | Language | Key Dependencies |
|--------|----------|-----------------|
| RoVer-CoRe (reference) | Python/JAX | `hj_reachability`, `auto_LiRPA` |
| NNV 2.0 | MATLAB | 9 MATLAB toolboxes |

## 1. RoVer-CoRe (Main Environment)

The main RoVer-CoRe Python environment handles:
- Training the pure ReLU NN controller
- Computing CROWN bounds (GridSet)
- Solving the HJ PDE (GridValue)
- Comparison visualization

### Setup

```bash
# Activate the project venv
source env/bin/activate

# Verify key packages
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import jax; print(f'JAX {jax.__version__}')
import do_mpc; print(f'do_mpc {do_mpc.__version__}')
"
```

### Quick test

```bash
# Verify the RoverBaseline system loads
python -c "from src.impl.systems.rover_baseline import RoverBaseline; s = RoverBaseline(); print(f'State dim: {s.state_dim}, Obstacles: {len(s.obstacles)}')"
```

## 2. NNV 2.0 (MATLAB)

Uses Star-set NN propagation + zonotope dynamics propagation.

### Step 1: Install MATLAB

NNV requires **MATLAB R2023a or later** with these toolboxes:

| Toolbox | Required |
|---------|----------|
| Computer Vision Toolbox | Yes |
| Control System Toolbox | Yes |
| Deep Learning Toolbox | Yes |
| Image Processing Toolbox | Yes |
| Optimization Toolbox | Yes (linprog LP solver) |
| Parallel Computing Toolbox | Yes |
| Statistics and Machine Learning Toolbox | Yes |
| Symbolic Math Toolbox | Yes |
| System Identification Toolbox | Yes |
| Deep Learning Toolbox Converter for ONNX | Recommended |

### Step 2: Clone NNV

```bash
git clone https://github.com/verivital/nnv.git libraries/NNV
```

### Step 3: Install NNV

```bash
# From MATLAB command window or batch mode:
matlab -batch "
  cd('libraries/NNV/code/nnv');
  install;
  disp('NNV installation complete');
"
```

The `install` script will:
- Download and configure `tbxmanager` (third-party toolbox manager)
- Install MPT (Multi-Parametric Toolbox) and dependencies
- Install LP solvers: GLPK, SeDuMi
- Add NNV to the MATLAB path

### Step 4: Verify NNV Installation

```bash
matlab -batch "
  cd('libraries/NNV/code/nnv');
  startup_nnv;
  check_nnv_setup;
"
```

This runs comprehensive diagnostics checking:
- NNV and MATLAB versions
- Required toolboxes
- Core classes (Star, NNCS, etc.)
- LP solver availability

### NNV Troubleshooting

- **"Undefined function 'Star'"**: Run `startup_nnv` to re-add NNV to path.
- **"Undefined function 'zonotope'"**: Initialize the CORA submodule:
  `cd libraries/NNV && git submodule update --init code/nnv/engine/cora`
- **"Missing toolbox"**: Install the required toolbox via MATLAB Add-Ons.
- **GLPK mex errors**: The installer auto-detects your platform. If it fails,
  manually build glpkmex.

## Shared Neural Network

Both methods use the same trained pure ReLU MLP controller:

```
baselines/nn/
├── RoverBaseline_MPC_NN.onnx   # ONNX format (general use)
└── RoverBaseline_MPC_NN.mat    # MATLAB .mat with weights/bias cell arrays (for NNV)
```

The PyTorch checkpoint is at `.cache/nn_inputs/RoverBaseline_MPC_NN.pth`
with metadata at `.meta.json`.

To regenerate exports after retraining:
```bash
source env/bin/activate
python scripts/baselines/export_nn.py
```

## Running the Experiments

After all environments are set up:

```bash
# 1. Train the shared NN (if not already done)
source env/bin/activate
python scripts/baselines/train_baseline_nn.py --num-samples 500000 --epochs 100000

# 2. Export to all formats
python scripts/baselines/export_nn.py

# 3. Compute RoVer-CoRe BRT
python scripts/grid_value/build_grid_value.py \
  --dynamics RoverBaseline --system RoverBaseline \
  --control-grid-set-tag RoverBaseline_MPC_NN_Box_Clamped \
  --tag RoverBaseline_BRT_T5 --time-horizon 5.0 --time-steps 100 \
  --accuracy very_high

# 4. Run NNV baseline (all partition configs)
bash baselines/nnv/run_all_configs.sh

# 5. Visualize results
python scripts/baselines/visualize_brt_time_series.py
```

## Directory Structure

```
RoVer-CoRe-PRIVATE/
├── baselines/                    # Baseline experiment artifacts
│   ├── SETUP.md                  # This file
│   ├── nn/                       # Exported NN in all formats
│   ├── nnv/                      # NNV MATLAB scripts
│   │   ├── verify_reversed_dubins.m        # Main NNV verification (boxed)
│   │   ├── verify_individual_propagation.m # Individual Star propagation (tightest)
│   │   ├── extract_results.py              # Extract results from .mat files
│   │   └── run_all_configs.sh              # Run all partition configs
│   └── results/                  # Results from all methods
├── libraries/
│   ├── NNV/                      # NNV (cloned from verivital/nnv)
│   ├── hj_reachability/          # RoVer-CoRe's HJ solver
│   └── auto_LiRPA/              # Shared NN verification library
├── src/impl/
│   ├── systems/rover_baseline.py         # RoverBaseline system definition
│   ├── hj_reachability/rover_baseline.py # HJ dynamics class
│   └── inputs/standalone/controls/rover_baseline/
│       └── mpc.py                        # MPC controller
├── scripts/baselines/
│   ├── train_baseline_nn.py      # Train the pure ReLU NN
│   ├── export_nn.py              # Export NN to ONNX/MAT/text
│   ├── visualize_brt_time_series.py  # Time-series BRT comparison plot
│   ├── compute_brt_volume.py        # RoVer-CoRe BRT volume computation
│   └── compare_results.py           # Compile comparison tables
└── BASELINES.md                  # Experimental design and rationale
```
