# Baseline Experiments for RoVer-CoRe

## Overview

This document describes the baseline comparison experiments for evaluating
RoVer-CoRe's HJ reachability approach against established NNCS (Neural Network
Control System) verification tools.

The baseline computes the backward reachable tube (BRT) from a single obstacle
region in the Rover environment. The baseline uses **forward reachability
with time-reversed dynamics** to compute the equivalent of the BRT, while
RoVer-CoRe uses backward HJ reachability directly.

## Experimental Setup

### Simplifications

1. **Single obstacle** — Both methods compute the BRT from one obstacle region
   for simplicity. Multiple obstacles could be handled by computing per-obstacle
   BRTs and taking their union in the single-player disturbance setting.

2. **Time-invariant uncertainty** — Perception error bounds are fixed (not
   growing with time), avoiding the need for time-reversed uncertainty schedules
   that the baseline tool does not natively support.

3. **Pure ReLU controller** — The NN controller uses only Linear+ReLU layers
   (no PeriodicEmbedding with sin/cos), since most baseline tools cannot
   propagate sets through sin/cos NN layers. A new NN is trained for this
   experiment and used consistently by both methods.

4. **Time-reversed forward computation** — The baseline computes forward
   reachable tubes starting from the obstacle set under negated dynamics, which
   is mathematically equivalent to backward reachability for the original system.

### System

- **Dynamics:** Dubins car — x' = v*cos(theta), y' = v*sin(theta), theta' = omega
- **Speed:** v = 1.0 m/s
- **Control:** omega in [-1, 1] rad/s
- **State space:** [0, 20] x [-5, 5] x [-pi, pi]
- **Initial state:** (0, 0, 0)
- **Controller:** Pure ReLU MLP trained to imitate the MPC (see below)
- **Perception uncertainty:** Fixed bounds epsilon = (+/-0.5 m, +/-0.5 m, +/-0.1 rad)
- **Time horizon:** T = 5 s

### MPC Controller

The nominal controller is a nonlinear MPC that drives the rover toward the
goal state (20, 0, 0) while avoiding obstacles. It uses:
- Quadratic stage cost on distance to goal: (p_x - 20)^2 + (p_y - 0)^2
- Collision avoidance penalty on proximity to obstacles
- Control bounds: omega in [-1, 1] rad/s

A pure ReLU MLP is trained via supervised learning to imitate the MPC output
over the state space. This NN is used by both methods (NNV + RoVer-CoRe) to
ensure a fair comparison.

### Obstacle

A single circular obstacle placed on the rover's nominal path:
- **Center:** (16.0, 0.0)
- **Radius:** 1.0 m
- **Bounding box (for polytope-based tools):** [15, 17] x [-1, 1] x [-pi, pi]

This obstacle sits directly ahead on the centerline (the MPC's nominal
trajectory from (0,0) toward goal (20,0)), forcing the controller to steer
around it. This is a standalone case study separate from the RoverDark
multi-obstacle environment.

### Time-Reversed Dynamics

For the forward-reachability baseline, the dynamics are negated:
- x' = -v*cos(theta), y' = -v*sin(theta), theta' = -omega
- Initial set: the obstacle bounding box
- Forward reachable tube under reversed dynamics = backward reachable tube
  under original dynamics

The NN controller pi(x + e) is still applied at the current state (it is a
memoryless static map, agnostic to time direction).

## Baseline: NNV 2.0

- **Method:** Star-set NN propagation + zonotope dynamics propagation
- **Paper:** Lopez et al., "NNV 2.0: The Neural Network Verification Tool" (CAV 2023) [canonical NNV 2.0 citation];
  Tran et al., "NNV: The Neural Network Verification Tool for Deep Neural
  Networks and Learning-Enabled Cyber-Physical Systems" (CAV 2020) [NNV 1.0 / NNCS pipeline];
  Tran et al., "Star-Based Reachability Analysis of Deep Neural Networks" (FM 2019) [star set method]
- **Code:** `libraries/NNV/` (see `baselines/SETUP.md` for installation)
- **Language:** MATLAB
- **Why:** Most cited NNCS verification tool; represents the star-set approach;
  longest ARCH-COMP participation history

### NNV Pipeline

1. **NN reachability:** approx-star method propagates Star sets through the
   ReLU network to compute output bounds (omega_lb, omega_ub)
2. **Plant reachability:** CORA zonotope reachability (NonLinearODE.stepReachStar)
   propagates the state set through the nonlinear Dubins dynamics
3. **Perception uncertainty:** NN input Star is inflated by epsilon; CORA
   propagates from the true state bounds (correctly separating perception
   and dynamics uncertainty)
4. **Partitioning:** Theta-only partitioning (1x1xN) to keep cos/sin intervals
   tight. x/y partitioning does not help because NN output bounds saturate
   to the full actuator range regardless.

### NNV Configurations

- **ntheta=1** (no partitioning): Single initial set, no subdivision
- **ntheta=100** (100 theta partitions): Primary reported result

## RoVer-CoRe (reference)

- **Method:** Hamilton-Jacobi reachability (grid-based PDE solver)
- **Solver:** `hj_reachability` (JAX, GPU-accelerated)
- **NN bounds:** alpha,beta-CROWN via auto_LiRPA
- **Advantage:** Computes optimal (tightest) BRT on exact nonlinear dynamics;
  no set representation artifacts or wrapping effect

## Evaluation

Both methods (NNV baseline + RoVer-CoRe) compute the BRT from the same
obstacle under the same dynamics, controller, and uncertainty.

### Quantitative Metrics

1. **BRT volume** — Volume of the backward reachable tube (in state-space
   units: m^2 * rad). RoVer-CoRe produces the tightest (smallest) BRT; the
   baseline volume is equal or larger (more conservative).

2. **Volume ratio** — Ratio of the baseline's BRT volume to RoVer-CoRe's BRT
   volume. A ratio of 1.0 means the baseline matches HJ exactly. Ratios > 1.0
   indicate conservatism (over-approximation).

### Qualitative Evaluation

1. **Shape fidelity** — Visual comparison on 2D slices (p_x vs p_y at fixed theta).
   RoVer-CoRe's BRT has smooth, non-rectangular boundaries that follow the
   nonlinear dynamics. The NNV BRT is compositionally rectangular (box-based propagation) and
   generally bloated along dimensions where wrapping accumulates.

### Expected Outcomes

- **Conservatism:** The NNV baseline produces a BRT that is a superset of
  RoVer-CoRe's BRT. The degree of conservatism grows with time horizon due to
  the wrapping effect (box-based overapproximation compounding through nonlinear
  dynamics). By T=2.5s the ratio reaches ~3x.

### Propagation Modes

We evaluate two NNV propagation modes:

1. **Boxed (primary):** Per-step interval hull, following NNV's documented
   approx-star pipeline (Tran et al. CAV 2020). Tractable at all time horizons.
   With ntheta=100 partitions, produces 2.38x conservatism at T=1s.

2. **Individual (supplementary):** Each Star from CORA propagated independently
   (no hull). Tightest possible NNV result (2.04x at T=1s) but intractable
   beyond T~1s due to exponential Star growth — a well-documented limitation
   of set-based NNCS reachability (Tran et al. FM 2019, CAV 2020;
   Schilling et al. AAAI 2022).

## Directory Structure

```
libraries/
├── NNV/                # Baseline: Star sets + zonotopes
├── hj_reachability/    # RoVer-CoRe's HJ solver
└── auto_LiRPA/         # Shared NN verification library

baselines/
├── nn/                 # Exported NN model files
├── nnv/                # NNV baseline scripts
└── results/            # All results and plots
```

## References

NNV 2.0 is the tool version we run; the approx-star + CORA pipeline we use
was introduced in NNV 1.0. Both should be cited per NNV's README.

```bibtex
@inproceedings{nnv2_cav2023,
  author    = {Lopez, Diego Manzanas and Choi, Sung Woo and Tran, Hoang-Dung and Johnson, Taylor T.},
  title     = {NNV 2.0: The Neural Network Verification Tool},
  booktitle = {Computer Aided Verification: 35th International Conference, CAV 2023,
               Paris, France, July 17--22, 2023, Proceedings, Part II},
  pages     = {397--412},
  publisher = {Springer-Verlag},
  year      = {2023},
  doi       = {10.1007/978-3-031-37703-7_19},
}

@inproceedings{nnv_cav2020,
  author    = {Tran, Hoang-Dung and Yang, Xiaodong and Manzanas Lopez, Diego and Musau, Patrick
               and Nguyen, Luan Viet and Xiang, Weiming and Bak, Stanley and Johnson, Taylor T.},
  title     = {NNV: The Neural Network Verification Tool for Deep Neural Networks
               and Learning-Enabled Cyber-Physical Systems},
  booktitle = {Computer Aided Verification: 32nd International Conference, CAV 2020,
               Los Angeles, CA, USA, July 21--24, 2020, Proceedings, Part I},
  pages     = {3--17},
  publisher = {Springer-Verlag},
  year      = {2020},
  doi       = {10.1007/978-3-030-53288-8_1},
}
```
