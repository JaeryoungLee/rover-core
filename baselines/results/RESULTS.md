# Baseline Comparison Results

## Experimental Setup

**System:** Dubins car (px, py, theta), v=1.0 m/s, control omega in [-1,1] rad/s,
state space [0,20] x [-5,5] x [-pi,pi].

**Obstacle:** Circular, center (16.0, 0.0), radius 1.0 m (bounding box [15,17]x[-1,1]x[-pi,pi]).
Placed on the rover's nominal path toward goal (20, 0, 0).

**Controller:** Pure ReLU MLP [3->128->128->1] trained via supervised imitation of an MPC.
Training MSE: 0.0027. Used identically by both methods.

**Perception uncertainty:** Fixed bounds epsilon = (+-0.5 m, +-0.5 m, +-0.1 rad).
NN sees perturbed state x_hat = x + e; dynamics use true state x.

**Time horizon:** T = 5 s (50 control steps at 0.1 s control period).

**Baseline approach:** Forward reachability under time-reversed dynamics, starting from
the obstacle bounding box. Mathematically equivalent to backward reachability.

**Volume metric:** Grid-rasterized union of all reachable boxes on a 100^3 grid
matching RoVer-CoRe's state-space discretization ([0,20] x [-5,5] x [-pi,pi]).

---

## BRT Volume Over Time (Primary Result)

| T (s) | RoVer-CoRe (m^2*rad) | NNV v2.0 (m^2*rad) | Ratio |
|------:|---------------------:|--------------------:|------:|
| 0.5   | 27.38                | 50.89               | 1.86x |
| 1.0   | 39.86                | 94.91               | 2.38x |
| 1.5   | 58.05                | 162.22              | 2.79x |
| 2.0   | 82.83                | 260.90              | 3.15x |
| 2.5   | 115.33               | 365.31              | 3.17x |
| 5.0   | 356.69               | 653.45              | 1.83x* |

*At T=5s both methods clip at the grid boundary; ratio is compressed.

NNV configuration: 1x1x100 theta-only partitions, approx-star, per-step
interval hull (following NNV's documented approx-star pipeline).

## NNV Tightest Result (T=1s)

At T=1s, we also compute NNV's theoretical tightest result by propagating
each Star from CORA individually (no interval hull between steps):

| Method | Volume (m^2*rad) | Ratio | Runtime |
|--------|----------------:|------:|--------:|
| RoVer-CoRe | 39.86 | 1.00x | <1s (GPU) |
| NNV individual (ntheta=1) | 81.16 | 2.04x | 363s |
| NNV boxed (ntheta=100) | 94.91 | 2.38x | ~80s |
| NNV boxed (ntheta=1) | 124.28 | 3.12x | ~20s |

Individual propagation is intractable beyond T~1s due to exponential Star
growth (1 -> 16 -> 32 -> 64 -> 128 -> 200 Stars over 10 steps). This is
the well-documented set explosion problem in Star-based reachability
(Tran et al. FM 2019, CAV 2020).

---

## Method Details

### RoVer-CoRe (reference)

Hamilton-Jacobi viscosity solution on a 100x100x100 grid. CROWN (auto_LiRPA) for
NN output bounds, integrated into the HJ PDE via the GridSet/GridValue pipeline.
Produces the optimal (exact) backward reachable tube — no overapproximation.
GridValue: `.cache/grid_values/RoverBaseline_BRT_T5.pkl` (100 time steps).

### NNV v2.0 (primary baseline)

NNV's approx-star method for NN controller reachability + CORA zonotope
reachability for nonlinear plant dynamics (NonLinearODE.stepReachStar).
1x1x100 theta-only partitions (no x/y partitioning — NN bounds saturate
regardless). 0.1s control period with 0.01s ODE sub-steps.
Perception uncertainty correctly separated: NN input Star inflated by epsilon,
CORA propagates from true state Star.

Per-step interval hull follows NNV's documented approx-star pipeline
(Tran et al. CAV 2020: "the over-approximate method computes the interval
hull of all reachable sets at each time step and maintains a single reachable
set of the plant throughout the computation"). This is implemented in NNV's
LinearNNCS class via `Star.get_hypercube_hull()`.

### NNV individual propagation (supplementary)

Same pipeline but without interval hull: when CORA returns multiple Stars
(from internal splitting), each is propagated independently. Produces the
tightest possible NNV result but suffers exponential Star growth —
intractable beyond ~T=1s for this system.

Script: `baselines/nnv/verify_individual_propagation.m`

---

## Key Findings

### Irreducible method gap

Even at NNV's theoretical tightest (individual propagation, 2.04x at T=1s),
the set-based approach produces roughly 2x more conservative BRTs than HJ
reachability. This gap reflects the fundamental difference between set-based
propagation (zonotope linearization) and grid-based PDE solving.

### Growing conservatism from wrapping

The NNV-to-RoVer-CoRe volume ratio grows from 1.86x at T=0.5s to ~3.2x at
T=2-2.5s for the boxed approach. The wrapping effect — overapproximation
errors from box-based set representation — accumulates multiplicatively
through the nonlinear Dubins dynamics (Neumaier 1993, Schilling et al. 2022).

### Set explosion

Without per-step interval hull, CORA internally splits Stars when the
linearization error is large (driven by wide theta ranges in cos/sin).
The Star count grows approximately 2x every 3-4 steps, reaching 200 Stars
at T=1s (10 steps). This exponential growth is the fundamental scalability
limitation of set-based NNCS reachability (Tran et al. FM 2019:
worst-case 2^N Stars per ReLU layer).

### NN bound saturation

With perception uncertainty epsilon = (+-0.5m, +-0.5m, +-0.1rad), NNV's
approx-star produces NN output bounds that saturate to the full actuator range
[-1, 1] for all partitions. The conservatism is dominated by the dynamics
propagation, not the NN verification.

### Partitioning sensitivity

Theta partitioning (ntheta) provides ~24% volume reduction at T=1s
(from 124.3 to 94.9 m^2*rad going from ntheta=1 to 100). Diminishing
returns beyond ntheta=25 (~3% further improvement). x/y partitioning
does not help because NN bounds already saturate.

| ntheta | T=0.5s | T=1.0s | T=1.5s | T=2.0s | T=2.5s | T=5.0s |
|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 1      | 70.5   | 124.3  | 179.8  | 261.3  | 367.6  | 653.5  |
| 5      | 59.1   | 111.3  | 175.3  | 260.7  | 366.1  | 653.5  |
| 10     | 54.7   | 103.4  | 171.1  | 266.5  | 348.6  | 653.5  |
| 25     | 52.2   | 97.9   | 165.7  | 262.2  | 366.1  | 653.5  |
| 50     | 51.4   | 95.9   | 163.2  | 261.4  | 365.5  | 653.5  |
| 100    | 50.9   | 94.9   | 162.2  | 260.9  | 365.3  | 653.5  |

All configs converge by T=5s — the wrapping effect dominates regardless
of initial partitioning.

---

## Artifacts

| File | Description |
|------|-------------|
| `comparison_table.json` | Machine-readable comparison data |
| `comparison_table.md` | Formatted comparison table |
| `nnv_partitioning_sensitivity.json` | Full partition sensitivity data |
| `nnv_all_configs.json` | Per-config extraction results |
| `rovercore_brt_volume.json` | RoVer-CoRe volume data |
| `brt_time_series.png` | Main comparison plot (theta=0 slice, T=0.5-2.5s + individual at T=1s) |
| `nnv/reach_results_T5_ntheta{N}.mat` | Raw NNV boxed reachability results |
| `nnv/reach_results_indiv_T1_ntheta1.mat` | NNV individual propagation result (T=1s) |

---

## Visualization

`brt_time_series.png` — BRT growth at theta=0 slice for T=0.5, 1.0, 1.5, 2.0, 2.5s.
Solid lines = RoVer-CoRe, dashed = NNV v2.0 (ntheta=100, boxed). Color progression
(Reds colormap) from light (T=0.5s) to dark (T=2.5s). Blue solid line = NNV
individual propagation at T=1s only (tightest possible, intractable beyond T~1s).

Script: `scripts/baselines/visualize_brt_time_series.py`
