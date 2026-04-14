# Baseline Comparison Results

BRT volume comparison: RoVer-CoRe vs NNV v2.0.

**Unit:** m^2·rad | **Grid:** 100^3 | **Reference:** RoVer-CoRe (optimal HJ BRT)

## BRT Volume Over Time (boxed, ntheta=100)

| T (s) | RoVer-CoRe | NNV v2.0 | Ratio |
|------:|-----------:|---------:|------:|
| 0.5 | 27.38 | 50.89 | 1.86x |
| 1.0 | 39.86 | 94.91 | 2.38x |
| 1.5 | 58.05 | 162.22 | 2.79x |
| 2.0 | 82.83 | 260.90 | 3.15x |
| 2.5 | 115.33 | 365.31 | 3.17x |
| 5.0 | 356.69 | 653.45 | 1.83x* |

*Both methods clip at grid bounds [0,20]x[-5,5]x[-pi,pi]; ratio compressed.

## NNV Tightest at T=1s (individual propagation)

| Method | Volume | Ratio |
|--------|-------:|------:|
| RoVer-CoRe | 39.86 | 1.00x |
| NNV individual (ntheta=1) | 81.16 | 2.04x |
| NNV boxed (ntheta=100) | 94.91 | 2.38x |
| NNV boxed (ntheta=1) | 124.28 | 3.12x |

Individual propagation is intractable beyond T~1s (exponential Star growth).

_Volume ratio = NNV volume / RoVer-CoRe volume. Ratios > 1.0 = conservatism._
_Volumes computed via grid rasterization on 100^3 grid with cell volume = (range/N)^3._
