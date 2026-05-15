# Guidance for Claude Code in HydroModelsForecast

Extends [/CLAUDE.md](../CLAUDE.md). This file narrows scope for work
inside this subpackage.

## Status

**Active** — PAR(1), Markov price-node, and historical-bootstrap
scenario generation landed in milestone C1 of the phased plan. The
stubs that previously lived under `HydroModelsOpt/src/uncertainty/`
have been deleted; this package is their new home.

## Scope

This package owns:

- **`PARProcess`** — periodic auto-regressive (order 1) inflow model.
  Fits μ, φ, and σ per period from multi-year historical data; samples
  full-year trajectories via OLS-fitted recurrence.
- **`MarkovPriceModel`** — PRISMOD-style discretized price chain. Fits
  per-stage equal-frequency node values (`K` bins, mean of each bin)
  and empirical transition matrices between consecutive stages.
- **Scenario generation** — `historical_bootstrap(rng, history, n)`
  whole-row resampling, and `to_stagewise_independent(trajectories)`
  converting a list of trajectories into Core's
  `StagewiseIndependent{T}` for direct use with the SDDP bridge from
  milestone B2.

The API follows the `fit(::Type{Model}, data; kwargs...)` /
`sample(rng, model, n)` convention; every sampling routine takes an
explicit `AbstractRNG` for reproducibility.

## Out of scope

- Uncertainty-model **containers** (`ScenarioTree`,
  `MarkovianUncertainty`, `ProdRiskUncertainty`,
  `StagewiseIndependent`) → `HydroModelsCore`. Those are the data
  structures; the *fitting code* lives here.
- Solver algorithms → `HydroModelsOpt`.
- Plotting / dashboards → `HydroModelsViz`.

## Dependencies allowed

Regular deps:

- `HydroModelsCore` — for `StagewiseIndependent{T}` construction.
- Stdlibs: `Random`, `Statistics`, `LinearAlgebra` (just for `dot`).

Intentional non-deps (today):

- **No `Distributions`** — `randn`-based normal noise and a simple
  inline categorical sampler suffice. Add when a model needs heavier
  distributional support (e.g. SkewNormal residuals).
- **No `Clustering`** — quantile-based discretization beats k-means
  on 1D price clusters in practice, and avoids a dep with its own
  algorithm-selection knobs.
- **No `Optim` / `JuMP`** — the model fits in this package use OLS
  closed forms or empirical-count normalization. The ProdRisk
  conditional-mean transition adjustment (a small QP) is the only
  thing that would need an optimizer; that's a follow-up.

## Style narrowings

- **`fit` / `sample` are the verbs**. Defined in this package and
  exported. Users who also `using Distributions` get a clash; both
  packages then need module-qualified calls. Documented in the module
  docstring.
- **Explicit `AbstractRNG`** in every sampling API. No
  `Random.default_rng()` shortcut — tests need reproducibility, and
  hiding the RNG makes test failures hard to bisect.
- **Period and stage indexing**. PAR uses `period` (1..52 for weekly
  data), Markov uses `stage` (1..n_stages). The wrappers `num_periods`
  and `num_stages` make the distinction explicit at the call site.
- **No autodiff support yet**. Sampling uses `randn` which is
  AD-incompatible; fitting uses `Statistics.std` (`corrected = true`).
  If `Float32` / `Dual` / autodiff becomes a goal, switch to
  AD-friendly variance estimators and re-route the RNG through a
  reparameterized form.

## End-to-end pattern

```julia
using Random, HydroModelsForecast, HydroModelsCore

# 10 historical years × 52 weeks
history = randn(10, 52) .+ 10.0
par = fit(PARProcess, history)

rng = MersenneTwister(0)
trajectories = sample(rng, par, 12.0, 100)   # 100 simulated years
uncertainty = to_stagewise_independent(trajectories)
# uncertainty :: StagewiseIndependent{Float64} — feed straight to
# LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(...).
```

## Testing

```
julia --project=. -e 'using Pkg; Pkg.test("HydroModelsForecast")'
```

Tests cover: PAR fit recovers constant means and known φ; sample
produces correct length, starts at `x0`, and converges to period
means; Markov fit produces row-stochastic transitions and
in-range trajectories; bootstrap preserves whole rows;
`to_stagewise_independent` round-trips into a Core
`StagewiseIndependent`.

## Pitfalls

1. **PAR period 1**. The first period has no previous period to
   regress against; `φ[1]` and `σ[1]` are conventionally zero.
   Trajectories start with the user-supplied `x0` at period 1 and
   advance from there.
2. **Markov node assignment is nearest-by-value**. Two scenarios with
   the same price could map to different nodes if the discretization
   places a bin boundary between them; the rule is "minimum absolute
   distance, tie-break to lower index". Deterministic but worth
   knowing.
3. **`to_stagewise_independent` assumes scalar realizations per
   stage**. For multi-variable uncertainty (inflow per reservoir +
   price), construct `StagewiseIndependent` manually with realization
   vectors of length `n_variables`.
4. **Empty Markov rows**. A node that is never visited at some stage
   leaves the corresponding row of the transition matrix all zero
   after counting. The fit replaces that row with a uniform
   distribution; if you care about realistic transitions there,
   either resample with more scenarios or treat the model as
   under-fit.
