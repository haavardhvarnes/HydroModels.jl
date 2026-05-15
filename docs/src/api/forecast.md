# `HydroModelsForecast` — API reference

The forecast layer hosts the statistical models that drive
stochastic scheduling:

- **`PARProcess`** — periodic auto-regressive order-1 inflow model
  (the Nordic backward-recursion default).
- **`MarkovPriceModel`** — PRISMOD-style discretized price chain.
- **Scenario generation utilities** — historical bootstrap plus a
  conversion to Core's `StagewiseIndependent` uncertainty type.

The API follows the SciML `fit(::Type{Model}, data)` /
`sample(rng, model, n)` convention. All sampling routines take an
explicit `AbstractRNG` for reproducibility.

## Statistical models

```@docs
PARProcess
MarkovPriceModel
```

## Fitting and sampling

```@docs
fit
sample
```

## Scenario generation

```@docs
historical_bootstrap
to_stagewise_independent
```

## Introspection

```@docs
num_periods
num_stages
num_nodes
```

## Worked example

```julia
using Random, HydroModelsForecast, HydroModelsCore

# 10 historical years × 52 weeks of inflow
history = randn(10, 52) .+ 10.0
par = fit(PARProcess, history)

rng = MersenneTwister(0)
trajectories = sample(rng, par, 12.0, 100)   # 100 simulated years

uncertainty = to_stagewise_independent(trajectories)
# `uncertainty::StagewiseIndependent{Float64}` — ready for SDDP
```
