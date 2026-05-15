"""
    HydroModelsForecast

Forecasting layer of the HydroModels meta-package. Implements the
statistical models used to drive stochastic scheduling:

- **`PARProcess`** — periodic auto-regressive (order 1) inflow model
  (Nordic backward-recursion default).
- **`MarkovPriceModel`** — PRISMOD-style discretized price chain.
- **Scenario generation utilities** — historical bootstrap +
  conversion to Core's `StagewiseIndependent` uncertainty type.

The API follows the `fit(::Type{Model}, data)` /
`sample(rng, model, n)` convention. All sampling routines take an
explicit `AbstractRNG` argument for reproducibility.

# Quick example

```julia
using Random, HydroModelsForecast, HydroModelsCore

# 10 historical years × 52 weeks of inflow
history = randn(10, 52) .+ 10.0
par = fit(PARProcess, history)

rng = MersenneTwister(0)
trajectories = sample(rng, par, 12.0, 100)   # 100 simulated years
uncertainty = to_stagewise_independent(trajectories)
# uncertainty :: StagewiseIndependent{Float64}, ready for SDDP
```
"""
module HydroModelsForecast

using LinearAlgebra: dot
using Random
using Statistics

using HydroModelsCore

include("par.jl")
include("markov.jl")
include("scenarios.jl")

export PARProcess, MarkovPriceModel
export fit, sample
export historical_bootstrap, to_stagewise_independent
export num_periods, num_stages, num_nodes

end # module HydroModelsForecast
