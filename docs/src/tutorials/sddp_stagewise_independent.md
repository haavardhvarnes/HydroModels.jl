# SDDP — stagewise-independent uncertainty via SDDP.jl

`HydroModels.jl` bridges `LongTermHydroProblem{T, StagewiseIndependent{T}}`
to [SDDP.jl](https://sddp.dev) through the
`HydroModelsOptSDDPExt` extension. This tutorial walks through a
small 2-reservoir cascade with weekly stages and two-realisation
stagewise-independent inflow.

## Setup

```julia
using HydroModelsOpt        # @reexports HydroModelsCore
using HiGHS
using SDDP
```

Loading `SDDP` activates the extension that registers
`solve(::LongTermHydroProblem{T, StagewiseIndependent}, ::SDDPHydroSolver)`.

## Build the problem

```julia
upper = HydroModule{Float64, RegulationReservoir}(
    name = :upper,
    max_vol = 100.0,
    initial_vol = 50.0,
    rated_discharge = 10.0,
    energy_factor = 1.0,
    discharge_to = :lower,
)
lower = HydroModule{Float64, RegulationReservoir}(
    name = :lower,
    max_vol = 100.0,
    initial_vol = 50.0,
    rated_discharge = 10.0,
    energy_factor = 0.8,
)
topology = build_topology([upper, lower])

num_stages = 12
realizations = [
    [Float64[8.0, 4.0], Float64[3.0, 2.0]]    # high vs low inflow per module
    for _ in 1:num_stages
]
probabilities = [[0.5, 0.5] for _ in 1:num_stages]

uncertainty = StagewiseIndependent{Float64}(realizations, probabilities)
stage_prices = [30.0 + 5.0 * (t - 1) for t in 1:num_stages]

prob = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
    modules     = [upper, lower],
    topology    = topology,
    num_stages  = num_stages,
    uncertainty = uncertainty,
    stage_prices = stage_prices,
)
```

`StagewiseIndependent` carries inflow realizations only;
deterministic per-stage prices go in `stage_prices` on the problem
itself. `ProdRiskUncertainty` and `MarkovianUncertainty` carry their
own price representations and leave `stage_prices = nothing`.

## Configure the solver

```julia
solver = SDDPHydroSolver(;
    subproblem_solver = JuMPSolver(HiGHS.Optimizer;
                                   options = Dict(:output_flag => false)),
    max_iterations    = 200,
    num_forward_scenarios = 100,
)
```

## Solve

```julia
sol = solve(prob, solver)
```

This trains the SDDP policy and runs the in-sample forward
simulation. Returns a [`Solution{Float64}`](@ref Solution) with:

| Field | Meaning |
|---|---|
| `sol.objective` | in-sample mean of the simulated total cost |
| `sol.bound` | `SDDP.calculate_bound(sddp_model)` |
| `sol.status` | termination status (`MOI.TerminationStatusCode`) |
| `sol.metadata[:sddp_model]` | the trained `SDDP.PolicyGraph` |
| `sol.metadata[:forward_backward_asymmetric]` | `false` (symmetric path for StagewiseIndependent) |

## Inspect cuts

The trained policy lives in `sol.metadata[:sddp_model]`. Export cuts
via SDDP.jl directly:

```julia
SDDP.write_cuts_to_file(sol.metadata[:sddp_model], "cuts.json")
```

You can also fold them straight into a short-term LP via the
toolchain — see [Long → short toolchain](toolchain_long_to_short.md).

## Unit convention

The bridge treats LP coefficients as unitless. Revenue per stage is:

```
stage_prices[t] · sum(energy_factor[i] · turbine[i] for i)
```

with `turbine` in m³/s; storage in Mm³. For Nordic weekly schedules
in kWh/m³ × EUR/MWh, the caller is responsible for unit scaling
(pre-multiply `energy_factor` by ~0.168 — see
[Code conventions](../architecture.md#code-style)).

## Other uncertainty variants

| Variant | Status |
|---|---|
| `LongTermHydroProblem{T, StagewiseIndependent{T}}` | **landed (B2)** — this tutorial |
| `LongTermHydroProblem{T, ScenarioTree{T}}` | stubbed — B2.5 follow-up via `SDDP.PolicyGraph` |
| `LongTermHydroProblem{T, MarkovianUncertainty{T}}` | stubbed — B2.5 follow-up via `SDDP.MarkovianPolicyGraph` |
| `LongTermHydroProblem{T, ProdRiskUncertainty{T}}` | landed (C2) via `SDDP.MarkovianGraph` — see deviation note below |

## C2 — `ProdRiskUncertainty` via SDDP.jl

For `ProdRiskUncertainty` the bridge builds an
`SDDP.MarkovianGraph(transitions)` from the price-node transitions.
At each stage the inflow realizations are the historical years
embedded in `u.historical_inflow[t, :, :]`.

**Deviation from the Nordic gold standard.** The Gjelsvik–Belsnes–
Haugstad 1999 algorithm uses **asymmetric** forward/backward sampling:
historical scenarios for the forward pass (preserving inflow-price
correlation), and a fitted PAR(1) + Markov node model for the
backward recursion. The C2 MVP uses SDDP.jl's **symmetric** Markovian
machinery for both passes. Returned via
`sol.metadata[:forward_backward_asymmetric] === false` for caller
detection.

A native asymmetric algorithm is **C2.5** on the roadmap; it would
integrate `HydroModelsForecast.PARProcess` sampling on the backward
recursion. See [Status](../status.md).
