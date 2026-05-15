# Toolchain — long-term → short-term

The Nordic convention is to solve a long-term stochastic problem
(weekly stages, SDDP-class), extract its Bellman cuts at the end-of-
horizon node, and feed those cuts into a short-term LP as its end-
value description. This tutorial wires both halves end-to-end.

The complete script lives at
[`examples/02_toolchain_long_to_short.jl`](https://github.com/haavardhvarnes/HydroModels.jl/blob/main/examples/02_toolchain_long_to_short.jl)
in the repo.

## Setup

```julia
using HydroModelsOpt
using HydroModelsData
using HiGHS
using SDDP
```

## Build the long-term problem

```julia
upper = HydroModule{Float64, RegulationReservoir}(
    name = :upper, max_vol = 100.0, initial_vol = 50.0,
    rated_discharge = 10.0, energy_factor = 1.0,
    discharge_to = :lower,
)
lower = HydroModule{Float64, RegulationReservoir}(
    name = :lower, max_vol = 100.0, initial_vol = 50.0,
    rated_discharge = 10.0, energy_factor = 0.8,
)
topology = build_topology([upper, lower])

num_stages = 6
realizations = [[Float64[6.0, 3.0], Float64[2.0, 1.0]] for _ in 1:num_stages]
probabilities = [[0.5, 0.5] for _ in 1:num_stages]
uncertainty = StagewiseIndependent{Float64}(realizations, probabilities)
stage_prices = [30.0 + 5.0 * (t - 1) for t in 1:num_stages]

long_term = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
    modules     = [upper, lower],
    topology    = topology,
    num_stages  = num_stages,
    uncertainty = uncertainty,
    stage_prices = stage_prices,
)
```

## Build the short-term template

```julia
short_yaml = joinpath(dirname(pathof(HydroModelsData)),
                      "..", "test", "data", "minimal.yaml")
parsed = read_shop_yaml(short_yaml; market_name = "NO3")
short_term_tmpl = ShopShortTermProblem(parsed)
```

## Wire the toolchain

```julia
SOLVER = JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false))

tc = HydropowerToolchain{Float64}(
    longterm_model     = long_term,
    shortterm_template = short_term_tmpl,
    longterm_solver    = SDDPHydroSolver(;
        subproblem_solver = SOLVER,
        max_iterations    = 20,
        num_forward_scenarios = 10,
    ),
    shortterm_solver   = SOLVER,
    reservoir_map      = Dict{Symbol, String}(),    # names match directly
)
```

`reservoir_map` is empty here because long-term `:upper` / `:lower`
match short-term `"upper"` / `"lower"` after `String(symbol)`. If
the two models use different conventions (e.g. long-term
`:Aurland_upper` vs short-term `"Aurland_HRV_upper"`), populate
`reservoir_map::Dict{Symbol, String}` with the explicit translations.

## Path A — synthetic cuts (bypass long-term solve)

When cuts come from an external source (a previous ProdRisk run, the
literature, or a synthetic test fixture), supply them directly and
the long-term solve is skipped:

```julia
synthetic = synthetic_end_value_cuts(
    ["upper", "lower"],
    [50_000.0, 30_000.0, 70_000.0],
    [1.0e6 0.5e6 1.5e6; 0.5e6 0.25e6 0.75e6],
)
res_a = run_toolchain(tc; cuts = synthetic)
@show res_a.long === nothing      # true — skipped
@show res_a.short.termination
@show res_a.short.objective
@show length(res_a.shortterm_with_cuts.cut_groups)
```

## Path B — full chain (long-term solve + cut extraction)

```julia
res_b = run_toolchain(tc)
@show res_b.long.status
@show res_b.long.bound
@show length(res_b.cuts)
@show res_b.short.objective
```

`extract_end_value_cuts(sol, long_prob)` reaches into SDDP.jl's
`nodes[1].bellman_function.global_theta.cut_oracle.cuts` to pull the
stage-1 continuation-value cuts (which serve as the end-of-horizon
water value for a short-term horizon ending just before the
long-term starts). The path through SDDP.jl internals is fragile
across SDDP.jl versions; on failure the function warns and returns
an empty cut vector. Callers should detect this with `isempty(cuts)`
and either fall back to `synthetic_end_value_cuts` or skip cut
injection.

A robust round-trip — emitting `WaterValueCuts{T}` (Core's richer
multi-stage / multi-price-node form) from `SDDP.write_cuts_to_file`
JSON, and re-reading them on a fresh process — is a B3.5 follow-up.

## What gets returned

```julia
res = run_toolchain(tc; cuts = synthetic)
# res.long                 — Solution{T} from the long-term SDDP solve (nothing if bypassed)
# res.short                — HydroSolution{T} from the short-term LP with cuts injected
# res.cuts                 — Vector{CutGroup{T}} extracted from SDDP and fed in
# res.shortterm_with_cuts  — the configured short-term problem (cuts attached)
```

## Determinism

The short-term LP is purely deterministic; given the same cuts, the
dispatch is bit-identical across reruns (modulo solver-specific
numerical noise). The shipped `test/test_toolchain.jl` asserts this
with `==` on the storage and dispatch tables across two consecutive
`run_toolchain` calls with synthetic cuts.
