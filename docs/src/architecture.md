# Architecture

HydroModels.jl is a **Flyt-style monorepo** of six cooperating Julia
packages. Each subpackage lives in its own root directory with its
own `Project.toml`, UUID, and dependency graph.

```
HydroModels/                                  # repo root
├── Project.toml                              # monorepo manifest
├── dev.jl                                    # Pkg.develop bootstrap
├── docs/
├── examples/
│
├── HydroModels/                              # umbrella — @reexport
├── HydroModelsCore/                          # types + topology + backend abstraction
├── HydroModelsData/                          # SHOP YAML reader
├── HydroModelsOpt/                           # JuMP LP/MILP, SDDP bridge, Lagrangian
├── HydroModelsForecast/                      # PAR(1), Markov price, scenario gen
└── HydroModelsViz/                           # Makie dashboard
```

## The Core / Opt boundary

The hardest line in the codebase is the boundary between
`HydroModelsCore` and `HydroModelsOpt`. The rule is:

> **Core owns everything solver-agnostic. Opt owns everything that
> knows about a solver algorithm.**

### Core

`HydroModelsCore` declares:

- **Physical types** — `HydroModule`, `TurbineUnit`, reservoir-type
  marker types (`RegulationReservoir`, `BufferReservoir`), and SHOP-
  style finer-grained types (`Plant`, `Reservoir`, `Generator`,
  `Pump`, `Tunnel`, `Junction`).
- **Problem formulations** — `AbstractOptProblem` and its concrete
  subtypes (`LongTermHydroProblem`, `ShopShortTermProblem`,
  `ShortTermHydroProblem`, `StochasticShortTermProblem`,
  `FundamentalMarketProblem`), plus `EndValueDescription` and
  the risk-measure hierarchy (`Expectation`, `CVaR`, `NestedCVaR`).
- **Uncertainty containers** — `ScenarioTree`, `StagewiseIndependent`,
  `MarkovianUncertainty`, `ProdRiskUncertainty`. These are the *data
  structures* that hold inflow / price uncertainty information; they
  do not know how to fit themselves.
- **Solution / cut representations** — `Solution`, `WaterValueCuts`,
  `BendersCut`, `CutGroup`, `ReservoirWaterValue`. Cuts cross the
  long-term → short-term boundary as *data*, not as solver
  internals.
- **Backend abstraction** — `ComputeBackend`, `CPUBackend`,
  `GPUBackend`, `default_backend`, `set_default_backend!`. All
  array-allocating routines that downstream code might want to
  dispatch on a backend (`allocate`, `zeros_like`, `ones_like`,
  `ka_backend`, `is_gpu`) live here.
- **Topology** — `WaterTopology`, `build_topology`.
- **Generic utilities** — `PiecewiseLinear`/`PWL` (with `evaluate`
  and `slopes`), `InflowSeries`, `MarketSeries`.

Forbidden in Core:

- `JuMP`. The Core may declare problem types that the solvers
  later wrap in JuMP `Model`s, but Core itself does not import JuMP.
- Any LP / MILP / NLP optimizer (`HiGHS`, `MadNLP`, `SDDP`, …).
- `DataFrames`. Outputs use the lighter
  [Tables.jl](https://github.com/JuliaData/Tables.jl) interface; the
  user converts at the boundary if they want a `DataFrame`.
- Any plotting library.

### Opt

`HydroModelsOpt` declares:

- **Solver hierarchy** — `AbstractSolver`, `AbstractGPUSolver`,
  `JuMPSolver`, `SDDPHydroSolver`, `SLPHydroSolver`,
  `LagrangianHydroSolver`.
- **Solver-side strategy types** — `ParallelismMode`,
  `DecompositionStrategy`, `StepSizeRule` and concrete subtypes
  (`ConstantStep`, `DiminishingStep`, `PolyakStep`).
- **Concrete `solve(::Problem, ::Solver)` dispatches** — one method
  per `(Problem, Solver)` pair that's actually wired up.
- **KA kernels** — `kernels/subgradient.jl`, `kernels/bellman.jl`,
  `kernels/scenario_batch.jl`. These are KernelAbstractions kernels
  driving the Lagrangian GPU-DP path; they compile on every supported
  backend.
- **Long-term → short-term toolchain** —
  `HydropowerToolchain`, `run_toolchain`, cut extraction helpers
  (`extract_end_value_cuts`, `synthetic_end_value_cuts`,
  `with_end_value_cuts`).
- **JuMP baseline output** — `HydroSolution` and the table-builder
  functions.

`HydroModelsOpt` re-exports `HydroModelsCore` via
`Reexport.@reexport using HydroModelsCore`. Downstream users
typically `using HydroModelsOpt` and get the full Core API for free.

### Stubs

The three remaining subpackages (`HydroModelsData`,
`HydroModelsForecast`, `HydroModelsViz`) sit alongside Core and Opt
and depend on Core only. They started life as placeholders; A2, A4,
and C1 milestones grew them into functional packages:

| Subpackage | What it owns |
|---|---|
| `HydroModelsData` | `read_shop_yaml`, `inflow_table`, `market_table`, `reserve_obligation_table`. Tables.jl interface on `InflowSeries` / `MarketSeries`. |
| `HydroModelsForecast` | `PARProcess` (PAR(1) inflow model), `MarkovPriceModel`, `historical_bootstrap`, `to_stagewise_independent`. The `fit`/`sample` idiom from SciML. |
| `HydroModelsViz` | `plot_dashboard`, eleven `plot_*!(ax, sol; …)` panel builders. Makie recipes; backend (CairoMakie / GLMakie / WGLMakie) via weakdep + extension. |

## Type system

The user-facing types fall into three rough families.

### Physical types

Two families coexist:

**ProdRisk-style** — coarser, used by long-term SDDP-class problems:

- `HydroModule{T, R<:ReservoirType}` — logical unit (reservoir + plant
  + connections). The `R` type parameter is one of
  `RegulationReservoir` (contributes a state variable, gets cuts)
  or `BufferReservoir` (follows guidelines, no cuts).
- `TurbineUnit{T}` — unit-level turbine description for SHOP-style
  detail when needed inside a `HydroModule`.

**SHOP-style** — finer, used by short-term LP/MILP problems:

- `Plant{T}` — collection of generators / pumps at a single
  hydroelectric station.
- `Reservoir{T}` — storage with `s_min`, `s_max`, `s0`, optional
  `water_value::ReservoirWaterValue` or downstream
  `spill_to_reservoir` routing destination.
- `Generator{T}` — turbine with `from_res::Vector{String}` (multi-
  source), `to_res::String`, PQ curves, reserve specs.
- `Pump{T}` — symmetric to `Generator`; reversible pump-turbine
  plants own both a `Generator` and a `Pump`.
- `Tunnel{T}` — passive water-routing edge with capacity.
- `Junction{T}` — KP-node with `max_flow` cap (hard) and `flow`
  minimum (soft, high-penalty).

A single problem uses one family or the other; the two are not
mixed.

### Problem types

```
AbstractOptProblem
├── LongTermHydroProblem{T, U<:UncertaintyModel}
├── ShortTermHydroProblem{T}                — ProdRisk-style short-term
├── ShopShortTermProblem{T}                 — SHOP-style short-term LP/MILP
├── StochasticShortTermProblem{T, U}        — short-term with uncertainty
└── FundamentalMarketProblem{T}             — multi-area, endogenous price (EMPS)
```

The `U` type parameter on `LongTermHydroProblem` picks the
uncertainty model and steers solver dispatch.

### Uncertainty types

```
UncertaintyModel
├── ScenarioTree{T}                         — explicit tree of (node, prob)
├── StagewiseIndependent{T}                 — per-stage realisations, independent
├── MarkovianUncertainty{T}                 — per-stage Markov node
└── ProdRiskUncertainty{T}                  — Markov price × historical inflow scenarios
```

`ProdRiskUncertainty` is the Nordic-specific container: the **forward**
pass samples from historical scenarios (preserving inflow-price
correlation) while the **backward** recursion uses a fitted PAR(1)
inflow model + discretized Markov price chain. The container holds
both pieces; the solver chooses which to use where.

## Backend abstraction

All array-allocating routines that downstream code might dispatch on
a backend route through three Core functions:

```julia
allocate(::ComputeBackend, ::Type, dims...)
zeros_like(::ComputeBackend, ::AbstractArray)
ones_like(::ComputeBackend, ::AbstractArray)
```

These wrap [KernelAbstractions.allocate](https://juliagpu.github.io/KernelAbstractions.jl).
The `ka_backend(::ComputeBackend)` function returns the corresponding
`KernelAbstractions.Backend` to pass to a `@kernel` launch.

`default_backend()` reads from a private
`Ref{Union{Nothing, ComputeBackend}}` in Core and falls back to
`CPUBackend()` if nothing was set. GPU vendor extensions install
their backend via `set_default_backend!(GPUBackend(…))` from their
`__init__`. This is the canonical Julia 1.12+ pattern for an
extension that wants to change a parent-package default without
falling foul of the method-overwrite ban during precompile.

## Architectural principles

These are documented in the root [`CLAUDE.md`](https://github.com/haavardhvarnes/HydroModels.jl/blob/main/CLAUDE.md) and
restated here for reference. Every non-trivial change should be
checked against them.

1. **Separation of problem and solver.** A `HydroModule` knows
   nothing about SDDP or SLP. A `SDDPHydroSolver` knows nothing
   about ProdRisk-vs-Brazilian inflow models. The bridge is multiple
   dispatch on `solve(::Problem, ::Solver)`.
2. **Backend-agnostic computation.** All array operations go through
   KernelAbstractions. Vendor-specific code (CUDA / AMDGPU / Metal /
   oneAPI) lives only in `HydroModelsOpt/ext/HydroModelsOpt*Ext.jl`.
3. **Nordic naming.** The user-facing API uses ProdRisk / SHOP
   terminology (`HydroModule`, `WaterValueCuts`, `LoadBlock`)
   because that is what Nordic hydropower engineers actually say.
4. **Soft constraints everywhere.** Per ProdRisk convention every
   operational constraint is violation-with-penalty. Hard
   constraints are rare and explicit.
5. **Two application classes, two types.** Following
   Gjelsvik / Mo / Haugstad 2010, there are two fundamentally
   different problem classes: `LongTermHydroProblem` (single producer,
   exogenous price — ProdRisk) and `FundamentalMarketProblem`
   (multi-area, endogenous price — EMPS).
6. **Forward and backward can use different uncertainty.** Encoded
   explicitly in `ProdRiskUncertainty`: historical inflow scenarios
   for the forward pass, fitted PAR(1) + Markov for the backward
   recursion.

## Code style

All Julia code follows the
[SciML Style Guide](https://github.com/SciML/SciMLStyle):

- `lower_snake_case` for functions and variables, `CamelCase` for
  modules / types, `ALL_CAPS` for constants.
- Generic / parametric types where possible — `Struct{T, A<:AbstractArray{T}}`
  rather than concrete `Float64` / `Array`.
- Immutable structs by default.
- Broadcasting (`@.`) over manual loops.
- NamedTuples for structured returns.

Formatting is enforced via
[Runic.jl](https://github.com/fredrikekre/Runic.jl).
