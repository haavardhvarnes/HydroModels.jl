# Guidance for Claude Code in HydroModelsCore

Extends [/CLAUDE.md](../CLAUDE.md). This file narrows scope for work
inside this subpackage.

## Scope

This package owns:

- ProdRisk-style physical types — `HydroModule`, `TurbineUnit`,
  `ReservoirType`, `RegulationReservoir`, `BufferReservoir`,
  `ModuleConstraints`, `PenaltyValues`, `WaterTopology`. These are the
  coarser logical units used by long-term SDDP scheduling.
- SHOP-style physical types — `Plant`, `Reservoir`, `Generator`,
  `Pump`, `Tunnel`. These are the finer operational structs used by
  short-term scheduling (SHOP algorithm). The two families coexist;
  a single problem uses one or the other depending on horizon.
- Time-series containers — `InflowSeries`, `MarketSeries`. Used by the
  SHOP-style types; produced by parsers in `HydroModelsData`.
- Reserve-market types — `ReserveSpec` (unit-level), `ReserveGroup`
  (area-level), covering the eight canonical Nordic products.
- Problem formulations — `AbstractOptProblem` and its concrete
  subtypes (`LongTermHydroProblem`, `FundamentalMarketProblem`,
  `ShortTermHydroProblem`, `StochasticShortTermProblem`),
  `ElectricityMarket`, `LoadBlock`, `EndValueDescription`, and the
  risk-measure hierarchy (`AbstractRiskMeasure`, `Expectation`,
  `CVaR`, `NestedCVaR`).
- Uncertainty-model **containers** — `ScenarioTree`, `ScenarioNode`,
  `StagewiseIndependent`, `MarkovianUncertainty`,
  `ProdRiskUncertainty`. The *data structures* live here; the
  *fitting code* does not.
- Solution / cut representations — `Solution`, `BendersCut`,
  `WaterValueCuts` (multi-stage multi-reservoir collection),
  `ReservoirWaterValue` (per-reservoir scalar form for SHOP-style
  consumers), `CutGroup` (group-level multi-reservoir cuts). Cut
  representation is problem data (it crosses the long-term →
  short-term boundary), not a solver-internal type.
- Backend abstraction — `ComputeBackend`, `CPUBackend`, `GPUBackend`,
  `default_backend`, `allocate`, `zeros_like`, `ones_like`,
  `is_gpu`, `ka_backend`.
- Topology — `build_topology`.
- Generic utilities — `PiecewiseLinear` / `PWL` (alias, `evaluate`,
  `slopes`), time-stage helpers.

Out of scope — redirect to the right sibling:

- Solver algorithms (SDDP, SLP, Lagrangian, JuMP) → `HydroModelsOpt`.
- KA kernels for solver inner loops → `HydroModelsOpt`.
- Uncertainty **fitting** (PAR(1), Markov price, scenario generation)
  → currently `HydroModelsOpt/src/uncertainty/`, pending migration
  to `HydroModelsForecast`.
- File I/O (YAML, EMPS, HDF5) → `HydroModelsData`.
- Plotting → `HydroModelsViz`.

## Dependencies allowed

Regular deps only:

- `KernelAbstractions`, `GPUArrays` — backend abstraction.
- `MathOptInterface` — `Solution` uses `MOI.TerminationStatusCode`.
  We need MOI but **not** JuMP.
- `OrderedCollections` — `InflowSeries`, `MarketSeries`, reserve
  schedules, and time-varying constraint dicts all use
  `OrderedDict{DateTime, T}` so that insertion order survives the
  parsing → solving → reporting pipeline.
- Stdlibs: `Dates`, `LinearAlgebra`, `SparseArrays`, `Random`,
  `Statistics`.

Forbidden in Core:

- `JuMP` — Opt territory.
- Any solver dep (`HiGHS`, `SDDP`, `MadNLP`, etc.).
- `DataFrames`.
- Any plotting library.

If your change needs one of the forbidden deps, it belongs in `Opt`
or one of the other subpackages — not here.

## Style narrowings

- All types parametric on `T` for autodiff / Float32 / `Dual` / etc.
  compatibility. No concrete `Float64` in struct fields.
- Where arrays appear in structs, parameterize on array type too:
  `Struct{T, A<:AbstractArray{T}}`. This is what makes GPU portability
  possible.
- No solver verbs are defined here. `solve`, `train`, `fit` all live
  in `Opt` (or its future sibling, `Forecast`).
- Cut types (`BendersCut`, `WaterValueCuts`) are problem-data
  containers, not solver internals. They live here even though SDDP
  produces them — they cross the long-term / short-term boundary as
  data.

## Re-export contract

What this package `export`s is what `HydroModelsOpt`'s
`@reexport using HydroModelsCore` propagates to users of the umbrella
`HydroModels`. Keep the export list curated; adding to it is a
public-API change. Non-exported names stay non-exported even if they
look useful — the export list is the seam.

Note: `allocate` (a `KernelAbstractions` function) is *imported*
(`import KernelAbstractions: allocate`), then re-exported, because
Julia 1.12 requires `import` (not `using:`) to extend a function from
another module.

`default_backend()` reads from a private `Ref{Union{Nothing,
ComputeBackend}}` (`_DEFAULT_BACKEND`) and falls back to
`CPUBackend()` if the Ref is `nothing`. GPU vendor extensions in
`HydroModelsOpt` install their backend via `set_default_backend!` from
`__init__` — this avoids the precompile-time method overwriting ban
that hits on Julia 1.12 if extensions try to redefine
`default_backend()` directly.

## Testing

Closed-form / fixture tests for type construction, topology, PWL
curves, backend selection. Tests live in `test/`. Run with

```
julia --project=. -e 'using Pkg; Pkg.test("HydroModelsCore")'
```

from the repo root.
