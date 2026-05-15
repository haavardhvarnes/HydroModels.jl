# HydroModels.jl

A research-grade Julia **meta-package** for hydropower modelling. The
ambition is to let a user describe a hydro problem once — modules,
reservoirs, plants, uncertainty, market — and dispatch the same
problem to several solver families:

| Solver family | Problem class | Use case |
|---|---|---|
| **JuMP LP / MILP** | `ShopShortTermProblem` | short-term unit dispatch with reserves |
| **SDDP family** | `LongTermHydroProblem` | medium / long-term stochastic scheduling, Markovian price |
| **Lagrangian on GPU** | scenario-decomposable variants | research path for parallel decomposition |
| **SLP (SHOP)** | `ShortTermHydroProblem` | dynamic MILP linearisation (planned) |

Backend-agnostic from day one via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) —
NVIDIA CUDA, AMD ROCm, Apple Metal, and Intel oneAPI extensions all
plug into the same dispatch surface.

## Quick start

```julia
using HydroModelsOpt        # @reexports HydroModelsCore
using HydroModelsData
using HiGHS

parsed = read_shop_yaml(joinpath(dirname(pathof(HydroModelsData)),
                                 "..", "test", "data", "minimal.yaml"))
prob   = ShopShortTermProblem(parsed)
sol    = solve(prob, JuMPSolver(HiGHS.Optimizer))

@show sol.objective
@show keys(sol.storage)
```

## What's here

- **[Installation](installation.md)** — registry setup and the UUID
  workaround for the `HydroModels.jl` name collision.
- **[Architecture](architecture.md)** — the six subpackages, the
  Core / Opt boundary, the type system, the backend abstraction.
- **[Tutorials](tutorials/lp_baseline.md)** — end-to-end worked
  examples for every shipped solver path.
- **API reference** — per-subpackage `@docs` blocks for every
  exported type and function.
- **[Status](status.md)** — what's landed, what's stubbed, what's
  planned.
- **[References](references.md)** — canonical paper bibliography.

## Mental model

```
┌─────────────────────────────────────────────────────────────┐
│                    User-facing types (Core)                 │
│  HydroModule · Plant · Reservoir · Generator · Pump · …      │
│  LongTermHydroProblem · ShopShortTermProblem                │
│  ScenarioTree · MarkovianUncertainty · ProdRiskUncertainty  │
│  WaterValueCuts  (← the long → short bridge)                │
└──────────────────────────────┬──────────────────────────────┘
                               │  solve(problem, solver)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Solver families (Opt, dispatch)                │
│                                                             │
│  JuMPSolver              SDDPHydroSolver                    │
│   ├─ LP                   ├─ ScenarioTree                   │
│   └─ MILP                 ├─ Markovian                      │
│                           ├─ StagewiseIndependent           │
│                           └─ ProdRiskUncertainty            │
│  LagrangianHydroSolver    SLPHydroSolver                    │
│   ├─ LP path (C3)          ├─ UC mode  (planned)            │
│   └─ DP path (C3.5)        └─ ULD mode (planned)            │
└──────────────────────────────┬──────────────────────────────┘
                               │  (uses backend)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Backend abstraction (Core, vendor-neutral)          │
│  CPUBackend  ·  GPUBackend{CUDABackend|MetalBackend|…}      │
│           via KernelAbstractions.jl                         │
└─────────────────────────────────────────────────────────────┘
```

The Core/Opt split is the hard boundary. Anything solver-agnostic
(types, topology, backend abstraction, utilities) lives in
`HydroModelsCore`; anything that knows about a solver algorithm lives
in `HydroModelsOpt`. The two stub layers (`HydroModelsData` for I/O,
`HydroModelsViz` for plotting) talk to Core only.

See [Architecture](architecture.md) for the detailed boundary rules.
