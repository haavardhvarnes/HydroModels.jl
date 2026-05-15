# HydroModels.jl

> **Status: early scaffold.** Type system in place; solver implementations
> are stubs. See `CLAUDE.md` for the architectural roadmap and canonical
> references.

A research-grade Julia **meta-package** for hydropower modeling that
unifies multiple solver paradigms and modeling concerns behind a
single, backend-agnostic type system.

The deliberate ambition is to let a user define a hydro problem once
and dispatch it to:

- **SDDP-family solvers** for long/medium-term stochastic scheduling
  — textbook scenario-tree SDDP, Markovian SDDP, and the Nordic combined
  SDP/SDDP of Gjelsvik–Belsnes–Haugstad (the algorithm behind SINTEF's
  ProdRisk)
- **SLP-family solvers** for short-term operational scheduling
  — the SHOP algorithm of Skjelbred, Kong, Fosso
- **Lagrangian subgradient on GPU** for decomposable problems where
  per-scenario or per-reservoir subproblems can be solved in parallel
  without linear-system factorization
- **Direct LP/MILP/NLP via JuMP** as a baseline — HiGHS, Gurobi, CPLEX,
  MadIPM, NVIDIA cuOpt

GPU-portable from day one via KernelAbstractions.jl and AcceleratedKernels.jl.
NVIDIA, AMD, Apple Metal, and Intel oneAPI backends all supported.

## Meta-package layout

`HydroModels.jl` is a monorepo of cooperating subpackages, in the
style of [Flyt.jl](https://github.com/...) and [INLA.jl](https://github.com/...):

| Subpackage | Status | What it owns |
|---|---|---|
| **`HydroModels`** | scaffold | Umbrella — re-exports `HydroModelsCore` + `HydroModelsOpt` |
| **`HydroModelsCore`** | scaffold | Physical types, problem formulations, uncertainty containers, topology, backend abstraction, generic utilities |
| **`HydroModelsOpt`** | scaffold (stubs) | SDDP / SLP / GPU Lagrangian / JuMP solvers, KA kernels, uncertainty fitting, long-term → short-term toolchain |
| **`HydroModelsData`** | placeholder | I/O readers (SHOP YAML, EMPS v10, HDF5 results) — Tables.jl outputs |
| **`HydroModelsForecast`** | placeholder | PAR(1), Markov price, scenario generation |
| **`HydroModelsViz`** | placeholder | Makie-based plotting (CairoMakie / GLMakie / WGLMakie via weakdep + extension) |

Each subpackage is its own Julia package with its own `Project.toml`,
UUID, and dependency graph.

## Development setup

From the repo root:

```julia
julia --project=. dev.jl
```

This `Pkg.develop`s each subpackage from its local path. After that,
in this environment, `using HydroModels` brings the full umbrella API
into scope; or you can `using HydroModelsCore` / `using HydroModelsOpt`
directly for a tighter dependency footprint.

## Why another framework?

Existing tools each solve part of the problem:

- **SDDP.jl** is excellent for textbook stochastic dual dynamic programming
  but doesn't capture the ProdRisk hybrid SDP/SDDP approach used in the
  Nordics, nor does it expose enough structure for GPU-Lagrangian variants.
- **SINTEF's ProdRisk and SHOP** are production tools encoding 40+ years
  of operational wisdom, but they're closed-source binaries with limited
  research extensibility.
- **SINTEF ReSDDP** is an open SDDP for hydropower (GPLv3) — we reference
  its design choices but do not depend on it.
- **MadIPM, cuOpt, cuPDLP** are powerful GPU solvers but operate at the
  raw LP level — they don't know about reservoirs, water values, or the
  long-term/short-term toolchain integration.

`HydroModels.jl` is the glue layer: a typed, composable description of
hydropower problems with pluggable solver backends. It wraps and dispatches
to the above where appropriate; it implements custom algorithms (Nordic
SDP/SDDP, GPU Lagrangian) where the existing tools don't fit.

## Quick start

```julia
using HydroModels

upper = HydroModule{Float64,RegulationReservoir}(
    name = :upper, max_vol = 200.0, initial_vol = 200.0,
    rated_discharge = 70.0, energy_factor = 1.1,
    discharge_to = :lower,
)
lower = HydroModule{Float64,RegulationReservoir}(
    name = :lower, max_vol = 200.0, initial_vol = 200.0,
    rated_discharge = 70.0, energy_factor = 1.0,
)

topology = build_topology([upper, lower])
# ... (see examples/01_two_reservoir_cascade.jl)
```

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User-facing types (Core)                 │
│  HydroModule  ·  WaterTopology  ·  UncertaintyModel         │
│  LongTermHydroProblem  ·  ShortTermHydroProblem             │
│  WaterValueCuts  (← the long→short bridge)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │  solve(problem, solver)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Solver families (Opt, dispatch)                │
│                                                             │
│  SDDPHydroSolver         SLPHydroSolver                     │
│   ├─ scenario tree         ├─ UC mode (MILP)                │
│   ├─ Markovian             └─ ULD mode (LP)                 │
│   └─ ProdRisk SDP/SDDP                                      │
│                                                             │
│  LagrangianHydroSolver   JuMPSolver                         │
│   ├─ scenario decomp       (any MOI optimizer)              │
│   ├─ spatial decomp                                         │
│   └─ bundle / primal recovery                               │
└──────────────────────────────┬──────────────────────────────┘
                               │  (uses backend)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Backend abstraction (Core, vendor-neutral)          │
│  CPUBackend  ·  GPUBackend{CUDABackend|ROCBackend|...}      │
│           via KernelAbstractions.jl + AcceleratedKernels    │
└─────────────────────────────────────────────────────────────┘
```

## Repository layout

```
HydroModels/                            # repo root (monorepo)
├── Project.toml                        # monorepo manifest (develops all subpackages)
├── dev.jl                              # Pkg.develop bootstrap
├── README.md
├── CLAUDE.md                           # read this first
├── docs/                               # Documenter.jl site (covers all subpackages)
├── examples/
├── papers/
│
├── HydroModels/                        # umbrella subpackage
├── HydroModelsCore/                    # types + topology + utils + backend abstraction
├── HydroModelsOpt/                     # solvers + kernels + uncertainty fitting + toolchain
│   └── ext/                            # 8 package extensions for GPU vendors and solvers
├── HydroModelsData/                    # placeholder — I/O readers (SHOP YAML, EMPS, HDF5)
├── HydroModelsForecast/                # placeholder — PAR(1), Markov price, scenario gen
└── HydroModelsViz/                     # placeholder — Makie-based plotting
```

## Canonical references

The architecture is anchored in a small set of papers; see `CLAUDE.md`
for the full list. The most important:

- **Gjelsvik, Mo, Haugstad 2010**, Springer Handbook of Power Systems I,
  ch. 2 — the canonical Nordic SDDP review
- **Gjelsvik, Belsnes, Haugstad 1999**, PSCC — the combined SDP/SDDP
  algorithm
- **Helseth & Braaten 2015**, Energies — parallelization of SDDP
- **Skjelbred, Kong, Fosso 2019**, IJEPES — SHOP's dynamic MILP
  linearization

## Status of solver implementations

| Solver | Problem class | Status |
|--------|---------------|--------|
| `JuMPSolver` | `ShortTermHydroProblem` (deterministic) | stub |
| `SDDPHydroSolver` + `ScenarioTree` | `LongTermHydroProblem` | stub (delegate to SDDP.jl) |
| `SDDPHydroSolver` + `MarkovianUncertainty` | `LongTermHydroProblem` | stub (delegate to SDDP.jl) |
| `SDDPHydroSolver` + `ProdRiskUncertainty` | `LongTermHydroProblem` | stub (native impl needed) |
| `SLPHydroSolver` | `ShortTermHydroProblem` | stub |
| `LagrangianHydroSolver` | `LongTermHydroProblem` | stub |
| `LagrangianHydroSolver` | `StochasticShortTermProblem` | stub |

## Code style

All Julia code follows the [SciML Style Guide](https://github.com/SciML/SciMLStyle).
Formatting is enforced via [Runic.jl](https://github.com/fredrikekre/Runic.jl).

## License

TBD (likely MIT to match the surrounding Julia ecosystem).
