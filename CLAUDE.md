# HydroModels.jl — Working Notes for Claude

This file is read at the start of every Claude Code session. It encodes the
*why* behind decisions, the canonical references, and the conventions to
follow. **When in doubt, prefer doing nothing over guessing — ask first.**

## What this package is

`HydroModels.jl` is a research-grade Julia **meta-package** for hydropower
modeling that unifies multiple solver paradigms behind a single,
backend-agnostic type system. The repo is a monorepo of six cooperating
subpackages (umbrella `HydroModels` + `HydroModelsCore` + `HydroModelsOpt`
+ stubs `HydroModelsData` / `HydroModelsForecast` / `HydroModelsViz`),
modeled on the sibling `Flyt.jl` scaffold. The deliberate ambition is to
let a user define a hydro problem once and dispatch it to:

- **SDDP-family solvers** for long/medium-term stochastic scheduling
  (textbook scenario-tree SDDP, Markovian SDDP, and the Nordic combined
  SDP/SDDP of Gjelsvik–Belsnes–Haugstad)
- **SLP-family solvers** for short-term operational scheduling
  (the SHOP algorithm of Skjelbred, Kong, Fosso, et al.)
- **Lagrangian subgradient on GPU** for decomposable problems where
  per-scenario or per-reservoir subproblems can be solved in parallel
  without linear-system factorization
- **Direct LP/MILP/NLP solvers via JuMP** as a baseline (HiGHS, Gurobi,
  CPLEX, MadIPM, cuOpt)

The framework is **GPU-portable from day one** via KernelAbstractions.jl
and AcceleratedKernels.jl. Vendor-specific code (CUDA.jl, AMDGPU.jl,
Metal.jl, oneAPI.jl) lives behind a backend abstraction in
`HydroModelsCore/src/backends/`. Do not hardcode CUDA anywhere outside
the four GPU package extensions at `HydroModelsOpt/ext/HydroModelsOpt{CUDA,AMDGPU,Metal,oneAPI}Ext.jl`.

## What this package is NOT

- Not a re-implementation of SDDP.jl, MadNLP/MadIPM, or cuOpt. We
  *wrap* and *dispatch to* these where appropriate.
- Not a production replacement for ProdRisk or SHOP. SINTEF's tools
  encode 40+ years of operational hardening we cannot match.
- Not a general-purpose optimization framework. The type system is
  shaped by hydropower problems specifically. Adding unrelated
  features (e.g., portfolio optimization, ML training) requires
  explicit discussion first.

## Canonical references (read these before deep changes)

The architecture follows a small set of canonical papers. When making
non-trivial design decisions, the answer is usually in one of these:

### Long/medium-term (SDDP family)
- **Pereira & Pinto 1991** (Math. Prog.) — the original SDDP method
- **Gjelsvik, Mo, Haugstad 2010** (Springer Handbook of Power Systems I,
  ch. 2) — the canonical Nordic SDDP review. *The* reference for ProdRisk
  algorithmic choices.
- **Gjelsvik, Belsnes, Haugstad 1999** (PSCC) — the combined SDP/SDDP
  algorithm for exogenous stochastic price. This is what ProdRisk
  actually implements.
- **Helseth & Braaten 2015** (Energies, open access) — parallelization
  of SDDP for hydro. Source for our `ParallelismMode` types
  (synchronous / asynchronous / totally asynchronous).
- **Helseth, Mo, Hågenvik, Schäffer 2022** (J.WRPM) — state-dependent
  discharge constraints in SDDP.

### Short-term (SHOP family)
- **Skjelbred, Kong, Fosso 2019** (IJEPES, open access) — dynamic MILP
  linearization for SHOP. Source for our `SLPHydroSolver` design.
- **Belsnes, Wolfgang, Follestad, Aasgård 2016** (EPSR) — stochastic
  successive linear programming for short-term hydropower.
- **Skjelbred PhD thesis (NTNU 2019)** — comprehensive modern SHOP
  formulation. Get from NTNU Open.
- **Kong, Skjelbred, Fosso 2020** (EPSR) — overview of unit-based STHS
  formulations.

### GPU optimization
- **cuPDLP.jl paper** (Lu, Yang et al.) — first-order GPU LP solver
- **cuPDLPx / cuPDLP+** — improved variants
- **MadIPM.jl** — interior-point on GPU via cuDSS
- **NVIDIA cuOpt** — production GPU solver (LP/MIP/VRP)

### SINTEF official documentation (treat as ground truth for operational conventions)
- `docs.prodrisk.sintef.energy` — ProdRisk algorithm, modules, price
  model, inflow model
- `docs.shop.sintef.energy` — SHOP topology, markets, attributes
- `madsuite.org` — MadNLP/MadIPM ecosystem

## Architectural principles

These are non-negotiable. If a proposed change violates one, stop and
discuss first.

### 1. Separation of problem and solver
A `HydroModule` knows nothing about SDDP or SLP. A `SDDPHydroSolver`
knows nothing about Nordic vs Brazilian inflow models. The bridge is
multiple dispatch on `solve(::Problem, ::Solver)`.

### 2. Backend-agnostic computation
All array operations go through `KernelAbstractions.jl` and
`AcceleratedKernels.jl`. The only place CUDA/AMDGPU/Metal-specific
code is allowed is `src/backends/<vendor>/`. Public API uses
`AbstractArray{T}` and dispatches on the backend trait.

### 3. Mirror Nordic operational practice in naming
The user-facing API uses ProdRisk/SHOP terminology because that's
what Nordic hydropower engineers actually say:
- `HydroModule` not `Block` or `Node`
- `RegulationReservoir` vs `BufferReservoir`
- `WaterValueCuts` not `BendersApproximation`
- `LoadBlock` for within-stage price periods
- `module` topology references via `discharge_to`, `bypass_to`,
  `spill_to`

The internal solver code can use mathematical terminology freely.

### 4. Soft constraints everywhere
Per ProdRisk convention, every operational constraint is violation-
with-penalty. Hard constraints are rare and explicit. This makes
infeasibility recovery natural and avoids LP-solver gymnastics.

### 5. The two application classes are distinct types
Following Gjelsvik/Mo/Haugstad 2010, there are *two* fundamentally
different problem classes:
- `LongTermHydroProblem` — single producer, exogenous price (ProdRisk)
- `FundamentalMarketProblem` — multi-area, endogenous price (EMPS)

Most users only need the first. Don't conflate them.

### 6. Forward and backward can use different uncertainty
This is the ProdRisk innovation that textbook SDDP misses: the forward
simulation can use historical scenarios while the backward recursion
uses a fitted statistical model (PAR(1) + discretized Markov nodes).
The `ProdRiskUncertainty` type encodes this explicitly.

## Domain knowledge essentials

If you find yourself making a domain choice without these facts at
hand, you're guessing. Stop and re-read.

### Time scales (Nordic convention)
- **Long-term**: 3–5 years, weekly stages, stochastic inflow + price
- **Medium-term**: 1–2 years, weekly stages, often same model as long-term
- **Short-term**: 1–2 weeks, hourly resolution, deterministic with rolling
  horizon, water values from long-term

### The water value chain (the toolchain integration point)
Long-term scheduling produces *Benders cuts* parameterized by
(stage, price_node). Short-term scheduling consumes these cuts as
its end-value description. Get this interface right and the rest
follows.

### Hydropower production function (HPF)
The relationship P = f(Q, h) between power output, discharge, and head
is nonlinear and nonconvex. Three approaches in increasing fidelity:
1. **Constant energy factor** (η × Q, used in coarse long-term models)
2. **PQ curve with reference head** (PWL, used in ProdRisk)
3. **Dynamic PWL with head iteration** (Skjelbred & Kong 2019, used
   in SHOP)

### Module / reservoir distinction
A `module` is a logical unit (reservoir + plant + connections). A
`reservoir` is just the storage. Plants without storage (river
intakes, run-of-river) are still modules with `min_vol = max_vol = 0`.

### Markets (SHOP supports much richer than ProdRisk)
SHOP models day-ahead, FCR-N/D (up/down), FRR (up/down), and RR
(up/down) markets. ProdRisk typically only models day-ahead plus
optionally reserves via post-processing. Reserve capacity affects
both the production schedule and the dual prices.

## Project conventions

### Code style
- Follow the **SciML style guide**
  (https://github.com/SciML/SciMLStyle). Use Runic.jl for formatting.
- Type parameters carry units when possible (use Unitful.jl for
  user-facing quantities; strip units before solver calls).
- All public types and functions have docstrings with at least one
  reference to a canonical paper.
- Prefer parametric types `Struct{T, A<:AbstractArray{T}}` over
  concrete `Float64`/`Array` choices. This is what makes GPU support
  possible.

### Meta-package layout

The repo is a Flyt-style monorepo. Six subpackages live at the root,
each with its own `Project.toml` and UUID. Run `julia --project=. dev.jl`
from the repo root to `Pkg.develop` all of them.

```
HydroModels/                            # repo root (monorepo)
├── Project.toml                        # monorepo manifest
├── dev.jl                              # Pkg.develop bootstrap
├── docs/                               # Documenter site (all subpackages)
├── examples/
├── papers/
│
├── HydroModels/                        # umbrella — @reexport Core + Opt
│   └── src/HydroModels.jl
│
├── HydroModelsCore/                    # types + topology + utils + backend abstraction
│   └── src/
│       ├── HydroModelsCore.jl
│       ├── types/                      # core type definitions
│       │   ├── physical.jl             # HydroModule, TurbineUnit, etc.
│       │   ├── uncertainty.jl          # ScenarioTree, MarkovianUncertainty, ProdRiskUncertainty
│       │   ├── problems.jl             # LongTermHydroProblem, ShortTermHydroProblem, etc.
│       │   └── solutions.jl            # Solution, WaterValueCuts, BendersCut
│       ├── problems/
│       │   ├── builders.jl
│       │   └── topology.jl
│       ├── backends/
│       │   └── abstract.jl             # ComputeBackend, CPUBackend, GPUBackend
│       └── utils/
│           ├── pwl.jl
│           └── time.jl
│
├── HydroModelsOpt/                     # solvers + kernels + uncertainty fitting + toolchain
│   ├── src/
│   │   ├── HydroModelsOpt.jl           # @reexport using HydroModelsCore + solver includes
│   │   ├── uncertainty/                # uncertainty model construction & sampling
│   │   │   ├── par_model.jl
│   │   │   ├── markov_price.jl
│   │   │   └── scenario_generation.jl
│   │   ├── kernels/                    # KernelAbstractions kernels
│   │   │   ├── subgradient.jl
│   │   │   ├── bellman.jl
│   │   │   └── scenario_batch.jl
│   │   ├── solvers/
│   │   │   ├── abstract.jl             # AbstractSolver, StepSizeRule, etc.
│   │   │   ├── jump_backend.jl
│   │   │   ├── sddp/
│   │   │   ├── slp/
│   │   │   └── lagrangian/
│   │   └── toolchain/
│   │       └── longterm_to_shortterm.jl
│   └── ext/                            # package extensions
│       ├── HydroModelsOptCUDAExt.jl
│       ├── HydroModelsOptAMDGPUExt.jl
│       ├── HydroModelsOptMetalExt.jl
│       ├── HydroModelsOptoneAPIExt.jl
│       ├── HydroModelsOptAcceleratedKernelsExt.jl
│       ├── HydroModelsOptHiGHSExt.jl
│       ├── HydroModelsOptMadNLPExt.jl
│       └── HydroModelsOptSDDPExt.jl
│
├── HydroModelsData/                    # placeholder — I/O readers (SHOP YAML, EMPS, HDF5)
├── HydroModelsForecast/                # placeholder — PAR(1), Markov price, scenario gen
└── HydroModelsViz/                     # placeholder — Makie-based plotting
```

Boundary rule: solver-agnostic code (types, topology, backend abstraction,
utilities) lives in `HydroModelsCore`. Anything that knows about a solver
algorithm lives in `HydroModelsOpt`. `HydroModelsOpt` re-exports
`HydroModelsCore` via `Reexport.@reexport`, so `using HydroModelsOpt` is
the typical entry point for a full optimization workflow. `using
HydroModels` re-exports both layers.

### Testing
- Every public function needs at least one test
- Tests live in each subpackage's own `test/` directory, mirroring its
  `src/` layout. Run with `julia --project=HydroModelsCore -e 'using
  Pkg; Pkg.test()'` (and likewise for the other subpackages).
- Use `Test.jl` (stdlib), and `TestItems.jl` for finer-grained tests
- For solver tests, use small canonical instances from the literature
  (e.g., the 2-reservoir hydro valley from the SDDP.jl examples)

### Dependencies — be conservative
Each subpackage should compile fast and have few heavy dependencies.
The split:
- **`HydroModelsCore` required**: KernelAbstractions, GPUArrays,
  SparseArrays, LinearAlgebra, Random, Statistics.
- **`HydroModelsOpt` required**: HydroModelsCore, Reexport, JuMP,
  MathOptInterface, plus the same Core stdlibs and KA/GPUArrays.
- **`HydroModelsOpt` weak deps (extensions)**: CUDA, AMDGPU, Metal,
  oneAPI, AcceleratedKernels, SDDP.jl, MadNLP, HiGHS.
- **Umbrella `HydroModels`**: Reexport, HydroModelsCore, HydroModelsOpt.
- **Stubs (`Data` / `Forecast` / `Viz`)**: only HydroModelsCore for now.
  When content arrives: `Tables.jl` for I/O (not DataFrames as a hard
  dep); `Makie` + CairoMakie/GLMakie/WGLMakie weakdeps for viz (not
  PlotlyJS, not Plots).
- **Avoid pulling in**: heavy plotting libraries as hard deps, ML
  frameworks, web frameworks. Each subpackage is a library, not an
  application.

### Performance
- Profile before optimizing. Use BenchmarkTools.jl.
- Allocations in inner solver loops are forbidden. Use views and
  pre-allocated buffers.
- For GPU kernels, measure occupancy and bandwidth, not just wall
  time.
- Sparse matrix operations dominate runtime for SDDP-style solvers.
  Use NVIDIA cuSPARSE / rocSPARSE via the backend when on GPU.

## Working with the user (Haavard)

Some context that will save you time:

- **Domain expertise**: Haavard works deeply in Nordic power markets,
  has built hydrological models (HBVude.jl), and knows the operational
  reality of hydropower scheduling first-hand. Treat him as a domain
  expert. Don't over-explain hydropower fundamentals.
- **Julia expertise**: Strong. Familiar with the SciML ecosystem,
  JuMP, automatic differentiation issues, and GPU compute. You can
  use Julia idioms freely.
- **Code preferences**: Multiple dispatch over inheritance. Parametric
  types. Composable small functions. Documentation that links to
  papers.
- **Be direct**: Haavard prefers concise explanations and is happy to
  push back. Don't pad responses. Skip the apologies. If something
  looks wrong in his approach, say so with reasoning.

## Common pitfalls to avoid

These are mistakes Claude (or anyone) is likely to make in this
project. Verify before committing.

1. **Hardcoding CUDA**. Even when targeting NVIDIA exclusively, write
   it through KernelAbstractions. The portability is the point.

2. **Treating ProdRisk-style uncertainty as a special case of scenario
   trees**. It's not. The forward simulation uses historical data
   directly. Conflating these breaks the cut interpretation.

3. **Forgetting that all ProdRisk constraints are soft**. Adding hard
   constraints to subproblems makes them infeasible in edge cases and
   destroys the cut interpretation.

4. **Mixing up the two reservoir types**. `RegulationReservoir`
   contributes a state variable and gets cuts. `BufferReservoir`
   follows guidelines and does *not* get cuts. Mixing them in cut
   computations is a bug.

5. **Reinventing SDDP.jl**. For standard scenario-tree SDDP, dispatch
   to SDDP.jl as a backend. Only implement custom variants for the
   Nordic combined SDP/SDDP and the GPU Lagrangian.

6. **Premature MILP**. Most ProdRisk-class problems are pure LPs with
   piecewise-linear approximations. Integer variables enter at SHOP-
   scale (unit commitment). Don't add binary variables to the long-
   term problem unless explicitly modeling unit-level details.

7. **Ignoring the load-block / price-period distinction**. Within
   each weekly stage there are sub-periods (load blocks) with
   different prices. The cut applies to the *end of stage* reservoir
   state, but the LP solved for the stage includes the within-stage
   variation. This is in ProdRisk docs under "Time resolution".

## Definitions of done

A feature is "done" when:
- [ ] Tests pass on CPU
- [ ] Tests pass on at least one GPU backend (CUDA or AMDGPU)
- [ ] Public types/functions have docstrings with paper references
- [ ] An example in `examples/` demonstrates the feature
- [ ] If the feature touches solver dispatch, the type system rules
      (separation, backend-agnostic, naming) are upheld

## When you're stuck

In order of preference:
1. Re-read the relevant canonical paper from the list above
2. Check the ProdRisk or SHOP documentation
3. Look at SDDP.jl's source code for the standard SDDP case
4. Ask Haavard. He's faster than guessing.

## Things to discuss before changing

- The type hierarchy (`AbstractOptProblem` and its descendants)
- The backend abstraction layer
- Anything in `HydroModelsCore/src/types/`
- The Core/Opt boundary (what counts as solver-agnostic vs solver code)
- Adding a new dependency to any subpackage
- Adding a new solver family (not just a new solver in an existing family)
- Promoting one of the stub subpackages (`HydroModelsData`,
  `HydroModelsForecast`, `HydroModelsViz`) to active content

## Things you can change freely

- Solver implementations (provided dispatch types stay stable)
- Kernel implementations
- Examples
- Tests
- Documentation
- Internal utilities
