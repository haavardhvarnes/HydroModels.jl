# HydroModels.jl

A research-grade Julia **meta-package** for hydropower modelling that
unifies multiple solver paradigms behind a single, backend-agnostic
type system. The deliberate ambition is to let a user define a hydro
problem once and dispatch it to:

- **SDDP-family solvers** for long/medium-term stochastic scheduling
  — textbook scenario-tree SDDP, Markovian SDDP, and the Nordic
  combined SDP/SDDP of Gjelsvik–Belsnes–Haugstad (the algorithm
  behind SINTEF's ProdRisk).
- **SLP-family solvers** for short-term operational scheduling
  — the SHOP algorithm of Skjelbred, Kong, Fosso.
- **Lagrangian subgradient on GPU** for decomposable problems where
  per-scenario or per-reservoir subproblems can be solved in parallel
  without linear-system factorization.
- **Direct LP / MILP / NLP via JuMP** as a baseline — HiGHS, Gurobi,
  CPLEX, MadIPM, NVIDIA cuOpt.

GPU-portable from day one via KernelAbstractions.jl. NVIDIA, AMD,
Apple Metal, and Intel oneAPI backends all supported through
package extensions.

## Status

| Layer | Status |
|---|---|
| **`HydroModelsCore`** (types, topology, backend abstraction) | published, v0.1.0 |
| **`HydroModelsData`** (SHOP YAML reader + Tables.jl outputs) | published, v0.1.0 |
| **`HydroModelsOpt`** (JuMP LP/MILP baseline, SDDP.jl bridge, Lagrangian scaffold) | published, v0.1.0 |
| **`HydroModelsForecast`** (PAR(1), Markov price, scenario generation) | published, v0.1.0 |
| **`HydroModelsViz`** (Makie dashboards) | published, v0.1.0 |
| **`HydroModels`** (umbrella) | published, v0.1.0 |

The full status of each solver dispatch is tracked in
[`docs/src/status.md`](docs/src/status.md).

## Installation

The packages live on a private registry. Add it once:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry.git"))
```

### A name collision with `chooron/HydroModels.jl`

The General registry already hosts an unrelated package named
`HydroModels.jl` (`chooron/HydroModels.jl`, modular rainfall-runoff
models). When both registries are visible to Pkg, plain
`Pkg.add("HydroModels")` is ambiguous and errors. **Always
disambiguate by UUID**:

```julia
Pkg.add(PackageSpec(name = "HydroModels",
                    uuid = "3bb2d84c-373d-45fe-b371-a5244eeae5e2"))
using HydroModels
```

Equivalently, add a subpackage directly — those names are unique:

```julia
Pkg.add("HydroModelsOpt")        # imports HydroModelsCore transitively
using HydroModelsOpt
```

| Subpackage | UUID |
|---|---|
| `HydroModels` (umbrella) | `3bb2d84c-373d-45fe-b371-a5244eeae5e2` |
| `HydroModelsCore` | `35d12705-5efd-4182-93f7-afc386ecd870` |
| `HydroModelsData` | `b83cc262-783b-4d80-bd7e-3698e9ea66d0` |
| `HydroModelsOpt` | `3ad2896b-1445-4374-ab33-3e97259a88a5` |
| `HydroModelsForecast` | `7703e5f2-0d94-41b8-bfdf-7c3b45cd84fa` |
| `HydroModelsViz` | `a798d185-2f50-4def-b29e-c2b8747cb8b1` |

## Meta-package layout

`HydroModels.jl` is a Flyt-style monorepo of cooperating subpackages.
Each one is its own Julia package with its own `Project.toml`, UUID,
and dependency graph.

| Subpackage | What it owns |
|---|---|
| **`HydroModels`** | Umbrella — `@reexport` of `HydroModelsCore` + `HydroModelsOpt`. |
| **`HydroModelsCore`** | Physical types (ProdRisk-style + SHOP-style), problem formulations, uncertainty containers, topology builders, solution / cut representations, backend abstraction. |
| **`HydroModelsData`** | SHOP YAML reader, Tables.jl-conforming output tables. |
| **`HydroModelsOpt`** | JuMP LP/MILP baseline (incl. reserves + MILP unit commitment), SDDP.jl bridge, Lagrangian scaffold (CPU LP path + GPU-native Bellman DP path), long-term → short-term toolchain. |
| **`HydroModelsForecast`** | PAR(1) inflow model, Markov price discretization, scenario generation. |
| **`HydroModelsViz`** | Makie-based 11-panel dashboard, plus per-panel `plot_*!(ax, sol; …)` helpers. CairoMakie / GLMakie / WGLMakie via weakdep + extension. |

## Quick start

### LP baseline against a SHOP YAML

```julia
using HydroModelsOpt, HydroModelsData, HiGHS

parsed = read_shop_yaml(joinpath(dirname(pathof(HydroModelsData)),
                                 "..", "test", "data", "minimal.yaml"))
prob   = ShopShortTermProblem(parsed)
sol    = solve(prob, JuMPSolver(HiGHS.Optimizer))

@show sol.objective
@show keys(sol.storage)            # ::NamedTuple — Tables.jl-compatible
```

### Long-term → short-term toolchain (synthetic cuts)

```julia
using HydroModelsOpt

tc  = HydropowerToolchain{Float64}(
    longterm_model     = long_problem,
    shortterm_template = short_template,
    longterm_solver    = SDDPHydroSolver(; subproblem_solver = JuMPSolver(HiGHS.Optimizer)),
    shortterm_solver   = JuMPSolver(HiGHS.Optimizer),
    reservoir_map      = Dict(:upper => "upper", :lower => "lower"),
)
cuts = synthetic_end_value_cuts(["upper", "lower"],
                                [50_000.0, 30_000.0, 70_000.0],
                                [1.0e6 0.5e6 1.5e6; 0.5e6 0.25e6 0.75e6])
res  = run_toolchain(tc; cuts = cuts)
```

### Dashboard

```julia
using HydroModelsViz, CairoMakie

fig = plot_dashboard(sol)        # 4×3 panel Figure
save("dashboard.png", fig)
```

## Documentation

A full Documenter.jl site (architecture, per-subpackage API reference,
worked examples, canonical-paper bibliography) lives under [`docs/`](docs/).
Build locally with:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop([
    PackageSpec(path="HydroModelsCore"),
    PackageSpec(path="HydroModelsData"),
    PackageSpec(path="HydroModelsOpt"),
    PackageSpec(path="HydroModelsForecast"),
    PackageSpec(path="HydroModelsViz"),
    PackageSpec(path="HydroModels"),
])'
julia --project=docs docs/make.jl
```

The HTML output ends up in `docs/build/`. Hosting on GitHub Pages is
deliberately deferred until CI lands.

## Development setup

Clone the repo and bootstrap a development environment:

```bash
git clone https://github.com/haavardhvarnes/HydroModels.jl
cd HydroModels.jl
julia --project=. dev.jl
```

`dev.jl` runs `Pkg.develop(path = "HydroModelsCore")` etc. for every
subpackage, then `Pkg.instantiate()`. After that the umbrella's
`Project.toml` is wired to the local paths and `using HydroModels`
loads the umbrella with full live editing.

Run the full test sweep from the repo root:

```bash
julia --project=. -e 'using Pkg
for pkg in ["HydroModelsCore", "HydroModelsData", "HydroModelsOpt",
            "HydroModelsForecast", "HydroModelsViz", "HydroModels"]
    Pkg.test(pkg)
end'
```

(CI configuration is a deliberate follow-up — local tests are
green on macOS + Julia 1.11/1.12.)

## Why another framework?

Existing tools each solve part of the problem:

- **SDDP.jl** is excellent for textbook stochastic dual dynamic
  programming but does not capture the ProdRisk hybrid SDP/SDDP
  approach used in the Nordics, nor does it expose enough structure
  for GPU-Lagrangian variants.
- **SINTEF's ProdRisk and SHOP** are production tools encoding 40+
  years of operational wisdom, but they are closed-source binaries
  with limited research extensibility.
- **SINTEF ReSDDP** is an open SDDP for hydropower (GPLv3) — we
  reference its design choices but do not depend on it.
- **MadIPM, cuOpt, cuPDLP** are powerful GPU solvers but operate at
  the raw LP level — they do not know about reservoirs, water
  values, or the long-term / short-term toolchain integration.

`HydroModels.jl` is the glue layer: a typed, composable description
of hydropower problems with pluggable solver backends. It wraps and
dispatches to the above where appropriate; it implements custom
algorithms (Nordic SDP/SDDP, GPU Lagrangian) where the existing tools
do not fit.

## Canonical references

The architecture is anchored in a small set of papers; see
[`CLAUDE.md`](CLAUDE.md) for the full list and
[`docs/src/references.md`](docs/src/references.md) for paper
abstracts. The most important:

- **Pereira & Pinto 1991** (Math. Prog.) — original SDDP method.
- **Gjelsvik, Mo, Haugstad 2010** (Springer Handbook of Power
  Systems I, ch. 2) — canonical Nordic SDDP review.
- **Gjelsvik, Belsnes, Haugstad 1999** (PSCC) — combined SDP/SDDP
  algorithm with exogenous stochastic price.
- **Helseth & Braaten 2015** (Energies, open access) —
  parallelization of SDDP for hydro.
- **Skjelbred, Kong, Fosso 2019** (IJEPES, open access) — dynamic
  MILP linearization for SHOP.

## Code style

All Julia code follows the
[SciML Style Guide](https://github.com/SciML/SciMLStyle). Formatting
is enforced via [Runic.jl](https://github.com/fredrikekre/Runic.jl).

## License

MIT.
