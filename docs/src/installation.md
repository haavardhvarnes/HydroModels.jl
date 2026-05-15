# Installation

The HydroModels.jl meta-package is published on a private Julia
registry, `haavardhvarnes/JuliaRegistry`. The General registry is
not used.

## 1. Add the private registry

Once per Julia install:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry.git"))
```

## 2. Add the package(s)

### The name collision

There is an unrelated package named `HydroModels.jl` in the **General
registry** (`chooron/HydroModels.jl` — modular rainfall-runoff
models). When both registries are visible to Pkg and you run

```julia
Pkg.add("HydroModels")     # ambiguous — errors
```

Pkg refuses to resolve because two registered `HydroModels` packages
exist with different UUIDs.

There are two ways around this.

### Option A: disambiguate by UUID (recommended)

```julia
Pkg.add(PackageSpec(name = "HydroModels",
                    uuid = "3bb2d84c-373d-45fe-b371-a5244eeae5e2"))
using HydroModels
```

The UUID `3bb2d84c-…` is the one for this package. It is stable
across versions; bookmark it.

### Option B: add a subpackage by name

The six subpackage names are unique to this registry, so plain
`Pkg.add` works:

```julia
Pkg.add("HydroModelsOpt")        # also brings in HydroModelsCore
using HydroModelsOpt             # @reexports HydroModelsCore
```

`HydroModelsOpt` is the typical entry point for the optimization
layer — it transitively brings `HydroModelsCore` (types, topology,
backend abstraction) into scope via `Reexport.@reexport`.

## Package UUIDs

| Subpackage | UUID |
|---|---|
| `HydroModels` (umbrella) | `3bb2d84c-373d-45fe-b371-a5244eeae5e2` |
| `HydroModelsCore` | `35d12705-5efd-4182-93f7-afc386ecd870` |
| `HydroModelsData` | `b83cc262-783b-4d80-bd7e-3698e9ea66d0` |
| `HydroModelsOpt` | `3ad2896b-1445-4374-ab33-3e97259a88a5` |
| `HydroModelsForecast` | `7703e5f2-0d94-41b8-bfdf-7c3b45cd84fa` |
| `HydroModelsViz` | `a798d185-2f50-4def-b29e-c2b8747cb8b1` |

## Picking the right subpackage

A minimal install pulls only what's needed:

| If you want to … | …`using` |
|---|---|
| Build a problem from a SHOP YAML | `HydroModelsData` |
| Define `HydroModule` / `ShopShortTermProblem` etc. by hand | `HydroModelsCore` |
| Solve a LP/MILP via JuMP | `HydroModelsOpt` + a JuMP optimizer (e.g. `HiGHS`) |
| Solve a stochastic long-term problem | `HydroModelsOpt` + `SDDP` (loads the SDDP.jl bridge extension) |
| Fit a PAR(1) inflow model or a Markov price chain | `HydroModelsForecast` |
| Plot the resulting 11-panel dashboard | `HydroModelsViz` + a Makie backend (`CairoMakie`, `GLMakie`, or `WGLMakie`) |
| Everything | `HydroModels` (umbrella) |

## Solver extensions

`HydroModelsOpt` declares several **weak dependencies**. Loading the
corresponding package activates the extension that wires it into the
solver dispatch:

| Load this | …to enable |
|---|---|
| `HiGHS` | `default_lp_solver()` returns a HiGHS-backed `JuMPSolver` |
| `SDDP` | `solve(::LongTermHydroProblem, ::SDDPHydroSolver)` dispatches via SDDP.jl |
| `MadNLP` | (planned — interior-point dispatch) |

## GPU extensions

`HydroModelsOpt` also declares GPU vendor weakdeps:

| Load this | …to enable |
|---|---|
| `CUDA` | `default_backend()` returns `GPUBackend(CUDABackend())` (and dispatches `LagrangianHydroSolver` to CUDA kernels) |
| `AMDGPU` | …`GPUBackend(ROCBackend())` |
| `Metal` | …`GPUBackend(MetalBackend())` — Apple Silicon |
| `oneAPI` | …`GPUBackend(oneAPIBackend())` |

Vendor extensions register themselves via
`HydroModelsCore.set_default_backend!(GPUBackend(…))` from their
`__init__`. There is no precompile-time method redefinition; the hook
is a `Ref{Union{Nothing, ComputeBackend}}` in Core, which avoids the
Julia 1.12 method-overwrite ban.

If multiple GPU backends are loaded, the last one to call
`set_default_backend!` wins. Set the backend explicitly when this
matters:

```julia
using HydroModelsOpt
using CUDA
using Metal              # both extensions load — last one wins on the Ref

# pin the backend on the solver itself:
solver = LagrangianHydroSolver(; backend = GPUBackend(CUDABackend()))
```

## Development setup (from source)

If you want to hack on the meta-package itself:

```bash
git clone https://github.com/haavardhvarnes/HydroModels.jl
cd HydroModels.jl
julia --project=. dev.jl
```

`dev.jl` runs `Pkg.develop(path = "HydroModelsCore")` and so on for
every subpackage, then `Pkg.instantiate()`. After that the umbrella's
`Project.toml` is wired to the local paths and changes to subpackage
source are picked up on `Revise.revise()` or a fresh `using`.

Run the full per-subpackage test sweep from the repo root:

```bash
julia --project=. -e 'using Pkg
for pkg in ["HydroModelsCore", "HydroModelsData", "HydroModelsOpt",
            "HydroModelsForecast", "HydroModelsViz", "HydroModels"]
    Pkg.test(pkg)
end'
```
