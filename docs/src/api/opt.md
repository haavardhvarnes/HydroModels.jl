# `HydroModelsOpt` — API reference

The Opt layer hosts the solver hierarchy, KA kernels, and the
long-term → short-term toolchain. It also re-exports
`HydroModelsCore`.

## Solver hierarchy

```@docs
AbstractSolver
AbstractGPUSolver
```

## Solver types

```@docs
JuMPSolver
SDDPHydroSolver
SLPHydroSolver
LagrangianHydroSolver
UnitCommitmentMode
UnitLoadDispatchMode
```

## Solver-side strategy types

```@docs
ParallelismMode
SynchronousParallel
AsynchronousParallel
TotallyAsynchronous
DecompositionStrategy
ScenarioDecomposition
SpatialDecomposition
StepSizeRule
ConstantStep
DiminishingStep
PolyakStep
compute_step
```

## Output

```@docs
HydroSolution
```

The JuMP LP/MILP baseline returns a `HydroSolution{T}` with the
following Tables.jl-compatible fields:

| Field | Columns |
|---|---|
| `storage` | `(time, reservoir, S, S_violation_up, S_violation_dn)` |
| `spill` | `(time, reservoir, Q_spill)` |
| `q_gen` | `(time, generator, Q)` |
| `p_gen` | `(time, generator, P)` |
| `q_pump` | `(time, pump, Q)` |
| `e_pump` | `(time, pump, E)` |
| `q_tunnel` | `(time, tunnel, Q)` |
| `revenue` | `(time, revenue, …)` |
| `r_alloc` | `(time, unit, product, R)` |
| `r_slack` | `(time, group, product, R_slack)` |
| `water_value` | `(reservoir, cut, alpha, theta)` |
| `junctions` | `(time, junction, Q_junc, R_min_slack)` |

## LP solver factory

```@docs
default_lp_solver
is_milp
```

`default_lp_solver()` is declared as `function default_lp_solver end`
with no methods; loading `HiGHS` activates the
`HydroModelsOptHiGHSExt` extension, which adds the zero-arg method
returning `JuMPSolver(HiGHS.Optimizer)`.

## Long-term → short-term toolchain

```@docs
HydropowerToolchain
run_toolchain
with_end_value_cuts
synthetic_end_value_cuts
extract_end_value_cuts
```

## Core verb

```@docs
solve
```

## Solver extensions

Several solver and backend integrations ship as **package
extensions** that auto-load when both `HydroModelsOpt` and the
corresponding optional dependency are loaded:

| Extension | Trigger | What it adds |
|---|---|---|
| `HydroModelsOptHiGHSExt` | `using HiGHS` | `default_lp_solver() = JuMPSolver(HiGHS.Optimizer)` |
| `HydroModelsOptSDDPExt` | `using SDDP` | `solve(::LongTermHydroProblem, ::SDDPHydroSolver)` via SDDP.jl |
| `HydroModelsOptMadNLPExt` | `using MadNLP` | (planned) interior-point NLP dispatch |
| `HydroModelsOptCUDAExt` | `using CUDA` | sets default backend to `GPUBackend(CUDABackend())` |
| `HydroModelsOptAMDGPUExt` | `using AMDGPU` | sets default backend to `GPUBackend(ROCBackend())` |
| `HydroModelsOptMetalExt` | `using Metal` | sets default backend to `GPUBackend(MetalBackend())` |
| `HydroModelsOptoneAPIExt` | `using oneAPI` | sets default backend to `GPUBackend(oneAPIBackend())` |
| `HydroModelsOptAcceleratedKernelsExt` | `using AcceleratedKernels` | uses AK reductions in the Lagrangian aggregation path |

GPU extensions all wire themselves via
`HydroModelsCore.set_default_backend!`; none of them redefine
`default_backend()` directly, which is the Julia 1.12+ way to avoid
the precompile-time method-overwriting ban.
