# Lagrangian — scenario decomposition on CPU and GPU

`LagrangianHydroSolver` is the research solver path. It ships in
two flavours:

| Dispatch | Variant | Where it runs |
|---|---|---|
| `:lp` (C3 path) | per-scenario LP via JuMP + HiGHS | CPU |
| `:dp` (C3.5 path) | per-scenario Bellman DP via KA kernels | CPU or any GPU backend |
| `:auto` | DP on `GPUBackend(_)`, LP on `CPUBackend()` | default |

The DP path dualises **both** the end-of-horizon storage targets
(`λ`) and the cross-reservoir mass-balance equalities (`μ`),
reducing each `(scenario, reservoir)` subproblem to a 1-D Bellman DP
on a storage grid. The LP path dualises only `λ` and uses HiGHS for
each per-scenario subproblem.

## Setup

```julia
using HydroModelsOpt
using HiGHS

# (optional) GPU backend — installs itself via Core.set_default_backend!
using Metal           # Apple Silicon
# using CUDA          # NVIDIA
# using AMDGPU        # AMD
# using oneAPI        # Intel
```

## Build the problem

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

num_stages = 12
realizations = [[Float64[8.0, 4.0], Float64[3.0, 2.0]] for _ in 1:num_stages]
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

The Lagrangian solver requires `prob.stage_prices` to be set — the
StagewiseIndependent uncertainty carries inflows only.

## LP path (C3) — CPU only

```julia
solver = LagrangianHydroSolver(;
    backend           = CPUBackend(),
    subproblem_solver = JuMPSolver(HiGHS.Optimizer;
                                   options = Dict(:output_flag => false)),
    step_size         = DiminishingStep(1.0, 0.9),
    max_iter          = 100,
    n_scenarios       = 16,
    dispatch          = :lp,
)
sol = solve(prob, solver)
```

Returns a `Solution{T}` with:

- `sol.primal :: NamedTuple` of `(scenario_objectives, scenario_S_end)`
- `sol.dual   :: NamedTuple` of `(multipliers, targets)`
- `sol.metadata[:variant] == :LagrangianStagewiseIndependent`

The status is `MOI.OTHER_LIMIT` (vanilla subgradient does not
produce a monotone upper bound).

## DP path (C3.5) — CPU or GPU

```julia
solver = LagrangianHydroSolver(;
    # backend defaults to Metal on Apple Silicon when `using Metal`
    step_size          = DiminishingStep(1.0, 0.9),
    max_iter           = 50,
    n_scenarios        = 64,
    K_grid             = 64,                 # storage grid points per reservoir
    dual_flow_coupling = true,               # dualise mass balance too
    dispatch           = :auto,              # :auto / :lp / :dp
)
sol = solve(prob, solver)
@show sol.metadata[:variant]      # :LagrangianDP
@show sol.metadata[:dispatch]     # :dp
@show size(sol.primal.S_traj)     # (T+1, nmod, n_scen)
@show size(sol.dual.μ)            # (T,   nmod, n_scen)
```

## Float32 on Metal

Apple GPUs have no native Float64. The
`HydroModelsOptMetalExt._backend_eltype` hook demotes
`Float64 → Float32` for device buffers; the host-side outer
subgradient loop stays in Float64 and promotes back at the kernel
boundary. Discretisation error from the storage grid (`O(1/K)`)
dominates the precision loss in practice.

## When does the GPU actually beat the CPU?

The C3.5 milestone is the *correctness + portability* contract; not
every instance size is worth pushing to a GPU. From the bench script
[`HydroModelsOpt/test/bench_lagrangian_metal.jl`](https://github.com/haavardhvarnes/HydroModels.jl/blob/main/HydroModelsOpt/test/bench_lagrangian_metal.jl):

| Instance | CPU | Metal |
|---|---|---|
| 2 reservoirs × 4 stages × 16 scen × K=32 | wins | kernel-launch dominated |
| 10 reservoirs × 52 stages × 512 scen × K=64 | — | ≈ 25× faster |

The crossover is roughly `n_scen · n_reg · n_stages > 50_000` —
above that the per-thread work (K² reward evaluations per stage)
overwhelms the launch overhead.

Run the benchmark yourself with:

```bash
HYDROMODELS_RUN_BENCH=1 julia --project=HydroModelsOpt \
    HydroModelsOpt/test/bench_lagrangian_metal.jl
```

Results land in `benchmarks/c3_5_metal.csv`.

## Caveats (documented up front)

1. **Sublinear convergence.** Vanilla subgradient is `O(1/√k)`. A
   bundle method (`solver.use_bundle = true`) is the natural follow-
   up — placeholder in the struct, not yet wired.
2. **Mass balance is violated until `μ` converges.** The returned
   primal trajectory is the DP forward simulation along the optimal
   policy — not a feasible primal until the dual converges.
   Sophisticated primal recovery (re-solving one scenario LP at fixed
   `(λ, μ)`) is a C3.6 follow-up.
3. **`RegulationReservoir` only.** `_validate_dp_inputs` rejects
   problems containing `BufferReservoir` modules; pass-through nodes
   without storage need a different subproblem form. Defer to C3.6.

## Forcing CPU on a GPU box

Useful for cross-validating "DP vs LP" effects in isolation from the
spatial-coupling dualisation:

```julia
# Force the LP path on a GPU machine
solver_lp = LagrangianHydroSolver(; dispatch = :lp,
                                    backend = CPUBackend(),
                                    subproblem_solver = JuMPSolver(HiGHS.Optimizer))

# Force the DP path on CPU
solver_dp = LagrangianHydroSolver(; dispatch = :dp,
                                    backend = CPUBackend())
```

## See also

- [Architecture](../architecture.md) for the backend abstraction
  contract and the `set_default_backend!` Ref pattern.
- [Lagrangian opt CLAUDE.md](https://github.com/haavardhvarnes/HydroModels.jl/blob/main/HydroModelsOpt/CLAUDE.md)
  for the full C3.5 design note.
