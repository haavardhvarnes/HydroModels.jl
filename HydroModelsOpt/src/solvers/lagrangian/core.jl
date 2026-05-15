"""
Lagrangian subgradient solver — the GPU-native research direction.

Decomposes the problem by dualizing coupling constraints, then solves
the resulting independent subproblems in parallel on the GPU. The subgra-
dient update is element-wise and the aggregation is a reduction, so the
whole iteration maps cleanly to GPU primitives without any sparse linear
solves.

References:
- Lagrangian relaxation classics (Fisher 1985; Bertsekas 1999)
- GPU parallel LR: massively-parallel Lagrangian relaxation literature
- Helseth & Braaten (2015) on hydro scheduling parallelization
"""

"""
    LagrangianHydroSolver(; backend, max_iter, step_size, decomposition, ...)

GPU-portable Lagrangian subgradient solver. The `backend` field selects
the compute substrate; the `decomposition` field selects which coupling
constraints to dualize.
"""
Base.@kwdef struct LagrangianHydroSolver{B<:ComputeBackend, S} <: AbstractGPUSolver
    backend::B                            = default_backend()
    subproblem_solver::S                  = nothing
    max_iter::Int                         = 1000
    tol::Float64                          = 1.0e-4
    step_size::StepSizeRule
    decomposition::DecompositionStrategy  = ScenarioDecomposition()
    use_bundle::Bool                      = false
    primal_recovery::Symbol               = :round_and_repair  # placeholder
    n_scenarios::Int                      = 16
    # ----- C3.5 additions -----------------------------------------
    # `K_grid` is the number of discretised storage levels per
    # regulation reservoir in the Bellman DP subproblem. `K = 64` keeps
    # device memory under 200 MB for typical instances. See the C3.5
    # plan for the memory-vs-accuracy tradeoff.
    K_grid::Int                           = 64
    # `dual_flow_coupling = true` dualises the cross-reservoir mass-
    # balance equality with multipliers μ_{s,i,t}, decoupling each
    # (scenario, reservoir) into an independent 1-D DP. Set to `false`
    # to reduce to the C3 storage-only dualisation (used for isolating
    # "DP vs LP" effects from the spatial-coupling dualisation).
    dual_flow_coupling::Bool              = true
    # `dispatch` picks the inner solver. `:auto` resolves to `:dp` on
    # any `GPUBackend`, `:lp` on `CPUBackend()`. Explicit `:dp` forces
    # the GPU-native path on CPU (for cross-validation); explicit
    # `:lp` forces the JuMP / HiGHS path on a GPU backend.
    dispatch::Symbol                      = :auto
end

# C3 lands the StagewiseIndependent scenario-decomposition LP dispatch
# in `solvers/lagrangian/scenario_decomposition.jl`. C3.5 adds the
# KA-native Bellman-DP path in `solvers/lagrangian/dp_decomposition.jl`.
# Other problem types stay as informative-error stubs until C3.5+.

"""
    _resolve_dispatch(solver)

Resolve `solver.dispatch ∈ {:auto, :lp, :dp}` to a concrete symbol
based on the backend. `:auto` → `:dp` on any `GPUBackend`, `:lp` on
`CPUBackend`. Any other symbol is an `ArgumentError`.
"""
function _resolve_dispatch(solver::LagrangianHydroSolver)
    d = solver.dispatch
    if d === :auto
        return solver.backend isa GPUBackend ? :dp : :lp
    elseif d === :dp || d === :lp
        return d
    else
        throw(ArgumentError(
            "LagrangianHydroSolver.dispatch must be :auto, :lp, or :dp; " *
            "got $(repr(d))."))
    end
end
