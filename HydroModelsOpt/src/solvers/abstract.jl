"""
Solver abstractions and the core `solve` verb.

Concrete solvers extend `AbstractSolver` (or `AbstractGPUSolver`) and
implement `solve(::AbstractOptProblem, ::AbstractSolver)`. Dispatch on
both arguments is the central mechanism for choosing an algorithm.
"""

# ============================================================
# Abstract solver types
# ============================================================

"""
    AbstractSolver

Abstract supertype for all solvers.
"""
abstract type AbstractSolver end

"""
    AbstractGPUSolver <: AbstractSolver

Subtype for solvers that target GPU execution. Carries a `ComputeBackend`.
"""
abstract type AbstractGPUSolver <: AbstractSolver end

# ============================================================
# Step-size rules (for Lagrangian and any other iterative methods)
# ============================================================

"""
    StepSizeRule

Strategy for choosing step sizes in iterative methods (subgradient,
projected gradient, bundle, etc.).
"""
abstract type StepSizeRule end

"""
    ConstantStep{T}(α::T)

Constant step size `α` at every iteration. Convergence not guaranteed.
"""
struct ConstantStep{T} <: StepSizeRule
    α::T
end

"""
    DiminishingStep{T}(α₀::T, decay::T)

Step size `α₀ × decay^k` at iteration `k`. Choose `decay` close to 1 for
slow decay.
"""
struct DiminishingStep{T} <: StepSizeRule
    α₀::T
    decay::T
end

"""
    PolyakStep{T}(best_known::T)

Polyak step size:

    αₖ = (f(xₖ) - f*) / ‖gₖ‖²

where `f*` is a known lower bound (typically the best primal value seen).
"""
struct PolyakStep{T} <: StepSizeRule
    best_known::T
end

"""
    compute_step(rule, iter, g, current_obj)

Compute the step size at iteration `iter` given current subgradient `g`
and current dual objective `current_obj`.
"""
compute_step(r::ConstantStep, ::Int, _, _) = r.α
compute_step(r::DiminishingStep, k::Int, _, _) = r.α₀ * r.decay^(k-1)
function compute_step(r::PolyakStep, ::Int, g, current_obj)
    gnorm2 = sum(abs2, g)
    return iszero(gnorm2) ? zero(eltype(g)) :
                            (current_obj - r.best_known) / gnorm2
end

# ============================================================
# Risk measures (forward declared in types/problems.jl)
# ============================================================

# AbstractRiskMeasure, Expectation, CVaR, NestedCVaR
# are defined in types/problems.jl

# ============================================================
# Parallelism modes (used by SDDP variants)
# ============================================================

"""
    ParallelismMode

How forward and backward iterations are coordinated. From Helseth &
Braaten (2015) and follow-up work.
"""
abstract type ParallelismMode end

"""
    SynchronousParallel <: ParallelismMode

Forward and backward iterations advance in lockstep across processes:
all forward passes complete before any backward pass starts. The
baseline parallel mode in Helseth & Braaten (2015).
"""
struct SynchronousParallel <: ParallelismMode end

"""
    AsynchronousParallel <: ParallelismMode

Forward and backward processes overlap: backward passes may start as
soon as their input becomes available, without waiting for all
forward passes to complete. Faster wall-clock but the cut set is
slightly stale across processes.
"""
struct AsynchronousParallel <: ParallelismMode end

"""
    TotallyAsynchronous <: ParallelismMode

Forward and backward processes share no synchronization barriers.
Each process iterates independently against the latest available
cut set. Highest throughput, weakest convergence guarantees.
Helseth & Braaten (2015).
"""
struct TotallyAsynchronous <: ParallelismMode end

# ============================================================
# Decomposition strategies (used by Lagrangian solver)
# ============================================================

"""
    DecompositionStrategy

How a problem is decomposed for Lagrangian relaxation. The choice
determines which coupling constraints are dualized.
"""
abstract type DecompositionStrategy end

"""
    ScenarioDecomposition <: DecompositionStrategy

Dualize the per-scenario non-anticipativity (or end-of-horizon) cou-
pling. Each subproblem is one whole-path scenario; multipliers
enforce consistency across scenarios. The default for
`LagrangianHydroSolver` on `StagewiseIndependent` uncertainty.
"""
struct ScenarioDecomposition <: DecompositionStrategy end

"""
    SpatialDecomposition <: DecompositionStrategy

Dualize the per-(stage, scenario) cross-reservoir mass-balance
coupling. Each subproblem is a single `(scenario, reservoir)` 1-D
dynamic-programming problem; multipliers enforce the cascade
topology. Used by the GPU-native DP path (milestone C3.5).
"""
struct SpatialDecomposition <: DecompositionStrategy end

struct TemporalDecomposition <: DecompositionStrategy end
struct NonAnticipativityDecomposition <: DecompositionStrategy end

# ============================================================
# The core verb
# ============================================================

"""
    solve(problem, solver)

Solve `problem` with `solver`. Returns a `Solution{T}`.

Concrete solver/problem combinations are implemented in individual
files under `src/solvers/`.

Throws `MethodError` if no implementation exists for the combination.
"""
function solve end
