"""
Successive Linear Programming solver — the SHOP algorithm.

Reference: Skjelbred, Kong, Fosso (2019) "Dynamic incorporation of
nonlinearity into MILP formulation for short-term hydro scheduling",
IJEPES.

SHOP iterates LP relaxations linearized around the current operating
point. The nonlinearity comes from the hydropower production function
(head-dependent), and breakpoints of the piecewise-linear approximation
are dynamically adjusted between iterations.
"""

"""
    SLPHydroSolver(; mode_sequence, inner_solver, ...)

Successive Linear Programming solver for short-term scheduling. Follows
SHOP's two-mode architecture: Unit Commitment (UC) mode resolves binary
on/off decisions; Unit Load Dispatch (ULD) mode fixes the binaries and
solves a refined LP.
"""
Base.@kwdef struct SLPHydroSolver{S<:AbstractSolver} <: AbstractSolver
    max_outer_iterations::Int     = 10
    head_tolerance::Float64       = 0.01    # in meters
    inner_solver::S
    mode_sequence::Vector{Symbol} = [:UC, :ULD]
    use_dynamic_breakpoints::Bool = true
    enforce_forbidden_zones::Bool = false
end

"""
    UnitCommitmentMode

SHOP UC mode: MILP with binary unit on/off variables.
"""
struct UnitCommitmentMode end

"""
    UnitLoadDispatchMode

SHOP ULD mode: pure LP with on/off decisions fixed from a prior UC pass.
"""
struct UnitLoadDispatchMode end

# TODO: solve(::ShortTermHydroProblem, ::SLPHydroSolver)
