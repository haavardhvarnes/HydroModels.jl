"""
SDDP solver — the unified interface for stochastic dual dynamic programming
variants.

Dispatch on the `UncertaintyModel` type chooses the algorithmic variant:
- `ScenarioTree` → standard tree-based SDDP
- `StagewiseIndependent` → SDDP.jl LinearPolicyGraph
- `MarkovianUncertainty` → SDDP.jl MarkovianPolicyGraph
- `ProdRiskUncertainty` → Nordic combined SDP/SDDP (Gjelsvik 1999)
"""

"""
    SDDPHydroSolver(; max_iterations, num_forward_scenarios, ...)

Unified SDDP solver. The same solver works on all `UncertaintyModel`
types via multiple dispatch on the problem's `uncertainty` field.

# Fields
- `max_iterations`: outer SDDP iterations
- `num_forward_scenarios`: scenarios sampled in each forward pass
- `risk_measure`: `Expectation()` or `CVaR` or `NestedCVaR`
- `subproblem_solver`: the LP/MILP solver used for each stage subproblem
- `parallelism`: how forward / backward passes are parallelized
- `cut_management`: cut storage strategy (standard / level-method)
- `convergence_tolerance`: stop when relative gap < this
"""
Base.@kwdef struct SDDPHydroSolver{R<:AbstractRiskMeasure, S<:AbstractSolver} <: AbstractSolver
    max_iterations::Int            = 500
    num_forward_scenarios::Int     = 100
    risk_measure::R                = Expectation()
    subproblem_solver::S
    parallelism::ParallelismMode   = SynchronousParallel()
    cut_management::Symbol         = :standard
    convergence_tolerance::Float64 = 1.0e-3
end

# TODO: actual solve implementations for each (Problem, Uncertainty) pair
# See sddp/prodrisk_hybrid.jl, sddp/markovian.jl, sddp/scenario_tree.jl
