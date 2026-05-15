"""
    HydroModelsOpt

Optimization layer of the HydroModels meta-package — SDDP, SLP/SHOP,
GPU Lagrangian, and JuMP-backed solvers for the problem types defined
in `HydroModelsCore`. Also hosts uncertainty fitting (PAR(1), Markov
price, scenario generation) and the long-term → short-term toolchain.

Re-exports `HydroModelsCore` so that `using HydroModelsOpt` is enough
for a full optimization workflow.

See `CLAUDE.md` at the repo root for architectural principles and the
list of canonical references.
"""
module HydroModelsOpt

using Reexport
@reexport using HydroModelsCore

using Dates
using LinearAlgebra
using OrderedCollections
using SparseArrays
using Random
using Statistics

using KernelAbstractions
using GPUArrays

using JuMP
using MathOptInterface
const MOI = MathOptInterface

# Uncertainty-model fitting (PAR(1), Markov price nodes, scenario generation)
# migrated to `HydroModelsForecast` in milestone C1. Users wanting fitted
# inflow / price models should `using HydroModelsForecast` directly.

# Kernels (KernelAbstractions, backend-agnostic)
include("kernels/subgradient.jl")
include("kernels/bellman.jl")
include("kernels/scenario_batch.jl")

# Solvers
include("solvers/abstract.jl")
include("solvers/jump_backend.jl")
include("solvers/jump_baseline.jl")    # SHOP-style LP/MILP construction
include("solvers/jump_solve.jl")       # HydroSolution + solve dispatch
include("solvers/sddp/core.jl")
include("solvers/sddp/prodrisk_hybrid.jl")
include("solvers/sddp/markovian.jl")
include("solvers/sddp/scenario_tree.jl")
include("solvers/slp/core.jl")
include("solvers/slp/uc_mode.jl")
include("solvers/slp/uld_mode.jl")
include("solvers/lagrangian/core.jl")
include("solvers/lagrangian/bundle.jl")
include("solvers/lagrangian/primal_recovery.jl")
include("solvers/lagrangian/scenario_decomposition.jl")
include("solvers/lagrangian/dp_decomposition.jl")

# Toolchain (long-term → short-term workflows)
include("toolchain/longterm_to_shortterm.jl")

# ============================================================
# Public API
# ============================================================

# Solver types
export AbstractSolver, AbstractGPUSolver
export JuMPSolver
export SDDPHydroSolver
export SLPHydroSolver, UnitCommitmentMode, UnitLoadDispatchMode
export LagrangianHydroSolver

# Solver-side strategies
export ParallelismMode, SynchronousParallel, AsynchronousParallel, TotallyAsynchronous
export DecompositionStrategy, ScenarioDecomposition, SpatialDecomposition
export StepSizeRule, ConstantStep, DiminishingStep, PolyakStep
export compute_step

# JuMP baseline outputs and helpers
export HydroSolution
export default_lp_solver
export is_milp

# Toolchain
export HydropowerToolchain, run_toolchain
export with_end_value_cuts, synthetic_end_value_cuts, extract_end_value_cuts

# Core verb
export solve

end # module HydroModelsOpt
