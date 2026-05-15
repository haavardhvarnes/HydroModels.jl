"""
    HydroModelsCore

Pure-domain layer of the HydroModels meta-package — physical types,
problem formulations, uncertainty model containers, topology builders,
backend abstraction, and generic utilities. No solver code lives here.

See `CLAUDE.md` at the repo root for architectural principles and the
list of canonical references.
"""
module HydroModelsCore

using Dates
using LinearAlgebra
using SparseArrays
using Random
using Statistics

using OrderedCollections

using KernelAbstractions
import KernelAbstractions: allocate    # required to extend + re-export under Julia 1.12+
using GPUArrays

using MathOptInterface
const MOI = MathOptInterface

# Utilities (PiecewiseLinear / PWL is used by types/shop.jl, so it
# must be included before that file)
include("utils/pwl.jl")
include("utils/time.jl")

# Type definitions
# Order matters: types/problems.jl references SHOP-style types
# (Plant, Reservoir, Generator, Pump, Tunnel) from types/shop.jl in
# ShopShortTermProblem, and references MarketSeries / ReserveGroup /
# CutGroup from timeseries / markets / solutions. They must all be
# loaded first.
include("types/physical.jl")
include("types/uncertainty.jl")
include("types/solutions.jl")
include("types/timeseries.jl")
include("types/markets.jl")
include("types/shop.jl")
include("types/problems.jl")

# Backend abstraction (vendor-neutral compute substrate)
include("backends/abstract.jl")

# Problem construction
include("problems/topology.jl")
include("problems/builders.jl")

# ============================================================
# Public API
# ============================================================

# ProdRisk-style physical types
export HydroModule, TurbineUnit, ReservoirType
export RegulationReservoir, BufferReservoir
export ModuleConstraints, PenaltyValues
export WaterTopology

# SHOP-style physical types
export Plant, Reservoir, Generator, Pump, Tunnel, Junction

# Time-series containers
export InflowSeries, MarketSeries

# Reserve-market types
export ReserveSpec, ReserveGroup

# Uncertainty model containers
export UncertaintyModel
export ScenarioTree, ScenarioNode
export StagewiseIndependent
export MarkovianUncertainty
export ProdRiskUncertainty

# Problems
export AbstractOptProblem
export LongTermHydroProblem
export FundamentalMarketProblem
export ShortTermHydroProblem
export ShopShortTermProblem
export StochasticShortTermProblem
export ElectricityMarket, LoadBlock
export EndValueDescription

# Risk measures (used by SDDP-family solvers but defined here as
# problem-level concepts)
export AbstractRiskMeasure, Expectation, CVaR, NestedCVaR

# Solutions and cut representations (problem data, not solver artefacts)
export Solution, WaterValueCuts, BendersCut
export ReservoirWaterValue, CutGroup

# Backends
export ComputeBackend, CPUBackend, GPUBackend
export default_backend, set_default_backend!

# Utilities exposed for downstream solvers and tests
export PiecewiseLinear, PWL, evaluate, slopes
export build_topology
export allocate, zeros_like, ones_like, is_gpu, ka_backend

end # module HydroModelsCore
