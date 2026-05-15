#!/usr/bin/env julia
#
# Same pipeline as run_pipeline.jl, but the LP is handed to
# CoolPDLP (https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl)
# instead of HiGHS.
#
# Why CoolPDLP is interesting here
# --------------------------------
# CoolPDLP is a pure-Julia primal-dual hybrid gradient (PDLP) LP
# solver built on KernelAbstractions.jl:
#
#   • Hardware-agnostic — same source compiles on CPU, NVIDIA, AMD,
#     Apple, Intel via KA backends. Matches HydroModelsOpt's
#     backend abstraction exactly.
#   • MathOptInterface optimizer — drops into the existing
#     `JuMPSolver(CoolPDLP.Optimizer)` dispatch path. No bridge code.
#   • Pluggable scalar / index / sparse-matrix types via JuMP
#     attributes — `float_type`, `int_type`, `matrix_type`,
#     `backend`. The MOI wrapper forwards these straight to the
#     direct `PDLP(T, Ti, M; backend=…)` constructor.
#
# Capabilities and limitations
# ----------------------------
# • **LP only** — refuses MILP cleanly via MOI's
#   UnsupportedConstraint. `pumped_storage.yaml` (reversible
#   pump-turbine ⇒ MILP) is detected up front and the script
#   redirects to HiGHS.
# • **CPU Float64**: drop-in replacement for HiGHS. Objective
#   matches to ~1e-11 on the shipped fixtures.
# • **GPU Float32**: works on every vendor with a KA backend.
#   Apple Metal validated locally; CUDA / AMDGPU / oneAPI follow the
#   same attribute pattern. Float32 throughout — objective matches
#   HiGHS to ~1e-6 (Float32 rounding floor).
#
# Backend selection (via env var)
# -------------------------------
#   HM_PDLP_BACKEND=cpu      # default — Float64 + SparseMatrixCSC
#   HM_PDLP_BACKEND=metal    # Apple Silicon — Float32 + GPUSparseMatrixCSR
#   HM_PDLP_BACKEND=cuda     # NVIDIA
#   HM_PDLP_BACKEND=amdgpu   # AMD
#   HM_PDLP_BACKEND=oneapi   # Intel
#
# Usage
# -----
#   julia --project=. scripts/run_pipeline_pdlp.jl <yaml> [<market>]
#   HM_PDLP_BACKEND=metal julia --project=. scripts/run_pipeline_pdlp.jl <yaml>
#
# Install CoolPDLP first:
#   julia --project=. -e 'using Pkg; Pkg.add("CoolPDLP")'

using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using CoolPDLP
using HiGHS
using Tables
using Printf

const SOLVE = HydroModelsOpt.solve   # disambiguate from CoolPDLP.solve

# --------------------------------------------------------------------
# Backend selection — load the requested GPU vendor package and build
# the JuMP-attribute option dict that CoolPDLP's MOI_wrapper forwards
# to the direct `PDLP(T, Ti, M; backend=…)` constructor.
# --------------------------------------------------------------------
const _BACKEND_NAME = lowercase(get(ENV, "HM_PDLP_BACKEND", "cpu"))

# Load the requested GPU vendor package at top level so the world
# age advances before any function references it. Calling
# `@eval using …` inside a function body bumps the world age past
# the function's own compile point and triggers `UndefVarError` on
# the first reference.
if _BACKEND_NAME == "metal"
    @eval using Metal
elseif _BACKEND_NAME == "cuda"
    @eval using CUDA
elseif _BACKEND_NAME == "amdgpu"
    @eval using AMDGPU
elseif _BACKEND_NAME == "oneapi"
    @eval using oneAPI
elseif _BACKEND_NAME != "cpu"
    error("Unknown HM_PDLP_BACKEND=\"$_BACKEND_NAME\" — use one of cpu/metal/cuda/amdgpu/oneapi.")
end

function _build_pdlp_options(backend_name::AbstractString)
    if backend_name == "cpu"
        return Dict{Symbol, Any}(:show_progress => false), "CPU (Float64, SparseMatrixCSC)"
    end
    common = Dict{Symbol, Any}(
        :show_progress => false,
        :float_type    => Float32,
        :int_type      => Int32,
        :matrix_type   => CoolPDLP.GPUSparseMatrixCSR,
    )
    if backend_name == "metal"
        common[:backend] = Metal.MetalBackend()
        return common, "Apple Metal (Float32, GPUSparseMatrixCSR)"
    elseif backend_name == "cuda"
        common[:backend] = CUDA.CUDABackend()
        return common, "NVIDIA CUDA (Float32, GPUSparseMatrixCSR)"
    elseif backend_name == "amdgpu"
        common[:backend] = AMDGPU.ROCBackend()
        return common, "AMD ROCm (Float32, GPUSparseMatrixCSR)"
    elseif backend_name == "oneapi"
        common[:backend] = oneAPI.oneAPIBackend()
        return common, "Intel oneAPI (Float32, GPUSparseMatrixCSR)"
    end
    error("unreachable")
end

opts, backend_label = _build_pdlp_options(_BACKEND_NAME)

# --------------------------------------------------------------------
# Args.
# --------------------------------------------------------------------
const ARGS_PATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..", "HydroModelsData", "test", "data", "minimal.yaml")
const ARGS_MARKET = length(ARGS) >= 2 ? ARGS[2] : "NO3"

isfile(ARGS_PATH) || error("File not found: $ARGS_PATH")

println("Parsing $ARGS_PATH (market=$ARGS_MARKET)…")
@time parsed = read_shop_yaml(ARGS_PATH; market_name = ARGS_MARKET)
println("  plants=$(length(parsed.plants))  reservoirs=$(length(parsed.reservoirs))  ",
        "generators=$(length(parsed.generators))  pumps=$(length(parsed.pumps))  ",
        "tunnels=$(length(parsed.tunnels))")

prob = ShopShortTermProblem(parsed)

# CoolPDLP cannot handle MILP — refuse early with a useful message.
if is_milp(prob)
    println()
    println("⚠️  This problem has binary mode variables (reversible pump-")
    println("    turbine plants detected). CoolPDLP is a first-order LP")
    println("    solver and does not support MILP. Use HiGHS:")
    println()
    println("        julia --project=. scripts/run_pipeline.jl $ARGS_PATH $ARGS_MARKET")
    exit(1)
end

# --------------------------------------------------------------------
# Solve with CoolPDLP on the requested backend.
# --------------------------------------------------------------------
println("\nBuilding and solving LP with CoolPDLP on $backend_label…")
solver = JuMPSolver(CoolPDLP.Optimizer; options = opts)
@time sol_pdlp = SOLVE(prob, solver)

println("\nCoolPDLP results:")
println("  termination: ", sol_pdlp.termination)
@printf("  objective:   %.6f EUR\n", sol_pdlp.objective)

# --------------------------------------------------------------------
# Side-by-side reference solve with HiGHS.
# --------------------------------------------------------------------
println("\nReference: same problem solved with HiGHS (CPU)…")
solver_highs = JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false))
@time sol_highs = SOLVE(prob, solver_highs)

println("\nHiGHS results:")
println("  termination: ", sol_highs.termination)
@printf("  objective:   %.6f EUR\n", sol_highs.objective)

gap = abs(sol_pdlp.objective - sol_highs.objective) /
      max(abs(sol_highs.objective), 1.0e-9)
@printf("\nRelative objective gap (CoolPDLP vs HiGHS): %.3e\n", gap)
# Tighter tolerance on CPU Float64; looser on Float32 GPU paths.
tol = _BACKEND_NAME == "cpu" ? 1.0e-4 : 1.0e-3
if gap < tol
    println("  ✓ within $tol of HiGHS — CoolPDLP is a viable drop-in.")
else
    println("  ⚠ larger than $tol — check PDLP termination tolerances ",
            "(`termination_reltol`, `max_kkt_passes` …) or the Float32 ",
            "rounding floor on GPU paths.")
end

# --------------------------------------------------------------------
# Table shapes — sanity check that the JuMP→HydroSolution path
# behaves the same regardless of optimizer.
# --------------------------------------------------------------------
println("\nTable shape comparison:")
for fld in (:storage, :p_gen, :q_gen, :q_tunnel, :spill, :revenue)
    n_p = length(Tables.columns(getproperty(sol_pdlp, fld)).time)
    n_h = length(Tables.columns(getproperty(sol_highs, fld)).time)
    @printf("  %-12s  pdlp=%-5d  highs=%-5d  %s\n",
            String(fld), n_p, n_h, n_p == n_h ? "✓" : "✗ shape mismatch")
end
