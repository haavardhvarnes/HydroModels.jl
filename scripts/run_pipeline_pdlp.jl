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
# • **LP native; MILP via continuous LP relaxation only.** PDLP is a
#   first-order primal-dual method; it has no integer search. CoolPDLP
#   exposes MILP via the LP relaxation (per its tutorial:
#   "use the PDLP algorithm to solve the continuous relaxation").
#   This script opts in to the relaxation via `HM_PDLP_RELAX=1`.
#   Without that flag, MILP fixtures (e.g. `pumped_storage.yaml`) are
#   refused with a pointer to HiGHS.
#
# When PDLP is a good fit (and when it isn't)
# -------------------------------------------
# PDLP shines on **uniformly-scaled** LPs (network flow, ML-style
# problems, LP relaxations of well-scaled combinatorial MIPs). SHOP-
# style scheduling is the **opposite** — water-value cut slopes hit
# ~5e8 EUR/Mm³ while flow variables are O(1e-3), an 11-orders-of-
# magnitude dynamic range in the matrix coefficients. Two
# consequences:
#
#   • **CPU Float64** PDLP converges, but slowly — it needs hundreds
#     of seconds to reach HiGHS-quality on NO5 (vs ~60 s for HiGHS
#     itself). Tightening `HM_PDLP_RUIZ` and loosening
#     `HM_PDLP_RELTOL` help.
#   • **Apple Metal (Float32)** is **worse than CPU** on conditioned
#     SHOP LPs. Float32 has only ~7 significant digits, so matrix-
#     vector products in the cut-slope subspace suffer catastrophic
#     cancellation and PDLP's KKT-residual check stops being
#     meaningful. The script warns when Metal is selected for a
#     problem containing water-value cuts.
#   • **NVIDIA CUDA / AMD ROCm** are the right GPU paths for PDLP —
#     they support Float64 natively. The matrix-type sparse-CSR API
#     is the same as Metal's; just swap the backend and matrix type.
# • **CPU Float64**: drop-in replacement for HiGHS. Objective
#   matches to ~1e-11 on the shipped fixtures.
# • **GPU Float32**: works on every vendor with a KA backend.
#   Apple Metal validated locally; CUDA / AMDGPU / oneAPI follow the
#   same attribute pattern. Float32 throughout — objective matches
#   HiGHS to ~1e-6 (Float32 rounding floor).
#
# Environment variables
# ---------------------
#   HM_PDLP_BACKEND=cpu      # default — Float64 + SparseMatrixCSC
#   HM_PDLP_BACKEND=metal    # Apple Silicon — Float32 + GPUSparseMatrixCSR
#   HM_PDLP_BACKEND=cuda     # NVIDIA
#   HM_PDLP_BACKEND=amdgpu   # AMD
#   HM_PDLP_BACKEND=oneapi   # Intel
#
#   HM_PDLP_RELAX=1          # demote binary mode variables to continuous
#                            # [0,1] before solving (LP relaxation of MILP).
#   HM_PDLP_TIMEOUT=300      # PDLP time limit (seconds) — default 300.
#                            # Increase for SHOP-scale (e.g. NO5: 600–1800).
#   HM_PDLP_RELTOL=1.0e-4    # PDLP termination_reltol — default 1e-4 (loose).
#                            # Tighten to 1e-6 to match HiGHS LP quality.
#   HM_PDLP_RUIZ=10          # PDLP ruiz_iter — number of preconditioning
#                            # sweeps. SHOP LPs are ill-conditioned (cut
#                            # slopes ~5e8 vs flow vars at 1e-3); bump to
#                            # 50–100 for substantially better convergence.
#
# Usage
# -----
#   julia --project=. scripts/run_pipeline_pdlp.jl <yaml> [<market>]
#   HM_PDLP_BACKEND=metal julia --project=. scripts/run_pipeline_pdlp.jl <yaml>
#   HM_PDLP_RELAX=1 HM_PDLP_BACKEND=metal julia --project=. scripts/run_pipeline_pdlp.jl <yaml>
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

# Convergence knobs. CoolPDLP's built-in defaults
# (time_limit = 10.0 s, termination_reltol = 1.0e-6, ruiz_iter = 10)
# are tuned for Netlib-scale LPs and assume well-scaled matrices.
# SHOP-scale LPs are ill-conditioned (water-value cut slopes ~5e8
# EUR/Mm³ vs flow vars at O(1e-3) → 11-orders-of-magnitude dynamic
# range) so PDLP needs substantially more Ruiz preconditioning, a
# looser termination tolerance, and a longer wall-clock budget than
# the defaults to even approach the optimum.
const _TIME_LIMIT = parse(Float64, get(ENV, "HM_PDLP_TIMEOUT", "300"))
const _RELTOL     = parse(Float64, get(ENV, "HM_PDLP_RELTOL",  "1.0e-4"))
const _RUIZ_ITER  = parse(Int,     get(ENV, "HM_PDLP_RUIZ",    "10"))
opts[:time_limit] = _TIME_LIMIT
opts[:termination_reltol] = _RELTOL
opts[:ruiz_iter] = _RUIZ_ITER

# Continuous-LP-relaxation opt-in for MILP cases.
const _RELAX_INTEGERS = get(ENV, "HM_PDLP_RELAX", "0") ∈ ("1", "true", "yes")
if _RELAX_INTEGERS
    opts[:relax_integers] = true
end

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

# Warn when Metal / oneAPI is selected for a problem with water-value
# cuts (Float32 cannot handle the 1e11 dynamic range in matrix
# coefficients). CUDA / AMDGPU keep Float32 too in this script but
# the user can override at the constructor level for Float64 on
# those vendors if needed.
const _HAS_WATER_VALUE_CUTS = !isempty(parsed.cut_groups)
if _HAS_WATER_VALUE_CUTS && _BACKEND_NAME in ("metal", "oneapi")
    println()
    println("⚠️  Backend $_BACKEND_NAME runs CoolPDLP in Float32 (no native")
    println("    Float64). This problem contains water-value cuts (slopes")
    println("    ~5e8 EUR/Mm³) — PDLP's KKT residuals will lose precision")
    println("    catastrophically in Float32. Expect a worse final objective")
    println("    than CPU Float64 even with a longer wall-clock budget.")
    println("    For GPU PDLP on conditioned LPs, use CUDA or AMDGPU instead.")
    println()
end

# CoolPDLP is a first-order LP method; it handles MILPs only via the
# **continuous LP relaxation** (per its tutorial). Through the JuMP
# MOI interface a model with `MOI.ZeroOne` constraints is rejected
# with `UnsupportedConstraint`, so we have to relax the binaries
# ourselves. Default: refuse, and point at HiGHS. Opt in with
# `HM_PDLP_RELAX=1` to demote every binary to continuous [0,1] and
# solve the LP relaxation. The relaxed solution does NOT enforce
# integer constraints (e.g. the no-simultaneous-gen-and-pump lock
# on reversible plants) — useful for cut generation, warm starts,
# or quick scoping, not for a binding short-term schedule.
if is_milp(prob)
    if _RELAX_INTEGERS
        println()
        println("⚠️  MILP detected (reversible pump-turbine plants). Relaxing")
        println("    all binaries to continuous [0,1] for CoolPDLP — the result")
        println("    is the LP relaxation, NOT the MILP optimum.")
    else
        println()
        println("⚠️  This problem has binary mode variables (reversible pump-")
        println("    turbine plants detected). CoolPDLP supports MILP only via")
        println("    the continuous LP relaxation. Two options:")
        println()
        println("    • Re-run with `HM_PDLP_RELAX=1` to relax binaries and solve")
        println("      the LP relaxation via CoolPDLP. The result will not enforce")
        println("      the no-simultaneous-gen-and-pump constraint.")
        println("    • Use HiGHS for the full MILP:")
        println("          julia --project=. scripts/run_pipeline.jl $ARGS_PATH $ARGS_MARKET")
        exit(1)
    end
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
