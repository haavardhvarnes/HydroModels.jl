#!/usr/bin/env julia
#
# Same pipeline as run_pipeline.jl, but the LP is handed to
# CoolPDLP (https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl)
# instead of HiGHS.
#
# Why CoolPDLP is interesting here
# --------------------------------
# CoolPDLP is a pure-Julia primal-dual hybrid gradient (PDLP) LP
# solver built on KernelAbstractions.jl. The interesting properties
# for this package:
#
#   • Hardware-agnostic — same source compiles on CPU, NVIDIA,
#     AMD, Apple, Intel. Matches HydroModelsOpt's backend abstraction
#     conceptually.
#   • MathOptInterface optimizer — drop-in replacement for HiGHS via
#     `JuMPSolver(CoolPDLP.Optimizer)`. No bridge code required.
#   • Pluggable sparse matrix type — swap `SparseMatrixCSC` for
#     `cuSPARSE.CuSparseMatrixCSR` on NVIDIA, etc.
#
# Capabilities and limitations
# ----------------------------
# CoolPDLP is **LP only** — it cleanly rejects MILP via MOI:
#
#   ERROR: UnsupportedConstraint:
#     `MOI.VariableIndex`-in-`MOI.ZeroOne` constraints are not
#     supported by the solver you have chosen.
#
# So `pumped_storage.yaml` (reversible pump-turbine ⇒ MILP) will not
# run here; use HiGHS instead.
#
# For LP-only SHOP cases (minimal.yaml, minimal_with_reserves.yaml,
# cascade_spill.yaml, multi_source.yaml, junction.yaml, plus any
# real SHOP case without binary modes) CoolPDLP solves to the same
# objective as HiGHS, typically within ~1e-11 relative gap.
#
# GPU notes
# ---------
#   • NVIDIA: load CUDA + cuSPARSE and pass
#       JuMPSolver(CoolPDLP.Optimizer; options = Dict(
#           :backend => CUDABackend(),
#           :matrix_type => cuSPARSE.CuSparseMatrixCSR,
#       ))
#     Expected to work out of the box per CoolPDLP's README.
#   • AMD: same pattern with `AMDGPU.ROCBackend()` and
#     `AMDGPU.rocSPARSE.ROCSparseMatrixCSR`.
#   • Apple Metal: NOT yet usable. Metal has no native Float64 and
#     no Metal-flavored sparse matrix type in the Julia ecosystem
#     today; CoolPDLP's `adapt(MetalBackend(), ::Vector{Float64})`
#     errors out. For Metal use `run_pipeline_gpu.jl` (Lagrangian
#     Bellman DP), not this script.
#   • Intel oneAPI: same Metal-style limitations.
#
# Usage:
#   julia --project=. scripts/run_pipeline_pdlp.jl <path-to-yaml> [<market_name>]
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

# Disambiguate — both HydroModelsOpt and CoolPDLP export `solve`.
const SOLVE = HydroModelsOpt.solve

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
# Solve.
# --------------------------------------------------------------------
println("\nBuilding and solving LP with CoolPDLP (PDLP first-order, CPU)…")
solver = JuMPSolver(CoolPDLP.Optimizer; options = Dict(:show_progress => false))
@time sol_pdlp = SOLVE(prob, solver)

println("\nCoolPDLP results:")
println("  termination: ", sol_pdlp.termination)
@printf("  objective:   %.6f EUR\n", sol_pdlp.objective)

# --------------------------------------------------------------------
# Side-by-side comparison with HiGHS.
# --------------------------------------------------------------------
println("\nReference: same problem solved with HiGHS…")
solver_highs = JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false))
@time sol_highs = SOLVE(prob, solver_highs)

println("\nHiGHS results:")
println("  termination: ", sol_highs.termination)
@printf("  objective:   %.6f EUR\n", sol_highs.objective)

gap = abs(sol_pdlp.objective - sol_highs.objective) /
      max(abs(sol_highs.objective), 1.0e-9)
@printf("\nRelative objective gap (CoolPDLP vs HiGHS): %.3e\n", gap)
if gap < 1.0e-4
    println("  ✓ within 1e-4 of HiGHS — CoolPDLP is a viable drop-in for this LP.")
else
    println("  ⚠ larger than 1e-4 — check PDLP termination tolerances (",
            "`primal_weight_damping`, `termination_reltol` …)")
end

# --------------------------------------------------------------------
# Sanity-check that the primal tables look comparable.
# --------------------------------------------------------------------
println("\nTable shape comparison:")
for fld in (:storage, :p_gen, :q_gen, :q_tunnel, :spill, :revenue)
    n_p = length(Tables.columns(getproperty(sol_pdlp, fld)).time)
    n_h = length(Tables.columns(getproperty(sol_highs, fld)).time)
    @printf("  %-12s  pdlp=%-5d  highs=%-5d  %s\n",
            String(fld), n_p, n_h, n_p == n_h ? "✓" : "✗ shape mismatch")
end
