#!/usr/bin/env julia
#
# Benchmark CoolPDLP (CPU Float64) against HiGHS on every shipped LP
# fixture in `HydroModelsData/test/data/`. Prints a tidy table:
# variables, constraints, build time, warm solve time, objective, gap.
#
# Both solvers run on CPU in Float64 — apples-to-apples. The
# pumped_storage.yaml fixture is MILP; HiGHS solves the MILP, CoolPDLP
# solves the continuous LP relaxation (with `relax_integers=true`),
# so the objective gap on that row reflects the integrality gap, not a
# solver disagreement.
#
# Usage:
#   julia --project=. scripts/bench_pdlp_vs_highs.jl
#
# Optional — also run the local NO5 PreSpot real-data case:
#   HYDROMODELS_REAL_NO5=../HydroModels_depr/data/real/Cases_NO5-PreSpotNoScenarios-2024-04-26-12-41-43-13.yaml \
#       julia --project=. scripts/bench_pdlp_vs_highs.jl
#
# Tune CoolPDLP convergence:
#   HM_PDLP_TIMEOUT=300   # default 60 s per case
#   HM_PDLP_RELTOL=1.0e-6 # default 1e-6 (HiGHS-quality)
#   HM_PDLP_RUIZ=10       # default 10 (CoolPDLP default)

using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using CoolPDLP
using HiGHS
using Printf
using Statistics: mean

const SOLVE = HydroModelsOpt.solve   # disambiguate from CoolPDLP.solve

const TIMEOUT = parse(Float64, get(ENV, "HM_PDLP_TIMEOUT", "60"))
const RELTOL  = parse(Float64, get(ENV, "HM_PDLP_RELTOL",  "1.0e-6"))
const RUIZ    = parse(Int,     get(ENV, "HM_PDLP_RUIZ",    "10"))

const FIXTURE_DIR = joinpath(dirname(pathof(HydroModelsData)), "..", "test", "data")
const FIXTURES = String[
    joinpath(FIXTURE_DIR, "minimal.yaml"),
    joinpath(FIXTURE_DIR, "minimal_with_reserves.yaml"),
    joinpath(FIXTURE_DIR, "cascade_spill.yaml"),
    joinpath(FIXTURE_DIR, "multi_source.yaml"),
    joinpath(FIXTURE_DIR, "junction.yaml"),
    joinpath(FIXTURE_DIR, "pumped_storage.yaml"),    # MILP — relaxed for PDLP
]
const NO5_PATH = get(ENV, "HYDROMODELS_REAL_NO5", "")
if !isempty(NO5_PATH) && isfile(NO5_PATH)
    push!(FIXTURES, NO5_PATH)
end

# Solver wall-clock is dominated by JIT on cold cache; warm-up once
# per solver before timing anything.
function _warmup!(prob)
    SOLVE(prob, JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false)))
    opts = Dict{Symbol, Any}(
        :show_progress      => false,
        :relax_integers     => true,
        :time_limit         => TIMEOUT,
        :termination_reltol => RELTOL,
        :ruiz_iter          => RUIZ,
    )
    SOLVE(prob, JuMPSolver(CoolPDLP.Optimizer; options = opts))
    return nothing
end

# Count JuMP variables and constraints from a built model. Cheaper to
# rebuild than to plumb through the solver tuple, and the build is
# already cached after warm-up.
using JuMP
function _model_stats(prob)
    _, model = HydroModelsOpt._build_short_term_lp(prob, HiGHS.Optimizer)
    nvars = JuMP.num_variables(model)
    nbins = count(JuMP.is_binary, JuMP.all_variables(model))
    nconss = sum(JuMP.num_constraints(model, F, S)
                 for (F, S) in JuMP.list_of_constraint_types(model);
                 init = 0)
    return (; nvars, nbins, nconss)
end

struct BenchRow
    name::String
    nvars::Int
    nbins::Int
    nconss::Int
    highs_term::String
    highs_obj::Float64
    highs_solve_s::Float64
    pdlp_term::String
    pdlp_obj::Float64
    pdlp_solve_s::Float64
    rel_gap::Float64
end

function _bench_one(yaml_path::AbstractString)
    name = basename(yaml_path)
    market = startswith(name, "Cases_NO5") ? "NO5" : "NO3"
    parsed = read_shop_yaml(yaml_path; market_name = market)
    prob   = ShopShortTermProblem(parsed)

    stats = _model_stats(prob)

    # Warm both code paths so the timings reflect solve, not JIT.
    _warmup!(prob)

    # HiGHS — MILP-aware: if any binaries exist, HiGHS solves the MILP.
    highs_solver = JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false))
    t_h = @elapsed sol_h = SOLVE(prob, highs_solver)

    # CoolPDLP — always solve the LP (relax integers if any).
    pdlp_opts = Dict{Symbol, Any}(
        :show_progress      => false,
        :relax_integers     => true,
        :time_limit         => TIMEOUT,
        :termination_reltol => RELTOL,
        :ruiz_iter          => RUIZ,
    )
    pdlp_solver = JuMPSolver(CoolPDLP.Optimizer; options = pdlp_opts)
    t_p = @elapsed sol_p = SOLVE(prob, pdlp_solver)

    rel_gap = abs(sol_p.objective - sol_h.objective) /
              max(abs(sol_h.objective), 1.0e-9)

    return BenchRow(name, stats.nvars, stats.nbins, stats.nconss,
                    sol_h.termination, sol_h.objective, t_h,
                    sol_p.termination, sol_p.objective, t_p, rel_gap)
end

rows = BenchRow[]
for yaml_path in FIXTURES
    if !isfile(yaml_path)
        @warn "Skipping missing fixture: $yaml_path"
        continue
    end
    println("Benchmarking $(basename(yaml_path))…")
    try
        push!(rows, _bench_one(yaml_path))
    catch e
        @warn "$(basename(yaml_path)) failed: $(typeof(e))"
    end
end

println()
println("CoolPDLP (CPU Float64) vs HiGHS — warm wall-clock, identical LP build")
println("PDLP options: time_limit=$TIMEOUT s, reltol=$RELTOL, ruiz_iter=$RUIZ")
println(repeat("=", 130))

@printf("%-46s  %5s  %4s  %5s  %12s  %10s  %12s  %10s  %10s\n",
        "fixture", "vars", "bins", "cons",
        "highs obj", "highs s", "pdlp obj", "pdlp s", "rel gap")
println(repeat("-", 130))
for r in rows
    @printf("%-46s  %5d  %4d  %5d  %12.4g  %10.3f  %12.4g  %10.3f  %10.2e\n",
            r.name, r.nvars, r.nbins, r.nconss,
            r.highs_obj, r.highs_solve_s,
            r.pdlp_obj,  r.pdlp_solve_s, r.rel_gap)
end
println(repeat("=", 130))

# Summary headline.
n = length(rows)
n == 0 && exit(0)
n_optimal = count(r -> r.pdlp_term == "OPTIMAL", rows)
n_within_1e6 = count(r -> r.rel_gap < 1.0e-6, rows)
n_within_1e3 = count(r -> r.rel_gap < 1.0e-3, rows)
total_h = sum(r.highs_solve_s for r in rows)
total_p = sum(r.pdlp_solve_s  for r in rows)

println("\nSummary across $n fixture(s):")
@printf("  CoolPDLP reached OPTIMAL termination on %d/%d.\n", n_optimal, n)
@printf("  Relative objective gap <1e-6 on %d/%d, <1e-3 on %d/%d.\n",
        n_within_1e6, n, n_within_1e3, n)
@printf("  Total warm wall-clock — HiGHS %.3fs   CoolPDLP %.3fs   ratio %.2fx\n",
        total_h, total_p, total_p / max(total_h, 1.0e-9))
