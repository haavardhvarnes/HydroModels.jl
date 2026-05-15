#!/usr/bin/env julia
#
# GPU pipeline — same SHOP YAML in, but solved with the Lagrangian
# KA Bellman DP path instead of HiGHS.
#
# ============================================================
# IMPORTANT — this is NOT an apples-to-apples comparison with
# run_pipeline.jl. The two scripts solve fundamentally different
# problems.
#
# run_pipeline.jl
#     ShopShortTermProblem  — deterministic LP / MILP (24–168 h
#                             horizon), full SHOP detail (PQ curves,
#                             reserves, junctions, head-clamping,
#                             pumped-storage binary modes …).
#     Solver  : JuMPSolver(HiGHS.Optimizer)  — CPU only.
#     Objective: EUR (revenue – penalties) over the short horizon.
#
# run_pipeline_gpu.jl  (this script)
#     LongTermHydroProblem  — stochastic weekly-stage scheduling
#                             with simplified `HydroModule` types
#                             (energy factor × flow, no head, no
#                             reserves, no junctions).
#     Solver  : LagrangianHydroSolver(; dispatch = :auto)
#                  — KA Bellman DP path on the default backend.
#     Backend : whichever GPU vendor extension is loaded — CUDA /
#               AMDGPU / Metal / oneAPI — falls back to CPU if none.
#     Objective: in-sample mean of the dualised Lagrangian; NOT
#                directly comparable to the HiGHS LP objective.
#
# The reason for the asymmetry: no GPU LP / MILP solver is wired
# into HydroModelsOpt today (cuPDLP.jl, MadIPM.jl, cuOpt would be
# the natural fits). The only GPU dispatch in the package is the
# Lagrangian Bellman DP, which targets the long-term scheduling
# class. So we use the SHOP YAML as **input data** and reproject it
# onto the long-term type system.
#
# What this script demonstrates:
#   • the GPU dispatch path actually runs end-to-end
#   • which backend (CPU vs CUDA vs Metal vs …) got selected
#   • whether `:auto` resolved to `:lp` (CPU LP via HiGHS) or
#     `:dp` (KA Bellman DP)
#   • wall-clock and primal/dual shape so the size of the lift is
#     visible.
#
# Usage:
#   julia --project=. scripts/run_pipeline_gpu.jl <path-to-yaml> [<market_name>]
#
# Apple Silicon: add Metal to the project first
#   julia --project=. -e 'using Pkg; Pkg.add("Metal")'
# NVIDIA / AMD / Intel: add CUDA / AMDGPU / oneAPI similarly.
# ============================================================

using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HiGHS
using Tables
using Printf
using Statistics: mean

# ----------------------------------------------------------------
# Backend selection — try GPU vendors in a sensible order.
# ----------------------------------------------------------------
const _GPU_PACKAGE = let
    candidates = ["Metal", "CUDA", "AMDGPU", "oneAPI"]
    selected = nothing
    for pkg in candidates
        try
            @eval using $(Symbol(pkg))
            selected = pkg
            break
        catch _
            # Package isn't installed; keep trying.
        end
    end
    selected
end

const _BACKEND = default_backend()
println("Default backend after extension loading: ",
        _BACKEND === nothing ? "nothing" : nameof(typeof(_BACKEND)))
if _GPU_PACKAGE === nothing
    println("(no GPU vendor package found — running on CPU)")
else
    println("(GPU vendor active: $_GPU_PACKAGE)")
end

# ----------------------------------------------------------------
# Args
# ----------------------------------------------------------------
const ARGS_PATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..", "HydroModelsData", "test", "data", "minimal.yaml")
const ARGS_MARKET = length(ARGS) >= 2 ? ARGS[2] : "NO3"

isfile(ARGS_PATH) || error("File not found: $ARGS_PATH")

println()
println("Parsing $ARGS_PATH (market=$ARGS_MARKET)…")
@time parsed = read_shop_yaml(ARGS_PATH; market_name = ARGS_MARKET)
println("  plants=$(length(parsed.plants))  reservoirs=$(length(parsed.reservoirs))  ",
        "generators=$(length(parsed.generators))")

# ----------------------------------------------------------------
# SHOP YAML → LongTermHydroProblem conversion (simplified)
#
# This is a *deliberately coarse* adapter — its purpose is to give
# the Lagrangian DP a runnable problem derived from the same input
# data, not to faithfully reproduce the SHOP LP.
#
# Reservoir → HydroModule{Float64, RegulationReservoir}:
#   max_vol         ← Reservoir.s_max
#   initial_vol     ← Reservoir.s0
#   rated_discharge ← qmax of the first generator owning this reservoir
#   energy_factor   ← pmax / qmax of that generator (kWh/m³ ≈ MW per m³/s)
#   discharge_to    ← the generator's to_res (when a downstream reservoir
#                     exists in `parsed.reservoirs`)
#
# Inflow uncertainty: two equiprobable scenarios per stage, ±50 %
# around the mean of the YAML's inflow series (per reservoir).
# Price: one stage_price per weekly stage, equal to the mean of the
# YAML's day-ahead market series.
# ----------------------------------------------------------------
function _shop_to_longterm(parsed; num_stages::Int = 12)
    res_names = sort!(collect(keys(parsed.reservoirs)))
    modules = HydroModule{Float64, RegulationReservoir}[]

    for rname in res_names
        r = parsed.reservoirs[rname]
        # Find the first generator drawing from this reservoir.
        gens = filter(g -> rname in g.from_res, parsed.generators)
        if isempty(gens)
            rated = 1.0
            ef = 0.0
            disch_to = nothing
        else
            g = first(gens)
            rated = max(g.qmax, 1.0e-3)
            ef = g.qmax > 0 ? g.pmax / g.qmax : 0.0
            disch_to = (g.to_res in res_names) ? Symbol(g.to_res) : nothing
        end
        push!(modules, HydroModule{Float64, RegulationReservoir}(;
            name            = Symbol(rname),
            max_vol         = r.s_max,
            initial_vol     = clamp(r.s0, r.s_min, r.s_max),
            rated_discharge = rated,
            energy_factor   = ef,
            discharge_to    = disch_to,
        ))
    end

    topology = build_topology(modules)

    # Mean inflow per reservoir → two-scenario stagewise-independent.
    mean_inflow = map(res_names) do rname
        r = parsed.reservoirs[rname]
        vals = collect(values(r.inflow.by_time))
        isempty(vals) ? 1.0 : mean(vals)
    end
    realizations = [
        [1.5 .* mean_inflow, 0.5 .* mean_inflow]
        for _ in 1:num_stages
    ]
    probs = [[0.5, 0.5] for _ in 1:num_stages]
    uncertainty = StagewiseIndependent{Float64}(realizations, probs)

    # Stage prices: mean of the market series, replicated.
    price_vals = collect(values(parsed.market.by_time))
    pavg = isempty(price_vals) ? 30.0 : mean(price_vals)
    stage_prices = fill(Float64(pavg), num_stages)

    prob = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(;
        modules      = modules,
        topology     = topology,
        num_stages   = num_stages,
        uncertainty  = uncertainty,
        stage_prices = stage_prices,
    )

    return prob, res_names
end

const NUM_STAGES = 12
println("\nConverting SHOP cascade → LongTermHydroProblem with ",
        "num_stages=$NUM_STAGES, stagewise-independent inflow…")
long_prob, res_names = _shop_to_longterm(parsed; num_stages = NUM_STAGES)
nmod = length(long_prob.modules)
println("  modules=$nmod  stages=$(long_prob.num_stages)")

# ----------------------------------------------------------------
# Configure the Lagrangian solver. `dispatch = :auto` picks `:dp`
# on a GPUBackend and `:lp` on CPUBackend. K_grid controls the
# storage-discretisation accuracy of the DP path.
# ----------------------------------------------------------------
solver = LagrangianHydroSolver(;
    subproblem_solver  = JuMPSolver(HiGHS.Optimizer;
                                    options = Dict(:output_flag => false)),
    step_size          = DiminishingStep(1.0, 0.9),
    max_iter           = 50,
    n_scenarios        = 32,
    K_grid             = 48,
    dual_flow_coupling = true,
    dispatch           = :auto,
)

println("\nSolver configuration:")
println("  backend          : ", nameof(typeof(solver.backend)))
println("  dispatch (request): ", solver.dispatch, "  (:auto → :dp on GPU, :lp on CPU)")
println("  K_grid           : ", solver.K_grid)
println("  n_scenarios      : ", solver.n_scenarios)
println("  max_iter         : ", solver.max_iter)

# ----------------------------------------------------------------
# Solve.
# ----------------------------------------------------------------
println("\nSolving via LagrangianHydroSolver…")
sol = nothing
solve_time = @elapsed sol = solve(long_prob, solver)
@printf("  wall-clock: %.3f s\n", solve_time)

# ----------------------------------------------------------------
# Report.
# ----------------------------------------------------------------
println()
println("Termination: ", sol.status)
@printf("Objective:   %.6f  (Lagrangian dual; NOT comparable to HiGHS LP EUR)\n",
        sol.objective)
backend_used = get(sol.metadata, :ka_backend, nothing)
println("Backend used (resolved): ",
        backend_used === nothing ? "unknown" : string(nameof(typeof(backend_used))))
println("Dispatch (resolved):     ",
        get(sol.metadata, :dispatch, "unknown"))
println("Variant:                 ",
        get(sol.metadata, :variant, "unknown"))
if haskey(sol.metadata, :K_grid)
    @printf("K_grid:                  %d\n", sol.metadata[:K_grid])
end
if haskey(sol.metadata, :dual_flow_coupling)
    println("Dual flow coupling:      ", sol.metadata[:dual_flow_coupling])
end
hasproperty(sol, :iterations) && @printf("Outer iterations:        %d\n", sol.iterations)

# ----------------------------------------------------------------
# Primal / dual shapes — visible evidence of the DP code path.
# ----------------------------------------------------------------
if sol.primal !== nothing
    println("\nPrimal trajectories:")
    for k in keys(sol.primal)
        v = sol.primal[k]
        v isa AbstractArray && @printf("  %-22s  size=%s  eltype=%s\n",
                                       String(k), size(v), eltype(v))
    end
end
if sol.dual !== nothing
    println("\nDual variables:")
    for k in keys(sol.dual)
        v = sol.dual[k]
        v isa AbstractArray && @printf("  %-22s  size=%s  eltype=%s\n",
                                       String(k), size(v), eltype(v))
    end
end

# ----------------------------------------------------------------
# Quick comparison hint.
# ----------------------------------------------------------------
println()
println("To compare with the CPU HiGHS short-term LP on the same YAML:")
println("  julia --project=. scripts/run_pipeline.jl $ARGS_PATH $ARGS_MARKET")
println()
println("Remember: the two objectives are NOT comparable — see the header")
println("comment in this script for the full asymmetry note.")
