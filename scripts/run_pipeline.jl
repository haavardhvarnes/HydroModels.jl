#!/usr/bin/env julia
#
# Parse a SHOP / Harmonie YAML file, build the LP, solve it with HiGHS,
# and print a summary of objective + key totals + table sizes.
#
# Usage:
#   julia --project=. scripts/run_pipeline.jl <path-to-yaml> [<market_name>]
#
# Examples:
#   julia --project=. scripts/run_pipeline.jl HydroModelsData/test/data/minimal.yaml
#   julia --project=. scripts/run_pipeline.jl ../HydroModels_depr/data/real/dev_test\ 2.yaml NO3

using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HiGHS
using Tables
using Printf

const ARGS_PATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..", "HydroModelsData", "test", "data", "minimal.yaml")
const ARGS_MARKET = length(ARGS) >= 2 ? ARGS[2] : "NO3"

isfile(ARGS_PATH) || error("File not found: $ARGS_PATH")

println("Parsing $ARGS_PATH (market=$ARGS_MARKET)…")
@time parsed = read_shop_yaml(ARGS_PATH; market_name = ARGS_MARKET)
println("  plants=$(length(parsed.plants))  reservoirs=$(length(parsed.reservoirs))  ",
        "generators=$(length(parsed.generators))  pumps=$(length(parsed.pumps))  ",
        "tunnels=$(length(parsed.tunnels))  cut_groups=$(length(parsed.cut_groups))  ",
        "reserve_groups=$(length(parsed.reserve_groups))")

prob = ShopShortTermProblem(parsed)

println("\nBuilding and solving LP…")
@time sol = solve(prob, JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false)))

println()
println("Termination: ", sol.termination)
@printf("Objective:   %.6f MEUR\n", sol.objective / 1e6)

# Energy revenue
rev_cols = Tables.columns(sol.revenue)
total_rev = sum(rev_cols.revenue) / 1e6
total_gen_MWh = sum(rev_cols.gen_MWh)
total_pump_MWh = sum(rev_cols.pump_MWh)
@printf("Energy revenue: %.6f MEUR   gen=%.2f MWh   pump=%.2f MWh\n",
        total_rev, total_gen_MWh, total_pump_MWh)

# Spill split into two operational categories:
#
#   1. In-cascade spill  — Qspill at reservoirs whose
#      `spill_to_reservoir` resolves to another **reservoir** (or is
#      `nothing`). This is dam-overtopping spill within the modelled
#      system, what an operator would normally call "spill".
#
#   2. Junction outflow   — Qspill at reservoirs whose
#      `spill_to_reservoir` resolves to a **typed Junction** (system
#      boundary, e.g. BergheimKP / VangenSeaLevelKP at the end of
#      Hallingdalselvi / Aurlandselvi). This is regulatory release
#      water leaving the modelled domain. NOT operational spill —
#      a separate, named category in the diagnostic.
spill_cols = Tables.columns(sol.spill)
junction_names = Set(keys(parsed.junctions))
internal_spill_per_res = Dict{String, Float64}()
junction_outflow_per_res = Dict{String, Float64}()
for i in eachindex(spill_cols.Q_spill)
    q = spill_cols.Q_spill[i]
    q > 0.0 || continue
    rname = String(spill_cols.reservoir[i])
    dest = haskey(parsed.reservoirs, rname) ?
           parsed.reservoirs[rname].spill_to_reservoir : nothing
    if dest !== nothing && dest in junction_names
        junction_outflow_per_res[rname] = get(junction_outflow_per_res, rname, 0.0) + q
    else
        internal_spill_per_res[rname] = get(internal_spill_per_res, rname, 0.0) + q
    end
end
total_internal = sum(values(internal_spill_per_res); init = 0.0)
total_junc_out = sum(values(junction_outflow_per_res); init = 0.0)
@printf("In-cascade spill   : %.4f m³/s·timesteps\n", total_internal)
@printf("Junction outflow   : %.4f m³/s·timesteps  (system boundary, not spill)\n", total_junc_out)

function _print_top(per_res::Dict{String, Float64}, total::Float64, header::AbstractString)
    isempty(per_res) && return
    ranked = sort(collect(per_res); by = last, rev = true)
    println(header)
    cumulative = 0.0
    printed = 0
    for (r, q) in ranked
        q < 1e-3 && break
        cumulative += q
        share = 100 * q / total
        @printf("  %-30s %12.4f m³/s·tsteps  (%.1f %%)\n", r, q, share)
        printed += 1
        cumulative / total >= 0.99 && break
    end
    printed == 0 && println("  (none ≥ 1e-3 m³/s·timesteps)")
    return
end

if total_internal > 1e-6
    _print_top(internal_spill_per_res, total_internal,
        "Top in-cascade spillers (≥ 1e-3 m³/s·timesteps, capped at 99 % cumulative):")
end
if total_junc_out > 1e-6
    _print_top(junction_outflow_per_res, total_junc_out,
        "Top junction-outflow contributors (water leaving system, not spill):")
end

# Reserve allocation summary
r_alloc_cols = Tables.columns(sol.r_alloc)
if !isempty(r_alloc_cols.time)
    products = unique(r_alloc_cols.product)
    println("\nReserve allocation by product:")
    for p in sort(products)
        total = sum(r_alloc_cols.R_MW[i] for i in eachindex(r_alloc_cols.product)
                    if r_alloc_cols.product[i] == p)
        @printf("  %-12s  %.2f MW (sum across units × timesteps)\n", p, total)
    end
end

# Reserve group shortfalls (non-zero)
r_slack_cols = Tables.columns(sol.r_slack)
if !isempty(r_slack_cols.time)
    nonzero = findall(>(1e-6), r_slack_cols.slack_MW)
    if !isempty(nonzero)
        println("\nReserve group shortfalls (non-zero):")
        for i in nonzero
            @printf("  %s  %s  %.4f MW\n",
                    r_slack_cols.time[i], r_slack_cols.group[i], r_slack_cols.slack_MW[i])
        end
    end
end

# Junction (B1.7) — per-junction flow + soft-min slack summary.
junc_cols = Tables.columns(sol.junctions)
if !isempty(junc_cols.time)
    println("\nJunction (B1.7) flow summary:")
    by_name = Dict{String, Tuple{Float64, Float64, Float64, Float64}}()
    for i in eachindex(junc_cols.time)
        n = String(junc_cols.junction[i])
        q = junc_cols.Q_junc[i]
        s = junc_cols.R_min_slack[i]
        cur = get(by_name, n, (0.0, 0.0, 0.0, 0.0))
        # totals: (sum Q, sum slack, max Q, max slack)
        by_name[n] = (cur[1] + q, cur[2] + s,
                      max(cur[3], q), max(cur[4], s))
    end
    @printf("  %-20s  %12s  %12s  %12s  %12s\n",
            "junction", "ΣQ (m³/s·ts)", "max Q", "Σslack", "max slack")
    for (n, (sumQ, sumS, maxQ, maxS)) in sort(collect(by_name); by = first)
        @printf("  %-20s  %12.2f  %12.2f  %12.2f  %12.2f\n",
                n, sumQ, maxQ, sumS, maxS)
    end
end

# Water values
wv_cols = Tables.columns(sol.water_value)
if haskey(wv_cols, :group) && !isempty(wv_cols.group)
    println("\nEnd-of-horizon water values (cut groups):")
    for i in eachindex(wv_cols.group)
        @printf("  %-20s  %.6f MEUR\n", wv_cols.group[i], wv_cols.W_EUR[i] / 1e6)
    end
elseif haskey(wv_cols, :reservoir) && !isempty(wv_cols.reservoir)
    println("\nEnd-of-horizon water values (per-reservoir):")
    for i in eachindex(wv_cols.reservoir)
        @printf("  %-20s  %.6f MEUR\n", wv_cols.reservoir[i], wv_cols.W_EUR[i] / 1e6)
    end
end

println()
println("Table sizes (rows):")
@printf("  storage:     %d\n", length(Tables.columns(sol.storage).time))
@printf("  p_gen:       %d\n", length(Tables.columns(sol.p_gen).time))
@printf("  q_gen:       %d\n", length(Tables.columns(sol.q_gen).time))
@printf("  q_pump:      %d\n", length(Tables.columns(sol.q_pump).time))
@printf("  q_tunnel:    %d\n", length(Tables.columns(sol.q_tunnel).time))
@printf("  spill:       %d\n", length(Tables.columns(sol.spill).time))
@printf("  revenue:     %d\n", length(Tables.columns(sol.revenue).time))
@printf("  r_alloc:     %d\n", length(Tables.columns(sol.r_alloc).time))
@printf("  r_slack:     %d\n", length(Tables.columns(sol.r_slack).time))
