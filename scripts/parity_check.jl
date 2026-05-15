#!/usr/bin/env julia
#
# Run the new HydroModels LP baseline against every YAML in
# `HYDROMODELS_TEST_DATA_DIR` (or `../HydroModels_depr/data/real` by
# default) and report objective / generation / spill / water-value
# totals for manual comparison against depr's known outputs.
#
# This is *not* an automated parity assertion. depr's pipeline isn't
# loaded by this script (it pulls in DataFrames + PlotlyJS which we
# don't want as test deps). Run depr separately to obtain its numbers
# and eyeball the deltas; A3's tolerance target is < 1 % on the
# objective.
#
# Usage:
#   julia --project=. scripts/parity_check.jl
#   HYDROMODELS_TEST_DATA_DIR=/path/to/yamls julia --project=. scripts/parity_check.jl
#   julia --project=. scripts/parity_check.jl NO5    # market name override

using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HiGHS
using Tables
using Printf

const MARKET = length(ARGS) >= 1 ? ARGS[1] : "NO5"
const DEFAULT_DIR = joinpath(@__DIR__, "..", "..", "HydroModels_depr", "data", "real")
const DATA_DIR = get(ENV, "HYDROMODELS_TEST_DATA_DIR", DEFAULT_DIR)

if !isdir(DATA_DIR)
    @warn "Data directory not found: $DATA_DIR. Set HYDROMODELS_TEST_DATA_DIR or pass a path."
    exit(1)
end

yamls = filter(f -> endswith(lowercase(f), ".yaml"), readdir(DATA_DIR; join = true))
isempty(yamls) && (@warn "No .yaml files in $DATA_DIR"; exit(1))

println("Parity sweep over $(length(yamls)) YAML(s) in $DATA_DIR")
println("Market: $MARKET")
println()

@printf("%-58s  %12s  %12s  %12s  %12s  %12s  %12s  %12s\n",
        "case", "termination", "obj_MEUR", "gen_MWh", "spill",
        "reserve_MW", "slack_MW", "elapsed_s")
println(repeat("-", 162))

for path in yamls
    name = basename(path)
    try
        t0 = time()
        parsed = read_shop_yaml(path; market_name = MARKET)
        prob = ShopShortTermProblem(parsed)
        sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
                                     options = Dict(:output_flag => false)))
        elapsed = time() - t0

        rev_cols = Tables.columns(sol.revenue)
        spill_cols = Tables.columns(sol.spill)
        gen_MWh = sum(rev_cols.gen_MWh)
        total_spill = isempty(spill_cols.Q_spill) ? 0.0 : sum(spill_cols.Q_spill)

        # Reserve totals (MW·timesteps summed across units × products)
        r_alloc_cols = Tables.columns(sol.r_alloc)
        total_reserve = isempty(r_alloc_cols.R_MW) ? 0.0 : sum(r_alloc_cols.R_MW)
        r_slack_cols = Tables.columns(sol.r_slack)
        total_slack = isempty(r_slack_cols.slack_MW) ? 0.0 : sum(r_slack_cols.slack_MW)

        @printf("%-58s  %12s  %12.4f  %12.2f  %12.2f  %12.2f  %12.4f  %12.1f\n",
                first(name, 58),
                sol.termination,
                sol.objective / 1e6,
                gen_MWh,
                total_spill,
                total_reserve,
                total_slack,
                elapsed)
    catch err
        @printf("%-58s  ERROR: %s\n", first(name, 58), sprint(showerror, err))
    end
end
