using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HiGHS
using Tables
using Test

# ============================================================
# B1.5 — spill routing into the topological downstream module
# ============================================================
#
# Fixture: `cascade_spill.yaml`. A two-reservoir cascade where the
# upstream reservoir (`upper`) has small storage capacity, very large
# natural inflow, and bottlenecked outflow (one 5-m³/s turbine and one
# 5-m³/s bypass tunnel). The LP is forced to spill ~40 m³/s/h from
# `upper` every timestep. With the B1.5 routing change, that spill
# flows into `lower`'s mass balance instead of vanishing.

const CASCADE_SPILL_PATH = joinpath(@__DIR__, "..", "..",
    "HydroModelsData", "test", "data", "cascade_spill.yaml")

# Helper: rebuild the parsed reservoirs dict with all `spill_to_reservoir`
# fields forced to `nothing`. Lets us solve the same fixture under the
# "old sink behaviour" and compare against the routed solve.
function _strip_spill_routing(parsed)
    sunk = Dict{String, Reservoir{Float64}}()
    for (n, r) in parsed.reservoirs
        sunk[n] = Reservoir{Float64}(
            name               = r.name,
            vol_breaks         = r.vol_breaks,
            head_breaks        = r.head_breaks,
            s_min              = r.s_min,
            s_max              = r.s_max,
            s0                 = r.s0,
            inflow             = r.inflow,
            water_value        = r.water_value,
            spill_to_reservoir = nothing,
        )
    end
    return merge(parsed, (reservoirs = sunk,))
end

# Total spill at one reservoir name from a solution's spill table.
function _spill_at(sol, name::AbstractString)
    cols = Tables.columns(sol.spill)
    total = 0.0
    for i in eachindex(cols.Q_spill)
        String(cols.reservoir[i]) == name && (total += cols.Q_spill[i])
    end
    return total
end

# Storage at the last timestep for a given reservoir.
function _final_storage(sol, name::AbstractString)
    cols = Tables.columns(sol.storage)
    # the last row whose reservoir matches is the final-timestep entry
    last_val = 0.0
    for i in eachindex(cols.S)
        if String(cols.reservoir[i]) == name
            last_val = cols.S[i]
        end
    end
    return last_val
end

@testset "Spill routing — fixture parses with inferred destinations" begin
    parsed = read_shop_yaml(CASCADE_SPILL_PATH)
    @test parsed.reservoirs["upper"].spill_to_reservoir == "lower"
    @test parsed.reservoirs["lower"].spill_to_reservoir === nothing
end

@testset "Spill routing — `upper` is forced to spill" begin
    parsed = read_shop_yaml(CASCADE_SPILL_PATH)
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))
    @test sol.termination == "OPTIMAL"
    @test _spill_at(sol, "upper") > 1.0    # bottleneck binding ⇒ spill > 0
    @test _spill_at(sol, "lower") ≈ 0.0 atol = 1e-6
end

@testset "Spill routing — routed water reaches `lower`" begin
    parsed = read_shop_yaml(CASCADE_SPILL_PATH)
    prob_routed = ShopShortTermProblem(parsed)
    prob_sink   = ShopShortTermProblem(_strip_spill_routing(parsed))

    sol_routed = solve(prob_routed, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))
    sol_sink   = solve(prob_sink,   JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    @test sol_routed.termination == "OPTIMAL"
    @test sol_sink.termination   == "OPTIMAL"

    # With routing, the spilled water arrives in `lower` — its final
    # storage must be strictly higher than the sink-only run.
    @test _final_storage(sol_routed, "lower") >
          _final_storage(sol_sink,   "lower") + 1e-3

    # Spill amount at `upper` is the same in both runs (the LP can't
    # avoid it because the bottleneck is upstream capacity); the
    # difference is purely *where the water goes after spilling*.
    @test _spill_at(sol_routed, "upper") ≈ _spill_at(sol_sink, "upper") atol = 1e-3

    # And the routed run's objective must be at least as good as the
    # sink run — the LP now has the option to keep that water for
    # downstream use, never worse.
    @test sol_routed.objective >= sol_sink.objective - 1e-6
end

@testset "Spill routing — existing fixtures unchanged by routing" begin
    # `minimal.yaml` and `pumped_storage.yaml` have inferred
    # `upper → lower` routing for upper, terminal (`nothing`) for lower.
    # `minimal` produces no spill in its optimum; `pumped_storage` does
    # spill at `lower` but that's a terminal node ⇒ same sink behaviour
    # either way. Either fixture's objective must therefore be
    # bit-identical between the routed (default) and the
    # spill-stripped (all `spill_to_reservoir = nothing`) solves.
    for fixture in ("minimal.yaml", "pumped_storage.yaml")
        path = joinpath(@__DIR__, "..", "..",
            "HydroModelsData", "test", "data", fixture)
        parsed = read_shop_yaml(path)
        prob_routed = ShopShortTermProblem(parsed)
        prob_sink   = ShopShortTermProblem(_strip_spill_routing(parsed))

        sol_routed = solve(prob_routed, JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)))
        sol_sink   = solve(prob_sink,   JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)))

        @test sol_routed.termination == "OPTIMAL"
        @test sol_sink.termination   == "OPTIMAL"
        @test sol_routed.objective ≈ sol_sink.objective atol = 1e-3
    end
end
