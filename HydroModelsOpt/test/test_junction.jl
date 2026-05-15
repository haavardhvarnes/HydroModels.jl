using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HiGHS
using OrderedCollections
using Tables
using Test

# ============================================================
# B1.7 — typed Junction (KP) with hard max_flow + soft min_flow
# ============================================================
#
# Fixture: `junction.yaml`. One upper reservoir feeds DemoPlant
# (turbine + bypass tunnel) discharging into a typed Junction
# (`TerminalKP`) with `max_flow = 50` (never binds) and `flow = 3`
# (regulatory minimum — soft constraint with high penalty).

const JUNCTION_PATH = joinpath(@__DIR__, "..", "..",
    "HydroModelsData", "test", "data", "junction.yaml")

@testset "Junction — parser populates the junction map" begin
    parsed = read_shop_yaml(JUNCTION_PATH)
    @test length(parsed.junctions) == 1
    @test haskey(parsed.junctions, "TerminalKP")
    j = parsed.junctions["TerminalKP"]
    @test j.upstream_elevation ≈ 100.0
    @test !isempty(j.max_flow)
    @test !isempty(j.flow)
    @test first(values(j.max_flow)) ≈ 50.0
    @test first(values(j.flow)) ≈ 3.0
end

@testset "Junction — spill-routing BFS reaches the junction" begin
    parsed = read_shop_yaml(JUNCTION_PATH)
    # B1.7 widening: terminal reservoir resolves to the junction.
    @test parsed.reservoirs["upper"].spill_to_reservoir == "TerminalKP"
end

@testset "Junction — LP exposes Qjunc table and enforces min flow" begin
    parsed = read_shop_yaml(JUNCTION_PATH)
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    @test sol.termination == "OPTIMAL"

    jc = Tables.columns(sol.junctions)
    @test length(jc.time) == 4
    @test all(==(("TerminalKP")), jc.junction)
    # Hard upper cap (50) is never binding; soft min (3) binds at
    # every timestep, so Q_junc ≈ 3 and slack ≈ 0.
    @test all(>=(2.99), jc.Q_junc)
    @test all(<=(50.0 + 1e-6), jc.Q_junc)
    @test all(<(1e-6), jc.R_min_slack)
end

@testset "Junction — slack absorbs unreachable min flow" begin
    # Bump min_flow far above any physically achievable rate so the
    # LP cannot satisfy it without slack. The LP must remain OPTIMAL
    # and the slack variable must take the deficit. This proves the
    # constraint is **soft** (LP doesn't go infeasible) rather than
    # hard.
    parsed = read_shop_yaml(JUNCTION_PATH)
    j_orig = parsed.junctions["TerminalKP"]
    # Replace min_flow with 1e5 m³/s — well above the maximum the
    # upper reservoir can drain in any single hour (s_max = 100 Mm³
    # → 100e6 m³ / 3600 s ≈ 2.8e4 m³/s, capacity bound on Qspill).
    # Keep max_flow at 1e8 so the cap is never binding.
    high_min = OrderedDict(t => 1.0e5 for t in keys(j_orig.flow))
    high_max = OrderedDict(t => 1.0e8 for t in keys(j_orig.max_flow))
    overshoot_junction = Junction{Float64}(
        name = j_orig.name, upstream_elevation = j_orig.upstream_elevation,
        max_flow = high_max, flow = high_min,
    )
    junctions = Dict("TerminalKP" => overshoot_junction)
    over_parsed = merge(parsed, (junctions = junctions,))
    sol = solve(ShopShortTermProblem(over_parsed),
        JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false)))

    @test sol.termination == "OPTIMAL"
    jc = Tables.columns(sol.junctions)
    # Slack must be substantial at every timestep — the LP physically
    # can't release 1e5 m³/s anywhere.
    @test all(>(1.0e4), jc.R_min_slack)
    # Definitional identity: Q_junc + slack ≥ min_flow = 1e5.
    @test all(jc.Q_junc[i] + jc.R_min_slack[i] >= 1.0e5 - 1.0e-3
              for i in eachindex(jc.time))
end

@testset "Junction — fixtures without junctions return empty table" begin
    # `minimal.yaml` has no declared junctions; `sol.junctions` is empty
    # but type-stable.
    path = joinpath(@__DIR__, "..", "..", "HydroModelsData",
                    "test", "data", "minimal.yaml")
    parsed = read_shop_yaml(path)
    @test isempty(parsed.junctions)
    sol = solve(ShopShortTermProblem(parsed),
        JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false)))
    @test sol.termination == "OPTIMAL"
    jc = Tables.columns(sol.junctions)
    @test isempty(jc.time)
    @test isempty(jc.Q_junc)
end
