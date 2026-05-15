using HydroModelsCore
using HydroModelsOpt
using HydroModelsData
using HiGHS
using Tables
using Dates
using Test

const RESERVE_FIXTURE = joinpath(
    dirname(@__DIR__), "..",
    "HydroModelsData", "test", "data", "minimal_with_reserves.yaml",
)

@testset "Reserve fixture is reachable" begin
    @test isfile(RESERVE_FIXTURE)
end

@testset "Parser extracts reserve specs" begin
    parsed = read_shop_yaml(RESERVE_FIXTURE; market_name = "NO3")

    @test length(parsed.generators) == 1
    g = parsed.generators[1]
    @test length(g.reserves) == 1
    rs = g.reserves[1]
    @test rs.product == "FCR_N_UP"
    @test rs.pmin == 0.0
    @test rs.pmax == 5.0

    @test length(parsed.reserve_groups) == 1
    rg = parsed.reserve_groups[1]
    @test rg.name == "NO3_FCR_N_UP_Group"
    @test rg.product == "FCR_N_UP"
    @test length(rg.obligation) == 4   # obligations at every hour
    @test all(==(3.0), values(rg.obligation))
    @test all(==(1000.0), values(rg.penalty_cost))

    @test haskey(parsed.reserve_prices, "FCR_N_UP")
    rp = parsed.reserve_prices["FCR_N_UP"]
    @test length(rp) == 2
end

@testset "LP solves reserve fixture to optimality" begin
    parsed = read_shop_yaml(RESERVE_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    @test sol.termination == "OPTIMAL"
    @test isfinite(sol.objective)
end

@testset "Reserve allocation meets group obligation" begin
    parsed = read_shop_yaml(RESERVE_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    # 1 generator × 4 timesteps × 1 product = 4 rows
    cols = Tables.columns(sol.r_alloc)
    @test length(cols.time) == 4
    @test all(==("DemoPlant_|_G1"), cols.generator)
    @test all(==("FCR_N_UP"), cols.product)
    # Each timestep should allocate exactly 3 MW (the obligation).
    # Generator pmax for FCR_N_UP is 5; obligation is 3; reserve price
    # is positive but smaller than the marginal energy price → LP picks
    # the minimum allocation that satisfies the group obligation.
    @test all(isapprox.(cols.R_MW, 3.0; atol = 1.0e-3))
end

@testset "No slack — obligation is exactly satisfied" begin
    parsed = read_shop_yaml(RESERVE_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    slack_cols = Tables.columns(sol.r_slack)
    @test length(slack_cols.time) == 4
    @test all(==("NO3_FCR_N_UP_Group/FCR_N_UP"), slack_cols.group)
    @test all(isapprox.(slack_cols.slack_MW, 0.0; atol = 1.0e-6))
end

@testset "Reserve revenue contributes to objective" begin
    parsed = read_shop_yaml(RESERVE_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    # Reserve revenue = sum over (u, j, p) of (price[p, j] * dt[j] * R)
    # = 3 MW × 1h × (5 + 5 + 8 + 8) EUR/MW = 78 EUR exactly.
    energy_rev = sum(sol.revenue.revenue)
    expected_reserve_rev = 3.0 * (5.0 + 5.0 + 8.0 + 8.0)  # 78 EUR
    expected_objective = energy_rev + expected_reserve_rev
    @test isapprox(sol.objective, expected_objective; atol = 1.0)
end

@testset "FOS §8 joint headroom — Pg + R ≤ pmax per timestep" begin
    parsed = read_shop_yaml(RESERVE_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    # Reconstruct per-time Pg and R, check Pg + R ≤ pmax (60 MW) with
    # a generous tolerance. FCR_N_UP is symmetric so it also counts on
    # the down side (Pg - R ≥ 0), but that's just non-negativity here.
    p_gen_by_t = Dict(zip(sol.p_gen.time, sol.p_gen.P))
    r_alloc_by_t = Dict(zip(sol.r_alloc.time, sol.r_alloc.R_MW))
    pmax = 60.0
    for t in keys(p_gen_by_t)
        pg = p_gen_by_t[t]
        r = get(r_alloc_by_t, t, 0.0)
        @test pg + r <= pmax + 1.0e-3
    end
end

@testset "No-reserve fixture has empty reserve tables" begin
    # Sanity check that the original minimal.yaml (no reserves) doesn't
    # accidentally produce reserve rows.
    plain = joinpath(dirname(@__DIR__), "..", "HydroModelsData",
                     "test", "data", "minimal.yaml")
    parsed = read_shop_yaml(plain; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    @test isempty(Tables.columns(sol.r_alloc).time)
    @test isempty(Tables.columns(sol.r_slack).time)
end
