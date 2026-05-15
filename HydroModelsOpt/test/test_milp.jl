using HydroModelsCore
using HydroModelsOpt
using HydroModelsData
using HiGHS
using Tables
using Dates
using Test

const PUMPED_FIXTURE = joinpath(
    dirname(@__DIR__), "..",
    "HydroModelsData", "test", "data", "pumped_storage.yaml",
)

@testset "MILP fixture exists" begin
    @test isfile(PUMPED_FIXTURE)
end

@testset "is_milp detection" begin
    @testset "pure LP fixture has no MILP path" begin
        plain = joinpath(dirname(@__DIR__), "..", "HydroModelsData",
                         "test", "data", "minimal.yaml")
        parsed = read_shop_yaml(plain; market_name = "NO3")
        prob = ShopShortTermProblem(parsed)
        @test !is_milp(prob)
    end

    @testset "reservoirs without pumps -> LP" begin
        # Reserve fixture has a generator and no pump, so it stays LP
        # even though it has reserve products.
        plain = joinpath(dirname(@__DIR__), "..", "HydroModelsData",
                         "test", "data", "minimal_with_reserves.yaml")
        parsed = read_shop_yaml(plain; market_name = "NO3")
        prob = ShopShortTermProblem(parsed)
        @test !is_milp(prob)
    end

    @testset "reversible plant -> MILP" begin
        parsed = read_shop_yaml(PUMPED_FIXTURE; market_name = "NO3")
        prob = ShopShortTermProblem(parsed)
        @test is_milp(prob)
    end
end

@testset "Pumped-storage fixture: structural parse" begin
    parsed = read_shop_yaml(PUMPED_FIXTURE; market_name = "NO3")

    @test length(parsed.plants) == 1
    @test length(parsed.generators) == 1
    @test length(parsed.pumps) == 1

    g = parsed.generators[1]
    p = parsed.pumps[1]
    @test g.plant == p.plant   # both at the same plant -> reversible

    # Plant inference: generator flows upper -> lower, pump flows
    # lower -> upper (opposite direction).
    @test g.from_res == ["upper"] && g.to_res == "lower"
    @test p.from_res == ["lower"] && p.to_res == "upper"
end

@testset "MILP solve terminates OPTIMAL" begin
    parsed = read_shop_yaml(PUMPED_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))
    @test sol.termination == "OPTIMAL"
    @test isfinite(sol.objective)
end

@testset "Reversible-plant gen-XOR-pump per timestep" begin
    parsed = read_shop_yaml(PUMPED_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    # Build (time -> gen_MW), (time -> pump_MW) maps
    pg = Tables.columns(sol.p_gen)
    pp = Tables.columns(sol.e_pump)
    gen_by_t = Dict(zip(pg.time, pg.P))
    pump_by_t = Dict(zip(pp.time, pp.E))

    # At every timestep, at most one of (gen_MW, pump_MW) is non-zero.
    # The big-M constraint Pg <= M*mode + Ppump <= M*(1-mode) makes
    # this binary; we allow a tiny solver tolerance.
    eps = 1.0e-6
    for t in keys(gen_by_t)
        g = gen_by_t[t]
        p = get(pump_by_t, t, 0.0)
        @test g <= eps || p <= eps
    end
end

@testset "LP exploits price arbitrage (pumps cheap, generates peak)" begin
    parsed = read_shop_yaml(PUMPED_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    pg = Tables.columns(sol.p_gen)
    pp = Tables.columns(sol.e_pump)
    rev = Tables.columns(sol.revenue)

    gen_by_t = Dict(zip(pg.time, pg.P))
    pump_by_t = Dict(zip(pp.time, pp.E))
    price_by_t = Dict(zip(rev.time, rev.price))

    # At the cheapest hour (price = 20), the LP should be PUMPING.
    cheapest_t = argmin(t -> price_by_t[t], keys(price_by_t))
    @test get(pump_by_t, cheapest_t, 0.0) > 1.0   # >1 MW pump activity
    @test get(gen_by_t, cheapest_t, 0.0) < 1.0e-6 # not generating

    # At the peak hour (price = 80), the LP should be GENERATING.
    peak_t = argmax(t -> price_by_t[t], keys(price_by_t))
    @test get(gen_by_t, peak_t, 0.0) > 1.0
    @test get(pump_by_t, peak_t, 0.0) < 1.0e-6
end

@testset "Storage trajectory consistent with arbitrage" begin
    parsed = read_shop_yaml(PUMPED_FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    sc = Tables.columns(sol.storage)
    upper_idx = findall(==("upper"), sc.reservoir)
    upper_S = sc.S[upper_idx]
    upper_t = sc.time[upper_idx]
    perm = sortperm(upper_t)
    upper_S = upper_S[perm]

    # Upper storage should peak shortly after pumping (the cheapest
    # hour) and decline through subsequent generating periods.
    # Concretely: max storage is in the first two entries (not the last).
    @test argmax(upper_S) <= 2
end
