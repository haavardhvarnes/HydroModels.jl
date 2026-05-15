using HydroModelsCore
using HydroModelsOpt
using HydroModelsData
using HiGHS
using Tables
using Dates
using Test

const FIXTURE = joinpath(
    dirname(@__DIR__), "..",   # opt/test -> opt -> repo root
    "HydroModelsData", "test", "data", "minimal.yaml",
)

@testset "Fixture is reachable from Opt tests" begin
    @test isfile(FIXTURE)
end

@testset "End-to-end: parse → ShopShortTermProblem → solve" begin
    parsed = read_shop_yaml(FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)

    @test prob isa ShopShortTermProblem{Float64}
    @test length(prob.reservoirs) == 2
    @test length(prob.generators) == 1
    @test length(prob.tunnels) == 1

    solver = JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false))
    sol = solve(prob, solver)

    @testset "Termination and objective" begin
        @test sol isa HydroSolution{Float64}
        @test sol.termination == "OPTIMAL"
        @test isfinite(sol.objective)
        # Minimal fixture: generator pmax=60, prices ~30-60 EUR/MWh over 24h.
        # A naive upper bound is 60 MW × 24 h × 60 EUR/MWh = 86 400 EUR.
        # Storage / inflow limits and end-of-horizon constraints make the
        # actual optimum lower; check the order of magnitude.
        @test 0.0 < sol.objective < 100_000.0
    end

    @testset "Tables.jl interface on solution tables" begin
        for tbl in (sol.storage, sol.spill, sol.q_gen, sol.p_gen,
                    sol.q_pump, sol.e_pump, sol.q_tunnel, sol.revenue,
                    sol.r_alloc, sol.r_slack, sol.water_value)
            @test Tables.istable(tbl)
        end
    end

    @testset "Storage table shape" begin
        cols = Tables.columns(sol.storage)
        @test :time in keys(cols)
        @test :reservoir in keys(cols)
        @test :S in keys(cols)
        @test :H in keys(cols)
        @test length(cols.time) == length(cols.S)
        # 2 reservoirs × however many timesteps in the union time grid
        @test length(cols.time) > 0
        @test all(>=(0), cols.S)  # storage non-negative
        # Initial storage should match the fixture
        # (upper: 50.0, lower: 80.0)
        upper_first_idx = findfirst(==("upper"), cols.reservoir)
        lower_first_idx = findfirst(==("lower"), cols.reservoir)
        @test cols.S[upper_first_idx] ≈ 50.0 atol = 1e-6
        @test cols.S[lower_first_idx] ≈ 80.0 atol = 1e-6
    end

    @testset "Generation table shape" begin
        cols = Tables.columns(sol.p_gen)
        @test :time in keys(cols)
        @test :generator in keys(cols)
        @test :P in keys(cols)
        @test all(==("DemoPlant_|_G1"), cols.generator)
        @test all(>=(-1e-6), cols.P)   # non-negative within solver tol
        @test all(<=(60.0 + 1e-6), cols.P)  # bounded by pmax
        @test sum(cols.P) > 0   # solver actually generates something
    end

    @testset "Pump tables are empty (no pumps in fixture)" begin
        @test isempty(Tables.columns(sol.q_pump).time)
        @test isempty(Tables.columns(sol.e_pump).time)
    end

    @testset "Tunnel table has the bypass" begin
        cols = Tables.columns(sol.q_tunnel)
        @test all(==("w_upper_lower_bypass"), cols.tunnel)
        @test all(>=(-1e-6), cols.Q)
        @test all(<=(30.0 + 1e-6), cols.Q)  # tunnel qmax
    end

    @testset "Revenue table" begin
        cols = Tables.columns(sol.revenue)
        @test :time in keys(cols)
        @test :price in keys(cols)
        @test :gen_MWh in keys(cols)
        @test :pump_MWh in keys(cols)
        @test :revenue in keys(cols)
        # Total revenue should match objective minus the tiny spill penalty
        # and any reserve / water value contributions (none here).
        @test sum(cols.revenue) ≈ sol.objective rtol = 1e-3
    end

    @testset "Reserve / water-value tables empty for no-reserve fixture" begin
        @test isempty(Tables.columns(sol.r_alloc).time)
        @test isempty(Tables.columns(sol.r_slack).time)
        # No water value cuts in minimal.yaml → empty per-reservoir form
        wv_cols = Tables.columns(sol.water_value)
        @test :reservoir in keys(wv_cols)
        @test isempty(wv_cols.reservoir)
    end
end

@testset "solve with no optimizer is build-only" begin
    parsed = read_shop_yaml(FIXTURE; market_name = "NO3")
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(nothing))
    @test sol.termination == "NOT_CALLED"
    @test isnan(sol.objective)
end

@testset "default_lp_solver" begin
    # Bare HydroModelsOpt: returns JuMPSolver(nothing)
    bare = default_lp_solver()
    @test bare isa JuMPSolver
    # With HiGHS loaded, the extension should have re-defined it.
    highs = default_lp_solver()
    @test highs.optimizer === HiGHS.Optimizer
end
