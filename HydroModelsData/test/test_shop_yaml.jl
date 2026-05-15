using HydroModelsCore
using HydroModelsData
using Tables
using Dates
using Test

const FIXTURE = joinpath(@__DIR__, "data", "minimal.yaml")

@testset "Fixture exists" begin
    @test isfile(FIXTURE)
end

@testset "read_shop_yaml on minimal fixture" begin
    parsed = read_shop_yaml(FIXTURE; market_name = "NO3")

    @testset "Returns NamedTuple with all keys" begin
        for k in (:plants, :reservoirs, :generators, :pumps, :tunnels,
                  :market, :dt_schedule, :reserve_groups,
                  :reserve_prices, :cut_groups)
            @test haskey(parsed, k)
        end
    end

    @testset "Plants" begin
        @test length(parsed.plants) == 1
        @test haskey(parsed.plants, "DemoPlant")
        plant = parsed.plants["DemoPlant"]
        @test plant isa Plant{Float64}
        @test plant.name == "DemoPlant"
        @test plant.outlet_line == 100.0
        @test plant.main_loss == [0.0001]
        @test plant.penstock_loss == [0.0001]
        @test isempty(plant.maintenance_flag)
        @test isempty(plant.max_p_constr)
    end

    @testset "Reservoirs" begin
        @test length(parsed.reservoirs) == 2
        upper = parsed.reservoirs["upper"]
        @test upper isa Reservoir{Float64}
        @test upper.name == "upper"
        @test upper.vol_breaks == [0.0, 50.0, 100.0]
        @test upper.head_breaks == [200.0, 220.0, 240.0]
        @test upper.s0 == 50.0
        @test upper.s_min == 0.0   # lrl=200 → vol via vol_head curve → 0
        @test upper.s_max == 100.0 # hrl=240 → vol=100
        @test length(upper.inflow) == 2
        @test upper.inflow.by_time[DateTime("2024-01-01T00:00:00")] == 5.0
        @test upper.inflow.by_time[DateTime("2024-01-01T12:00:00")] == 10.0
        @test upper.water_value === nothing

        lower = parsed.reservoirs["lower"]
        @test lower.s0 == 80.0
        @test lower.s_min == 0.0
        @test lower.s_max == 200.0

        # Spill routing inferred from generator topology (B1.5).
        @test upper.spill_to_reservoir == "lower"
        @test lower.spill_to_reservoir === nothing
    end

    @testset "Generator and inferred connection" begin
        @test length(parsed.generators) == 1
        gen = parsed.generators[1]
        @test gen isa Generator{Float64}
        @test gen.name == "DemoPlant_|_G1"
        @test gen.plant == "DemoPlant"
        @test gen.from_res == ["upper"]   # inferred via tunnel BFS (vector form post-B1.6)
        @test gen.to_res == "lower"       # inferred via tunnel BFS
        @test length(gen.curves) == 1
        href, curve = gen.curves[1]
        @test href == 130.0
        @test curve isa PiecewiseLinear{Float64}
        # _ensure_origin prepends (0,0) only if first x != 0; curve
        # already starts at 0, so length stays 3.
        @test length(curve) == 3
        @test gen.qmin == 0.0
        @test gen.qmax == 50.0
        @test gen.p_nom == 60.0
        @test gen.penstock == 1
        @test gen.gen_eff === nothing
        @test isempty(gen.reserves)   # fixture has no reserve fields
    end

    @testset "Tunnel emission (bypass only)" begin
        # Penstock tunnels (reservoir <-> plant) collapse to self-loops
        # after plant→reservoir resolution and are skipped. Only the
        # direct upper→lower bypass survives.
        @test length(parsed.tunnels) == 1
        tun = parsed.tunnels[1]
        @test tun isa Tunnel{Float64}
        @test tun.name == "w_upper_lower_bypass"
        @test tun.from_node == "upper"
        @test tun.to_node == "lower"
        @test tun.qmax == 30.0
    end

    @testset "Market series" begin
        @test parsed.market isa MarketSeries{Float64}
        @test length(parsed.market) == 4
        @test parsed.market.by_time[DateTime("2024-01-01T00:00:00")] == 30.0
        @test parsed.market.by_time[DateTime("2024-01-01T12:00:00")] == 60.0
    end

    @testset "dt schedule" begin
        @test parsed.dt_schedule == [(DateTime("2024-01-01T00:00:00"), 60.0)]
    end

    @testset "Empty optional collections" begin
        @test isempty(parsed.pumps)
        @test isempty(parsed.reserve_groups)
        @test isempty(parsed.reserve_prices)
        @test isempty(parsed.cut_groups)
    end
end

@testset "Tables.jl interface" begin
    parsed = read_shop_yaml(FIXTURE; market_name = "NO3")

    @testset "MarketSeries is Tables-compatible" begin
        @test Tables.istable(parsed.market)
        cols = Tables.columns(parsed.market)
        @test cols.time isa Vector{DateTime}
        @test cols.price isa Vector{Float64}
        @test length(cols.time) == length(cols.price) == 4
        sch = Tables.schema(parsed.market)
        @test sch.names == (:time, :price)
        @test sch.types == (DateTime, Float64)
    end

    @testset "InflowSeries is Tables-compatible" begin
        upper_inflow = parsed.reservoirs["upper"].inflow
        @test Tables.istable(upper_inflow)
        cols = Tables.columns(upper_inflow)
        @test cols.time isa Vector{DateTime}
        @test cols.inflow isa Vector{Float64}
        @test length(cols.time) == 2
    end

    @testset "inflow_table builder (long format)" begin
        tab = inflow_table(parsed.reservoirs)
        @test tab.time isa Vector{DateTime}
        @test tab.reservoir isa Vector{String}
        @test tab.inflow isa Vector{Float64}
        # 2 from upper + 1 from lower
        @test length(tab.time) == 3
        @test "upper" in tab.reservoir
        @test "lower" in tab.reservoir
    end

    @testset "market_table builder" begin
        tab = market_table(parsed.market)
        @test tab.time isa Vector{DateTime}
        @test tab.price isa Vector{Float64}
    end

    @testset "reserve_obligation_table on empty groups" begin
        tab = reserve_obligation_table(parsed.reserve_groups)
        @test isempty(tab.time)
        @test isempty(tab.group)
        @test isempty(tab.product)
        @test isempty(tab.obligation)
    end
end

const RESERVE_FIXTURE = joinpath(@__DIR__, "data", "minimal_with_reserves.yaml")

@testset "Reserve-aware fixture parses" begin
    @test isfile(RESERVE_FIXTURE)
    parsed = read_shop_yaml(RESERVE_FIXTURE; market_name = "NO3")

    @testset "Generator reserve spec round-trip" begin
        g = parsed.generators[1]
        @test length(g.reserves) == 1
        @test g.reserves[1].product == "FCR_N_UP"
        @test g.reserves[1].pmax == 5.0
    end

    @testset "Reserve group with 4 obligation timestamps" begin
        @test length(parsed.reserve_groups) == 1
        rg = parsed.reserve_groups[1]
        @test rg.name == "NO3_FCR_N_UP_Group"
        @test rg.product == "FCR_N_UP"
        @test length(rg.obligation) == 4
        @test all(==(3.0), values(rg.obligation))
        @test length(rg.penalty_cost) == 4
        @test all(==(1000.0), values(rg.penalty_cost))
    end

    @testset "Reserve market sale price block" begin
        @test haskey(parsed.reserve_prices, "FCR_N_UP")
        @test length(parsed.reserve_prices["FCR_N_UP"]) == 2
    end

    @testset "reserve_obligation_table on non-empty groups" begin
        tab = reserve_obligation_table(parsed.reserve_groups)
        @test length(tab.time) == 4
        @test all(==("NO3_FCR_N_UP_Group"), tab.group)
        @test all(==("FCR_N_UP"), tab.product)
        @test all(==(3.0), tab.obligation)
    end
end

const PUMPED_FIXTURE = joinpath(@__DIR__, "data", "pumped_storage.yaml")

@testset "Pumped-storage fixture parses" begin
    @test isfile(PUMPED_FIXTURE)
    parsed = read_shop_yaml(PUMPED_FIXTURE; market_name = "NO3")

    @test length(parsed.generators) == 1
    @test length(parsed.pumps) == 1

    g = parsed.generators[1]
    p = parsed.pumps[1]
    @test g.plant == "DemoPlant"
    @test p.plant == "DemoPlant"
    @test g.from_res == ["upper"]
    @test g.to_res == "lower"
    # Pump direction is the reverse of the generator (inferred via
    # _infer_pump_connection in the parser).
    @test p.from_res == ["lower"]
    @test p.to_res == "upper"
    # Pump has its own PQ curve
    @test length(p.curves) == 1
    @test p.qmax == 25.0
    @test p.pmax == 31.25
end

# Optional: parity-check entry point when HYDROMODELS_TEST_DATA_DIR points
# at the depr NO5 cases. Skips gracefully in CI / when the env var is
# unset. Only verifies the parser runs without throwing on real data.
@testset "Real-data smoke test (opt-in via HYDROMODELS_TEST_DATA_DIR)" begin
    real_dir = get(ENV, "HYDROMODELS_TEST_DATA_DIR", "")
    if isempty(real_dir) || !isdir(real_dir)
        @test_skip "HYDROMODELS_TEST_DATA_DIR unset or not a directory; skipping real-data check"
    else
        yamls = filter(f -> endswith(f, ".yaml"), readdir(real_dir; join = true))
        if isempty(yamls)
            @test_skip "No .yaml files under HYDROMODELS_TEST_DATA_DIR=$real_dir"
        else
            for path in first(yamls, 1)   # only the first to keep CI fast
                @info "Smoke-parsing $(basename(path)) (size $(filesize(path)) bytes)"
                parsed = read_shop_yaml(path; market_name = "NO5")
                @test parsed isa NamedTuple
                @test haskey(parsed, :reservoirs)
                @test !isempty(parsed.reservoirs)
            end
        end
    end
end
