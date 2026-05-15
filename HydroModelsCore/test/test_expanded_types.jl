using HydroModelsCore
using Dates
using OrderedCollections
using Test

@testset "PWL alias" begin
    pwl = PWL([0.0, 1.0, 2.0], [0.0, 1.0, 3.0])
    @test pwl isa PiecewiseLinear{Float64}
    @test PWL === PiecewiseLinear
end

@testset "Time series" begin
    @test isempty(InflowSeries{Float64}())
    @test isempty(MarketSeries{Float64}())

    od = OrderedDict(DateTime(2024, 1, 1) => 10.0, DateTime(2024, 1, 2) => 12.0)
    inflow = InflowSeries(od)
    @test length(inflow) == 2
    @test first(keys(inflow.by_time)) == DateTime(2024, 1, 1)

    market = MarketSeries{Float32}(OrderedDict(DateTime(2024, 1, 1) => 30.0f0))
    @test length(market) == 1
    @test eltype(values(market.by_time)) == Float32
end

@testset "ReserveSpec construction" begin
    spec = ReserveSpec{Float64}(
        product = "FCR_N_UP",
        pmin    = 0.0,
        pmax    = 5.0,
    )
    @test spec.product == "FCR_N_UP"
    @test spec.pmax == 5.0
    @test isempty(spec.schedule)
    @test isempty(spec.penalty_cost)

    spec2 = ReserveSpec{Float64}(
        product      = "FRR_DOWN",
        pmin         = 0.0,
        pmax         = 3.0,
        schedule     = OrderedDict(DateTime(2024, 1, 1) => 2.0),
        penalty_cost = OrderedDict(DateTime(2024, 1, 1) => 500.0),
    )
    @test spec2.schedule[DateTime(2024, 1, 1)] == 2.0
end

@testset "ReserveGroup construction" begin
    grp = ReserveGroup{Float64}(
        name        = "NO5_FCR_N_UP",
        product     = "FCR_N_UP",
        obligation  = OrderedDict(DateTime(2024, 1, 1) => 10.0),
        penalty_cost = OrderedDict(DateTime(2024, 1, 1) => 1000.0),
    )
    @test grp.name == "NO5_FCR_N_UP"
    @test grp.obligation[DateTime(2024, 1, 1)] == 10.0
end

@testset "ReservoirWaterValue and CutGroup" begin
    rwv = ReservoirWaterValue([0.0, 100.0], [1.0e6, 0.5e6])
    @test rwv isa ReservoirWaterValue{Float64}
    @test length(rwv.ref) == 2
    @test length(rwv.slope) == 2

    cg = CutGroup{Float64}(
        name      = "NO5_group",
        res_names = ["res_a", "res_b"],
        ncuts     = 3,
        intercept = [10.0, 20.0, 15.0],
        slopes    = [1.0 2.0 1.5; 0.5 1.0 0.75],
    )
    @test cg.name == "NO5_group"
    @test cg.ncuts == 3
    @test size(cg.slopes) == (2, 3)
    @test isempty(cg.res_indices)  # default
end

@testset "Plant" begin
    p = Plant{Float64}(
        name        = "Aurland",
        outlet_line = 50.0,
    )
    @test p.name == "Aurland"
    @test p.outlet_line == 50.0
    @test isempty(p.main_loss)
    @test isempty(p.maintenance_flag)
end

@testset "Reservoir" begin
    r = Reservoir{Float64}(
        name        = "upper",
        vol_breaks  = [0.0, 50.0, 100.0],
        head_breaks = [200.0, 220.0, 240.0],
        s_min       = 0.0,
        s_max       = 100.0,
        s0          = 50.0,
    )
    @test r.name == "upper"
    @test r.s0 == 50.0
    @test r.water_value === nothing
    @test isempty(r.inflow)
    @test r.spill_to_reservoir === nothing      # default

    rwv = ReservoirWaterValue([0.0], [1.0e6])
    r2 = Reservoir{Float64}(
        name        = "lower",
        vol_breaks  = [0.0, 200.0],
        head_breaks = [100.0, 150.0],
        s_min       = 0.0,
        s_max       = 200.0,
        s0          = 80.0,
        water_value = rwv,
        spill_to_reservoir = "ocean",
    )
    @test r2.water_value === rwv
    @test r2.spill_to_reservoir == "ocean"
end

@testset "Generator" begin
    curve = PiecewiseLinear([5.0, 25.0, 50.0], [4.0, 22.0, 45.0])
    gen = Generator{Float64}(
        name     = "Aurland_G1",
        plant    = "Aurland",
        from_res = ["upper"],
        to_res   = "lower",
        curves   = [(200.0, curve)],
        qmin     = 5.0,
        qmax     = 50.0,
        pmin     = 4.0,
        pmax     = 45.0,
        p_nom    = 50.0,
    )
    @test gen.name == "Aurland_G1"
    @test gen.plant == "Aurland"
    @test length(gen.curves) == 1
    @test gen.curves[1][1] == 200.0
    @test gen.curves[1][2] isa PiecewiseLinear{Float64}
    @test gen.penstock == 0       # default
    @test gen.gen_eff === nothing  # default
    @test gen.startcost == 0.0     # default
    @test gen.initial_state == 0   # default
    @test isempty(gen.reserves)    # default
end

@testset "Pump" begin
    curve = PiecewiseLinear([2.0, 10.0, 20.0], [2.5, 12.0, 25.0])
    pmp = Pump{Float64}(
        name     = "Aurland_P1",
        plant    = "Aurland",
        from_res = ["lower"],
        to_res   = "upper",
        curves   = [(150.0, curve)],
        qmin     = 2.0,
        qmax     = 20.0,
        pmin     = 2.5,
        pmax     = 25.0,
        p_nom    = 25.0,
    )
    @test pmp.name == "Aurland_P1"
    @test pmp.penstock == 0
    @test pmp.startcost == 0.0
end

@testset "Tunnel" begin
    tun = Tunnel{Float64}(
        name         = "T_upper_lower",
        from_node    = "upper",
        to_node      = "lower",
        qmax         = 100.0,
        start_height = 240.0,
        end_height   = 150.0,
    )
    @test tun.name == "T_upper_lower"
    @test tun.qmax == 100.0
    @test tun.loss_factor == 0.0   # default
end

@testset "Junction (B1.7)" begin
    # Defaults — no fields beyond name/elev mean an unconstrained junction.
    bare = Junction{Float64}(name = "K")
    @test bare.name == "K"
    @test bare.upstream_elevation == 0.0
    @test isempty(bare.max_flow)
    @test isempty(bare.flow)

    # Populated junction with both constraint schedules.
    t1 = DateTime("2024-01-01T00:00:00")
    t2 = DateTime("2024-01-01T06:00:00")
    j = Junction{Float64}(
        name               = "BergheimKP",
        upstream_elevation = 0.001,
        max_flow           = OrderedDict(t1 => 1000.0),
        flow               = OrderedDict(t1 => 80.0, t2 => 120.0),
    )
    @test j.upstream_elevation == 0.001
    @test j.max_flow[t1] == 1000.0
    @test j.flow[t2] == 120.0
end

@testset "Type parameter consistency (Float32)" begin
    # All SHOP-style structs should compile and work with Float32 too.
    inflow32 = InflowSeries{Float32}()
    rwv32 = ReservoirWaterValue(Float32[0.0, 100.0], Float32[1.0e6, 0.5e6])
    r32 = Reservoir{Float32}(
        name        = "tiny",
        vol_breaks  = Float32[0.0, 1.0],
        head_breaks = Float32[100.0, 110.0],
        s_min       = 0.0f0,
        s_max       = 1.0f0,
        s0          = 0.5f0,
        inflow      = inflow32,
        water_value = rwv32,
    )
    @test r32 isa Reservoir{Float32}
    @test r32.water_value isa ReservoirWaterValue{Float32}
end
