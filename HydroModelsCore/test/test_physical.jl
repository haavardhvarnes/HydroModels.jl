using HydroModelsCore
using Test

@testset "HydroModule construction" begin
    m = HydroModule{Float64,RegulationReservoir}(
        name = :upper,
        max_vol = 100.0,
        initial_vol = 50.0,
        rated_discharge = 50.0,
        energy_factor = 1.0,
    )
    @test m.name == :upper
    @test m.reservoir_type == RegulationReservoir()
    @test m.max_vol == 100.0
    @test m.min_vol == 0.0
    @test isnothing(m.discharge_to)
end

@testset "Buffer reservoir" begin
    m = HydroModule{Float64,BufferReservoir}(
        name = :buffer,
        reservoir_type = BufferReservoir(),
        max_vol = 5.0,
        initial_vol = 2.0,
        rated_discharge = 10.0,
        energy_factor = 0.5,
    )
    @test m.reservoir_type == BufferReservoir()
end

@testset "TurbineUnit" begin
    u = TurbineUnit{Float64}(
        name = :unit1,
        min_discharge = 5.0,
        max_discharge = 50.0,
        hpf_breakpoints = [5.0, 25.0, 50.0],
        hpf_power = [4.0, 22.0, 45.0],
    )
    @test u.startup_cost == 0.0
    @test length(u.hpf_breakpoints) == 3
end
