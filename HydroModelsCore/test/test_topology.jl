using HydroModelsCore
using HydroModelsCore: build_topology
using Test

@testset "Linear cascade" begin
    upper = HydroModule{Float64,RegulationReservoir}(
        name = :upper, max_vol = 100.0, initial_vol = 50.0,
        rated_discharge = 50.0, energy_factor = 1.0,
        discharge_to = :lower,
    )
    lower = HydroModule{Float64,RegulationReservoir}(
        name = :lower, max_vol = 200.0, initial_vol = 80.0,
        rated_discharge = 80.0, energy_factor = 0.8,
    )
    topo = build_topology([upper, lower])
    @test topo.modules_by_name[:upper] == 1
    @test topo.modules_by_name[:lower] == 2
    @test (1, 2) in topo.discharge_edges
    @test topo.topological_order == [1, 2]
end

@testset "Cycle detection" begin
    a = HydroModule{Float64,RegulationReservoir}(
        name = :a, max_vol = 100.0, initial_vol = 50.0,
        rated_discharge = 50.0, energy_factor = 1.0,
        discharge_to = :b,
    )
    b = HydroModule{Float64,RegulationReservoir}(
        name = :b, max_vol = 200.0, initial_vol = 80.0,
        rated_discharge = 80.0, energy_factor = 0.8,
        discharge_to = :a,
    )
    @test_throws ArgumentError build_topology([a, b])
end

@testset "Unknown reference" begin
    m = HydroModule{Float64,RegulationReservoir}(
        name = :m, max_vol = 100.0, initial_vol = 50.0,
        rated_discharge = 50.0, energy_factor = 1.0,
        discharge_to = :ghost,
    )
    @test_throws ArgumentError build_topology([m])
end

@testset "Duplicate name" begin
    m1 = HydroModule{Float64,RegulationReservoir}(
        name = :dup, max_vol = 100.0, initial_vol = 50.0,
        rated_discharge = 50.0, energy_factor = 1.0,
    )
    m2 = HydroModule{Float64,RegulationReservoir}(
        name = :dup, max_vol = 200.0, initial_vol = 80.0,
        rated_discharge = 80.0, energy_factor = 0.8,
    )
    @test_throws ArgumentError build_topology([m1, m2])
end
