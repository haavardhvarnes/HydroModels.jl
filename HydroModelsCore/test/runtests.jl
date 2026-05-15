using HydroModelsCore
using Test

@testset "HydroModelsCore" begin
    @testset "Physical types" begin
        include("test_physical.jl")
    end
    @testset "Topology" begin
        include("test_topology.jl")
    end
    @testset "Piecewise linear" begin
        include("test_pwl.jl")
    end
    @testset "Backends" begin
        include("test_backends.jl")
    end
    @testset "Uncertainty" begin
        include("test_uncertainty.jl")
    end
    @testset "Expanded types (SHOP-style)" begin
        include("test_expanded_types.jl")
    end
end
