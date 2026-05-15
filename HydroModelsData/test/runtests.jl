using HydroModelsData
using Test

@testset "HydroModelsData" begin
    @testset "SHOP YAML reader" begin
        include("test_shop_yaml.jl")
    end
end
