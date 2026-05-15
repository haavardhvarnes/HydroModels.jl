using HydroModelsCore
using HydroModelsCore: allocate, zeros_like, ones_like, is_gpu, ka_backend
using Test

@testset "CPU backend" begin
    b = CPUBackend()
    @test !is_gpu(b)

    a = allocate(b, Float64, 5)
    @test a isa Array{Float64,1}
    @test length(a) == 5

    z = zeros_like(b, Float32, 3, 4)
    @test all(iszero, z)
    @test size(z) == (3, 4)

    o = ones_like(b, Int, 2)
    @test all(isone, o)
end

@testset "default_backend without GPU extensions" begin
    @test default_backend() isa CPUBackend
end
