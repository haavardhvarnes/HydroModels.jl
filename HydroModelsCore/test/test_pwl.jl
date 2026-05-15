using HydroModelsCore
using HydroModelsCore: PiecewiseLinear, evaluate, slopes
using Test

@testset "Construction" begin
    pwl = PiecewiseLinear([0.0, 1.0, 2.0], [0.0, 2.0, 3.0])
    @test length(pwl) == 3
    @test pwl.is_convex == false  # slopes 2.0, 1.0 — decreasing → concave

    pwl_convex = PiecewiseLinear([0.0, 1.0, 2.0], [0.0, 1.0, 3.0])
    @test pwl_convex.is_convex == true  # slopes 1.0, 2.0 — increasing → convex
end

@testset "Evaluation" begin
    pwl = PiecewiseLinear([0.0, 1.0, 2.0], [0.0, 2.0, 4.0])
    @test evaluate(pwl, 0.0) ≈ 0.0
    @test evaluate(pwl, 0.5) ≈ 1.0
    @test evaluate(pwl, 1.5) ≈ 3.0
    @test evaluate(pwl, 2.0) ≈ 4.0
end

@testset "Extrapolation" begin
    pwl = PiecewiseLinear([0.0, 1.0, 2.0], [0.0, 2.0, 4.0])
    @test evaluate(pwl, -1.0) ≈ -2.0
    @test evaluate(pwl, 3.0) ≈ 6.0
end

@testset "Validation" begin
    @test_throws ArgumentError PiecewiseLinear([0.0, 1.0], [0.0, 2.0, 4.0])  # length mismatch
    @test_throws ArgumentError PiecewiseLinear([0.0], [0.0])  # too few points
    @test_throws ArgumentError PiecewiseLinear([1.0, 0.0], [0.0, 1.0])  # unsorted
end

@testset "Slopes" begin
    pwl = PiecewiseLinear([0.0, 1.0, 3.0], [0.0, 2.0, 4.0])
    s = slopes(pwl)
    @test s == [2.0, 1.0]
end
