using HydroModelsCore
using Test

@testset "ScenarioTree construction" begin
    nodes = [ScenarioNode(1, [10.0]), ScenarioNode(2, [12.0]), ScenarioNode(2, [8.0])]
    tree = ScenarioTree(nodes, [0, 1, 1], [[2, 3], Int[], Int[]], [1.0, 0.5, 0.5])
    @test length(tree.nodes) == 3
end

@testset "MarkovianUncertainty" begin
    mu = MarkovianUncertainty(
        [[30.0, 50.0], [35.0, 55.0]],
        [Matrix{Float64}([0.7 0.3; 0.4 0.6])],
    )
    @test length(mu.state_values) == 2
end
