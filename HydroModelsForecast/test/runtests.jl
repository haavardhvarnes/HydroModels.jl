using HydroModelsForecast
using HydroModelsCore
using Random
using Statistics
using Test

@testset "HydroModelsForecast" begin

# ============================================================
# PARProcess
# ============================================================

@testset "PARProcess.fit recovers constant seasonal mean" begin
    # 5 years × 4 periods, each period has a fixed value
    period_means_true = Float64[10.0, 20.0, 15.0, 12.0]
    data = repeat(period_means_true', 5)   # 5 × 4
    par = fit(PARProcess, data)
    @test par isa PARProcess{Float64}
    @test num_periods(par) == 4
    @test isapprox(par.period_means, period_means_true; atol = 1.0e-10)
    # With zero residuals every period, σ ≈ 0
    @test all(<=(1.0e-9), par.period_residual_std)
end

@testset "PARProcess.fit OLS recovers known AR(1) coefficient" begin
    # Construct synthetic data with known φ in a single transition.
    # Use 2 periods, many years. Period 1: random. Period 2: 0.6 × p1 + noise.
    rng = MersenneTwister(42)
    n_years = 500
    p1 = randn(rng, n_years) .* 5.0 .+ 10.0
    p2 = 0.6 .* (p1 .- 10.0) .+ 20.0 .+ randn(rng, n_years) .* 0.5
    data = hcat(p1, p2)            # 500 × 2
    par = fit(PARProcess, data)
    @test isapprox(par.period_means[1], 10.0; atol = 0.5)
    @test isapprox(par.period_means[2], 20.0; atol = 0.5)
    # φ at period 2 should be close to 0.6
    @test isapprox(par.period_ar1[2], 0.6; atol = 0.1)
    # Residual std at period 2 should be close to 0.5
    @test isapprox(par.period_residual_std[2], 0.5; atol = 0.1)
end

@testset "PARProcess fit input validation" begin
    @test_throws ArgumentError fit(PARProcess, [1.0 2.0 3.0])   # only 1 year
end

@testset "PARProcess.sample produces correct length and mean" begin
    period_means = Float64[10.0, 20.0, 15.0, 12.0]
    par = PARProcess{Float64}(
        period_means,
        Float64[0.0, 0.3, 0.5, 0.4],
        Float64[1.0, 1.0, 1.0, 1.0],
    )
    rng = MersenneTwister(0)

    # Single trajectory
    traj = sample(rng, par, 10.0)
    @test length(traj) == 4
    @test traj[1] == 10.0   # starts at x0

    # Multiple trajectories
    trajs = sample(rng, par, 10.0, 200)
    @test length(trajs) == 200
    @test all(t -> length(t) == 4, trajs)

    # The mean across many trajectories should converge to the period mean
    mean_per_period = [Statistics.mean(t[p] for t in trajs) for p in 1:4]
    @test all(isapprox.(mean_per_period[2:end], period_means[2:end]; atol = 0.3))
end

@testset "PARProcess.sample with extended horizon (period cycling)" begin
    par = PARProcess{Float64}(
        Float64[10.0, 20.0], Float64[0.0, 0.5], Float64[0.1, 0.1],
    )
    rng = MersenneTwister(0)
    traj = sample(rng, par, 10.0; n_periods = 10)
    @test length(traj) == 10
end

# ============================================================
# MarkovPriceModel
# ============================================================

@testset "MarkovPriceModel.fit produces row-stochastic transitions" begin
    rng = MersenneTwister(7)
    n_scen = 200
    n_stages = 6
    # Synthetic prices with mild stage drift
    prices = 30.0 .+ 5.0 .* randn(rng, n_scen, n_stages) .+
        repeat(transpose(1.0:6.0), n_scen)
    model = fit(MarkovPriceModel, prices; K = 4)
    @test model isa MarkovPriceModel{Float64}
    @test num_stages(model) == n_stages
    @test num_nodes(model) == 4

    # Each transition matrix is row-stochastic
    for M in model.transition_matrices
        @test size(M) == (4, 4)
        @test all(>=(0), M)
        @test all(isapprox.(sum(M; dims = 2), 1.0; atol = 1.0e-10))
    end

    # Initial probabilities sum to 1
    @test isapprox(sum(model.initial_probs), 1.0; atol = 1.0e-10)

    # Node values are monotone non-decreasing within each stage
    for t in 1:n_stages
        @test issorted(model.node_values[t])
    end
end

@testset "MarkovPriceModel fit input validation" begin
    @test_throws ArgumentError fit(MarkovPriceModel, randn(3, 4); K = 5)
    @test_throws ArgumentError fit(MarkovPriceModel, randn(10, 4); K = 0)
end

@testset "MarkovPriceModel.sample produces in-range trajectories" begin
    rng = MersenneTwister(7)
    n_scen = 100
    n_stages = 4
    prices = 30.0 .+ 10.0 .* randn(rng, n_scen, n_stages)
    model = fit(MarkovPriceModel, prices; K = 3)

    trajs = sample(rng, model, 50)
    @test length(trajs) == 50
    @test all(t -> length(t) == n_stages, trajs)

    # Every sampled value must equal some node value at its stage
    for traj in trajs, (t, v) in enumerate(traj)
        @test v in model.node_values[t]
    end
end

# ============================================================
# Scenario generation
# ============================================================

@testset "historical_bootstrap preserves rows" begin
    history = Float64[1.0 2.0 3.0; 10.0 20.0 30.0; 100.0 200.0 300.0]
    rng = MersenneTwister(0)
    samples = historical_bootstrap(rng, history, 20)
    @test length(samples) == 20
    @test all(s -> length(s) == 3, samples)
    # Each sampled trajectory must equal one of the rows of `history`
    rows = [Vector{Float64}(history[s, :]) for s in 1:size(history, 1)]
    @test all(s -> s in rows, samples)
end

@testset "to_stagewise_independent converts trajectories" begin
    trajs = [
        Float64[1.0, 2.0, 3.0],
        Float64[4.0, 5.0, 6.0],
        Float64[7.0, 8.0, 9.0],
    ]
    uncertainty = to_stagewise_independent(trajs)
    @test uncertainty isa StagewiseIndependent{Float64}
    @test length(uncertainty.realizations) == 3   # n_stages
    @test length(uncertainty.realizations[1]) == 3   # n_scenarios
    @test uncertainty.realizations[1][1] == [1.0]
    @test uncertainty.realizations[1][2] == [4.0]
    @test uncertainty.realizations[2][3] == [8.0]
    # Equal probability per scenario per stage = 1/3
    @test all(p -> isapprox(p, 1.0 / 3; atol = 1.0e-12),
              uncertainty.probabilities[1])
end

@testset "to_stagewise_independent input validation" begin
    @test_throws ArgumentError to_stagewise_independent(Vector{Float64}[])
    @test_throws ArgumentError to_stagewise_independent([
        Float64[1.0, 2.0], Float64[1.0, 2.0, 3.0]
    ])
end

# ============================================================
# End-to-end: Forecast → StagewiseIndependent → SDDP-ready
# ============================================================

@testset "Forecast → Core uncertainty (smoke)" begin
    rng = MersenneTwister(123)
    par = PARProcess{Float64}(
        Float64[10.0, 12.0, 11.0, 9.0],
        Float64[0.0, 0.3, 0.3, 0.3],
        Float64[1.0, 1.0, 1.0, 1.0],
    )
    trajs = sample(rng, par, 10.0, 50)
    uncertainty = to_stagewise_independent(trajs)
    @test uncertainty isa StagewiseIndependent{Float64}
    @test length(uncertainty.realizations) == 4
    @test length(uncertainty.realizations[1]) == 50
end

end # @testset HydroModelsForecast
