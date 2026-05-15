using HydroModelsCore
using HydroModelsOpt
using HiGHS
using SDDP
using LinearAlgebra
using Test

# ============================================================
# Synthetic ProdRiskUncertainty instance for testing
# ============================================================

function prodrisk_cascade(num_stages::Int = 4; n_scenarios::Int = 5,
                          n_price_nodes::Int = 2)
    nmod = 2
    # Synthetic historical inflow: stages × scenarios × modules
    historical_inflow = Array{Float64, 3}(undef, num_stages, n_scenarios, nmod)
    for t in 1:num_stages, s in 1:n_scenarios, i in 1:nmod
        historical_inflow[t, s, i] = 3.0 + 2.0 * sin(2π * t / num_stages) +
            0.5 * (s - 1) + 0.3 * (i - 1)
    end
    historical_price = [25.0 + 5.0 * t + 2.0 * (s - 1)
                        for t in 1:num_stages, s in 1:n_scenarios]

    # Price nodes: simple K-node grid per stage
    price_nodes = [
        [20.0 + 10.0 * (k - 1) + 2.0 * t for k in 1:n_price_nodes]
        for t in 1:num_stages
    ]
    # Trivial transition: stay with probability 0.7, switch uniformly otherwise
    function trivial_transition(K_in, K_out)
        M = fill((1.0 - 0.7) / K_out, K_in, K_out)
        for k in 1:min(K_in, K_out)
            M[k, k] = 0.7 + (1.0 - 0.7) / K_out
        end
        # Renormalize rows just in case
        for i in 1:K_in
            M[i, :] ./= sum(M[i, :])
        end
        return M
    end
    price_transitions = [
        trivial_transition(n_price_nodes, n_price_nodes)
        for _ in 1:(num_stages - 1)
    ]

    # PAR(1) coefficients (unused by this MVP — symmetric forward/backward)
    par_coeffs = zeros(num_stages, nmod)
    par_residual_std = ones(num_stages, nmod)

    # Within-stage price profile: trivial single-period (no within-stage variation)
    price_profiles = [ones(n_price_nodes, 1) for _ in 1:num_stages]

    uncertainty = ProdRiskUncertainty{Float64}(
        par_coeffs, par_residual_std,
        historical_inflow, historical_price,
        price_nodes, price_transitions, price_profiles,
        n_price_nodes, 1,
    )

    upper = HydroModule{Float64, RegulationReservoir}(
        name = :upper, max_vol = 50.0, initial_vol = 25.0,
        rated_discharge = 10.0, energy_factor = 1.0, discharge_to = :lower,
    )
    lower = HydroModule{Float64, RegulationReservoir}(
        name = :lower, max_vol = 50.0, initial_vol = 25.0,
        rated_discharge = 10.0, energy_factor = 0.8,
    )

    return LongTermHydroProblem{Float64, ProdRiskUncertainty{Float64}}(
        modules = [upper, lower],
        topology = build_topology([upper, lower]),
        num_stages = num_stages,
        uncertainty = uncertainty,
        # stage_prices is intentionally nothing — ProdRisk path uses price_nodes
    )
end

# ============================================================

@testset "ProdRiskUncertainty bridge end-to-end on cascade" begin
    prob = prodrisk_cascade(4)
    solver = SDDPHydroSolver(;
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        max_iterations = 80,
        num_forward_scenarios = 50,
    )
    sol = solve(prob, solver)

    @test sol isa HydroModelsCore.Solution{Float64}
    @test sol.status == HydroModelsCore.MOI.OPTIMAL
    @test isfinite(sol.objective)
    @test sol.objective > 0
    @test isfinite(sol.bound)
    @test sol.bound > 0
    # SDDP for :Max gives a deterministic upper bound. The in-sample
    # simulation mean approaches it as iterations grow but with finite
    # iterations and MC noise it can still slightly exceed the bound;
    # allow a 10 % envelope on this tiny instance.
    @test sol.objective <= sol.bound + 0.10 * abs(sol.bound)
    @test haskey(sol.metadata, :sddp_model)
    @test sol.metadata[:variant] === :ProdRiskUncertainty
    @test sol.metadata[:forward_backward_asymmetric] === false
end

@testset "ProdRiskUncertainty validation" begin
    solver = SDDPHydroSolver(;
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        max_iterations = 3,
    )

    # Wrong number of modules vs gauges
    bad = prodrisk_cascade(4)
    bad_inflow = bad.uncertainty.historical_inflow[:, :, 1:1]  # 1 gauge, 2 modules
    bad_u = ProdRiskUncertainty{Float64}(
        bad.uncertainty.inflow_par_coeffs,
        bad.uncertainty.inflow_residual_std,
        bad_inflow, bad.uncertainty.historical_price,
        bad.uncertainty.price_nodes, bad.uncertainty.price_transitions,
        bad.uncertainty.price_profiles, bad.uncertainty.n_price_nodes,
        bad.uncertainty.n_min_scenarios,
    )
    bad_prob = LongTermHydroProblem{Float64, ProdRiskUncertainty{Float64}}(
        modules = bad.modules, topology = bad.topology,
        num_stages = bad.num_stages, uncertainty = bad_u,
    )
    @test_throws ArgumentError solve(bad_prob, solver)

    # Wrong number of price_transitions
    bad2_u = ProdRiskUncertainty{Float64}(
        bad.uncertainty.inflow_par_coeffs,
        bad.uncertainty.inflow_residual_std,
        bad.uncertainty.historical_inflow, bad.uncertainty.historical_price,
        bad.uncertainty.price_nodes,
        [bad.uncertainty.price_transitions[1]],   # length 1, should be 3
        bad.uncertainty.price_profiles, bad.uncertainty.n_price_nodes,
        bad.uncertainty.n_min_scenarios,
    )
    bad_prob2 = LongTermHydroProblem{Float64, ProdRiskUncertainty{Float64}}(
        modules = bad.modules, topology = bad.topology,
        num_stages = bad.num_stages, uncertainty = bad2_u,
    )
    @test_throws ArgumentError solve(bad_prob2, solver)

    # Non-row-stochastic transitions
    bad3_trans = copy.(bad.uncertainty.price_transitions)
    bad3_trans[1][1, :] = [0.5, 0.3]   # row sums to 0.8
    bad3_u = ProdRiskUncertainty{Float64}(
        bad.uncertainty.inflow_par_coeffs,
        bad.uncertainty.inflow_residual_std,
        bad.uncertainty.historical_inflow, bad.uncertainty.historical_price,
        bad.uncertainty.price_nodes, bad3_trans,
        bad.uncertainty.price_profiles, bad.uncertainty.n_price_nodes,
        bad.uncertainty.n_min_scenarios,
    )
    bad_prob3 = LongTermHydroProblem{Float64, ProdRiskUncertainty{Float64}}(
        modules = bad.modules, topology = bad.topology,
        num_stages = bad.num_stages, uncertainty = bad3_u,
    )
    @test_throws ArgumentError solve(bad_prob3, solver)
end

@testset "ProdRiskUncertainty solves with more price nodes" begin
    prob = prodrisk_cascade(3; n_scenarios = 3, n_price_nodes = 3)
    solver = SDDPHydroSolver(;
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        max_iterations = 10,
        num_forward_scenarios = 10,
    )
    sol = solve(prob, solver)
    @test sol.status == HydroModelsCore.MOI.OPTIMAL
    @test sol.metadata[:variant] === :ProdRiskUncertainty
end

@testset "ProdRiskUncertainty warns when stage_prices is set" begin
    prob = prodrisk_cascade(4)
    with_stage_prices = LongTermHydroProblem{Float64, ProdRiskUncertainty{Float64}}(
        modules = prob.modules, topology = prob.topology,
        num_stages = prob.num_stages, uncertainty = prob.uncertainty,
        stage_prices = [30.0, 35.0, 40.0, 45.0],
    )
    solver = SDDPHydroSolver(;
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        max_iterations = 3,
    )
    sol = @test_logs (:warn, r"ignores prob.stage_prices") solve(with_stage_prices, solver)
    @test sol.status == HydroModelsCore.MOI.OPTIMAL
end
