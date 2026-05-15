using HydroModelsCore
using HydroModelsOpt
using HiGHS
using SDDP
using LinearAlgebra
using Test

# ============================================================
# Canonical 2-reservoir cascade for SDDP testing
# ============================================================

"""
    cascade_problem(T)

Build a tiny 2-reservoir cascade (upper → lower) with `T` stages and
two-realization stagewise-independent inflow. Returns the
`LongTermHydroProblem` ready to feed to the SDDP bridge.

Numbers are deliberately small so SDDP.jl converges in milliseconds on
HiGHS. Storage / discharge limits are dimensionally consistent (1 m³/s
flowing for one stage adds 1 Mm³ to a downstream reservoir).
"""
function cascade_problem(num_stages::Int = 4)
    upper = HydroModule{Float64, RegulationReservoir}(
        name = :upper,
        max_vol = 100.0,
        initial_vol = 50.0,
        rated_discharge = 10.0,
        energy_factor = 1.0,
        discharge_to = :lower,
    )
    lower = HydroModule{Float64, RegulationReservoir}(
        name = :lower,
        max_vol = 100.0,
        initial_vol = 50.0,
        rated_discharge = 10.0,
        energy_factor = 0.8,
    )
    topology = build_topology([upper, lower])

    # Two-scenario stagewise-independent inflow: [wet, dry] for each stage.
    # ω[i] is the inflow to module i (in m³/s·stage units).
    realizations = [
        [Float64[6.0, 3.0], Float64[2.0, 1.0]] for _ in 1:num_stages
    ]
    probabilities = [
        [0.5, 0.5] for _ in 1:num_stages
    ]
    uncertainty = StagewiseIndependent{Float64}(realizations, probabilities)

    # Slightly increasing prices to make later-stage water more valuable.
    stage_prices = [30.0 + 5.0 * (t - 1) for t in 1:num_stages]

    return LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
        modules = [upper, lower],
        topology = topology,
        num_stages = num_stages,
        uncertainty = uncertainty,
        stage_prices = stage_prices,
    )
end

# ============================================================
# Bridge behaviour
# ============================================================

@testset "SDDP extension is loaded" begin
    @test isdefined(@__MODULE__, :SDDP)
end

@testset "Bridge — StagewiseIndependent on 2-reservoir cascade" begin
    prob = cascade_problem(4)
    solver = SDDPHydroSolver(;
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        max_iterations = 30,
        num_forward_scenarios = 20,
    )
    sol = solve(prob, solver)

    @test sol isa HydroModelsCore.Solution{Float64}
    @test sol.status == HydroModelsCore.MOI.OPTIMAL
    @test isfinite(sol.objective)
    @test sol.objective > 0
    @test isfinite(sol.bound)
    @test sol.bound > 0
    # In-sample mean should be at most the SDDP upper bound (we trained
    # to maximize), within numerical tolerance.
    @test sol.objective <= sol.bound + 1.0e-3 * max(abs(sol.bound), 1.0)
    @test sol.solve_time >= 0
    @test haskey(sol.metadata, :sddp_model)
    @test sol.metadata[:variant] === :StagewiseIndependent
end

@testset "Bridge validates problem inputs" begin
    # Missing stage_prices
    bad_prob = let p = cascade_problem(4)
        LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
            modules = p.modules,
            topology = p.topology,
            num_stages = p.num_stages,
            uncertainty = p.uncertainty,
            stage_prices = nothing,
        )
    end
    solver = SDDPHydroSolver(;
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        max_iterations = 5,
    )
    @test_throws ArgumentError solve(bad_prob, solver)

    # Mismatched stage_prices length
    wrong_len = let p = cascade_problem(4)
        LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
            modules = p.modules,
            topology = p.topology,
            num_stages = p.num_stages,
            uncertainty = p.uncertainty,
            stage_prices = [30.0],
        )
    end
    @test_throws ArgumentError solve(wrong_len, solver)
end

@testset "Bridge rejects non-JuMPSolver subproblem solvers" begin
    prob = cascade_problem(4)
    bad_solver = SDDPHydroSolver(;
        subproblem_solver = LagrangianHydroSolver(;
            step_size = ConstantStep(0.1),
        ),
        max_iterations = 5,
    )
    @test_throws ArgumentError solve(prob, bad_solver)
end

@testset "ScenarioTree / MarkovianUncertainty stubs raise informative errors" begin
    prob_st = cascade_problem(4)
    tree = ScenarioTree{Float64}(
        ScenarioNode{Float64}[ScenarioNode{Float64}(1, [5.0])],
        Int[0],
        Vector{Int}[Int[]],
        [1.0],
    )
    prob_with_tree = LongTermHydroProblem{Float64, ScenarioTree{Float64}}(
        modules = prob_st.modules,
        topology = prob_st.topology,
        num_stages = 4,
        uncertainty = tree,
        stage_prices = prob_st.stage_prices,
    )
    solver = SDDPHydroSolver(;
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        max_iterations = 5,
    )
    @test_throws ErrorException solve(prob_with_tree, solver)

    markov = MarkovianUncertainty{Float64}(
        [Float64[1.0, 2.0] for _ in 1:4],
        [Matrix{Float64}(I, 2, 2) for _ in 1:3],
    )
    prob_with_markov = LongTermHydroProblem{Float64, MarkovianUncertainty{Float64}}(
        modules = prob_st.modules,
        topology = prob_st.topology,
        num_stages = 4,
        uncertainty = markov,
        stage_prices = prob_st.stage_prices,
    )
    @test_throws ErrorException solve(prob_with_markov, solver)
end
