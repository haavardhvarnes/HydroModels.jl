using HydroModelsCore
using HydroModelsOpt
using HiGHS
using LinearAlgebra
using Test

# Metal is in the [extras]/[targets] test set so this always succeeds on
# macOS test environments. `Metal.functional()` is false on Intel Macs
# (no Apple Silicon GPU); the Metal testset gates on that flag.
const METAL_AVAILABLE = try
    @eval using Metal
    true
catch err
    @info "Metal package not available; skipping Metal testset." exception = err
    false
end

# ============================================================
# Cascade fixture: 2-reservoir, 4-stage, stagewise-independent
# ============================================================
#
# Identical shape to the SDDP bridge test (test_sddp.jl) so we exercise
# the same problem twice with two different solvers. The Lagrangian
# dispatch dualises per-reservoir end-of-horizon storage against the
# initial volume.

function lagrangian_cascade(num_stages::Int = 4)
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
    realizations = [
        [Float64[6.0, 3.0], Float64[2.0, 1.0]] for _ in 1:num_stages
    ]
    probabilities = [[0.5, 0.5] for _ in 1:num_stages]
    uncertainty = StagewiseIndependent{Float64}(realizations, probabilities)
    stage_prices = [30.0 + 5.0 * (t - 1) for t in 1:num_stages]

    return LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
        modules = [upper, lower],
        topology = build_topology([upper, lower]),
        num_stages = num_stages,
        uncertainty = uncertainty,
        stage_prices = stage_prices,
    )
end

# ============================================================

@testset "Lagrangian solver — end-to-end on cascade" begin
    prob = lagrangian_cascade(4)
    solver = LagrangianHydroSolver(;
        backend = CPUBackend(),       # pin to LP path even when Metal is loaded
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        step_size = DiminishingStep(1.0, 0.9),
        max_iter = 10,
        n_scenarios = 4,
        dispatch = :lp,
    )
    sol = solve(prob, solver)

    @test sol isa HydroModelsCore.Solution{Float64}
    @test isfinite(sol.objective)
    @test sol.objective > 0
    @test sol.solve_time >= 0
    @test sol.metadata[:variant] === :LagrangianStagewiseIndependent
    @test sol.metadata[:n_scenarios] == 4
    @test sol.metadata[:ka_backend] isa HydroModelsCore.ComputeBackend
    # Multipliers must be finite (no NaN/Inf escape from subgradient updates).
    λ = sol.dual.multipliers
    @test all(isfinite, λ)
    # The scenario-wise primal record must be populated and finite.
    objs = sol.primal.scenario_objectives
    @test length(objs) == 4
    @test all(isfinite, objs)
    S_end = sol.primal.scenario_S_end
    @test size(S_end) == (4, 2)
    @test all(>=(0), S_end)
end

@testset "Lagrangian solver — validation errors (LP path)" begin
    base = lagrangian_cascade(4)
    optimizer = JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false))

    # 1. Missing stage_prices
    no_prices = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
        modules = base.modules,
        topology = base.topology,
        num_stages = base.num_stages,
        uncertainty = base.uncertainty,
        stage_prices = nothing,
    )
    solver = LagrangianHydroSolver(;
        backend = CPUBackend(),
        subproblem_solver = optimizer,
        step_size = ConstantStep(0.1),
        max_iter = 2,
        dispatch = :lp,
    )
    @test_throws ArgumentError solve(no_prices, solver)

    # 2. Mismatched stage_prices length
    wrong_len = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
        modules = base.modules,
        topology = base.topology,
        num_stages = base.num_stages,
        uncertainty = base.uncertainty,
        stage_prices = [30.0, 35.0],
    )
    @test_throws ArgumentError solve(wrong_len, solver)

    # 3. subproblem_solver is not a JuMPSolver
    bad_inner = LagrangianHydroSolver(;
        backend = CPUBackend(),
        subproblem_solver = ConstantStep(0.1),    # nonsense — wrong type
        step_size = ConstantStep(0.1),
        max_iter = 2,
        dispatch = :lp,
    )
    @test_throws ArgumentError solve(base, bad_inner)

    # 4. subproblem_solver omitted (defaults to nothing)
    no_inner = LagrangianHydroSolver(;
        backend = CPUBackend(),
        step_size = ConstantStep(0.1),
        max_iter = 2,
        dispatch = :lp,
    )
    @test_throws ArgumentError solve(base, no_inner)

    # 5. subproblem_solver wraps optimizer = nothing (build-only path)
    nothing_optimizer = LagrangianHydroSolver(;
        backend = CPUBackend(),
        subproblem_solver = JuMPSolver(nothing),
        step_size = ConstantStep(0.1),
        max_iter = 2,
        dispatch = :lp,
    )
    @test_throws ArgumentError solve(base, nothing_optimizer)
end

@testset "Lagrangian solver — non-StagewiseIndependent dispatches throw" begin
    base = lagrangian_cascade(4)
    optimizer = JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false))

    tree = ScenarioTree{Float64}(
        ScenarioNode{Float64}[ScenarioNode{Float64}(1, Float64[5.0])],
        Int[0],
        Vector{Int}[Int[]],
        [1.0],
    )
    prob_with_tree = LongTermHydroProblem{Float64, ScenarioTree{Float64}}(
        modules = base.modules,
        topology = base.topology,
        num_stages = base.num_stages,
        uncertainty = tree,
        stage_prices = base.stage_prices,
    )
    solver = LagrangianHydroSolver(;
        backend = CPUBackend(),
        subproblem_solver = optimizer,
        step_size = ConstantStep(0.1),
        max_iter = 2,
        dispatch = :lp,
    )
    @test_throws ErrorException solve(prob_with_tree, solver)

    markov = MarkovianUncertainty{Float64}(
        [Float64[1.0, 2.0] for _ in 1:4],
        [Matrix{Float64}(I, 2, 2) for _ in 1:3],
    )
    prob_with_markov = LongTermHydroProblem{Float64, MarkovianUncertainty{Float64}}(
        modules = base.modules,
        topology = base.topology,
        num_stages = base.num_stages,
        uncertainty = markov,
        stage_prices = base.stage_prices,
    )
    @test_throws ErrorException solve(prob_with_markov, solver)
end

@testset "Lagrangian solver — multipliers respond to subgradient sign (LP path)" begin
    # With ConstantStep, after a few iterations the multipliers must have
    # moved away from zero unless mean(S_end) == target_i for every
    # reservoir at iteration 1 — which would happen by coincidence only.
    prob = lagrangian_cascade(4)
    solver = LagrangianHydroSolver(;
        backend = CPUBackend(),
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        step_size = ConstantStep(0.01),
        max_iter = 5,
        n_scenarios = 4,
        dispatch = :lp,
    )
    sol = solve(prob, solver)
    @test any(!iszero, sol.dual.multipliers)
end

# ============================================================
# Metal backend — runs only when Metal is loaded and functional
# ============================================================
#
# On Apple Silicon, `using Metal; Metal.functional()` enables the
# `GPUBackend(MetalBackend())` path. The kernel demotes Float64 → Float32
# inside the device aggregation; the outer subgradient loop stays in
# Float64. The CPU and Metal runs should produce visibly similar
# objectives on this 2-reservoir cascade.

if METAL_AVAILABLE
    @testset "Lagrangian solver — Metal backend, LP dispatch (Apple Silicon)" begin
        if Metal.functional()
            prob = lagrangian_cascade(4)
            metal_backend = HydroModelsCore.GPUBackend(MetalBackend())
            solver = LagrangianHydroSolver(;
                backend = metal_backend,
                subproblem_solver = JuMPSolver(HiGHS.Optimizer;
                    options = Dict(:output_flag => false)),
                step_size = DiminishingStep(1.0, 0.9),
                max_iter = 5,
                n_scenarios = 4,
                dispatch = :lp,            # force LP path on Metal for the C3 sanity check
            )
            sol = solve(prob, solver)

            @test sol isa HydroModelsCore.Solution{Float64}
            @test sol.metadata[:variant] === :LagrangianStagewiseIndependent
            @test sol.metadata[:ka_backend] === metal_backend
            @test isfinite(sol.objective)
            @test sol.objective > 0
            @test all(isfinite, sol.dual.multipliers)

            # Compare against the CPU path: a coarse equivalence check
            # (Float32 on Metal vs. Float64 on CPU) — objectives should
            # agree to within ~1 % on this tiny instance.
            cpu_solver = LagrangianHydroSolver(;
                backend = CPUBackend(),
                subproblem_solver = JuMPSolver(HiGHS.Optimizer;
                    options = Dict(:output_flag => false)),
                step_size = DiminishingStep(1.0, 0.9),
                max_iter = 5,
                n_scenarios = 4,
                dispatch = :lp,
            )
            sol_cpu = solve(prob, cpu_solver)
            @test abs(sol.objective - sol_cpu.objective) <=
                0.01 * max(abs(sol_cpu.objective), 1.0)
        else
            @info "Metal.functional() == false (not Apple Silicon, or " *
                  "Metal disabled); skipping Metal exercise."
            @test true
        end
    end
end

# ============================================================
# C3.5 — KA-native Bellman DP dispatch
# ============================================================
#
# These testsets exercise the `dispatch = :dp` path landed in C3.5.
# The cascade fixture is the same one used for the LP path; the test
# strategy is:
#
# 1. Single iteration, λ pinned at zero → both LP and DP must reach
#    full generation (q = q_max), giving identical Lagrangian objectives
#    (just revenue, no S_end term).
# 2. End-to-end finite objective + finite multipliers across several
#    iterations of the subgradient loop.
# 3. Stub-rejection: BufferReservoir problems and bad K_grid throw.
# 4. Metal: the same DP code runs on `GPUBackend(MetalBackend())`
#    without source changes.

@testset "C3.5 — DP dispatch matches LP at λ = 0 (single-iter sanity)" begin
    prob = lagrangian_cascade(4)
    common = (
        backend = CPUBackend(),
        step_size = ConstantStep(0.0),    # λ stays at 0 the whole time
        max_iter = 1,
        n_scenarios = 4,
    )
    lp_solver = LagrangianHydroSolver(; common...,
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        dispatch = :lp)
    dp_solver = LagrangianHydroSolver(; common...,
        dispatch = :dp, K_grid = 64, dual_flow_coupling = false)

    sol_lp = solve(prob, lp_solver)
    sol_dp = solve(prob, dp_solver)

    @test sol_lp.metadata[:dispatch] === :lp
    @test sol_dp.metadata[:dispatch] === :dp

    # Both should reach max revenue; the DP discretisation tolerance on
    # this fixture is one grid bin × max-price × max-η ≈ (100/63) * 45 ≈ 71.
    tol = 75.0
    @test abs(sol_lp.objective - sol_dp.objective) <= tol
end

@testset "C3.5 — DP end-to-end on cascade" begin
    prob = lagrangian_cascade(4)
    solver = LagrangianHydroSolver(;
        backend = CPUBackend(),
        step_size = DiminishingStep(1.0, 0.9),
        max_iter = 5,
        n_scenarios = 4,
        dispatch = :dp,
        K_grid = 32,
        dual_flow_coupling = true,
    )
    sol = solve(prob, solver)

    @test sol isa HydroModelsCore.Solution{Float64}
    @test sol.metadata[:variant] === :LagrangianDP
    @test sol.metadata[:dispatch] === :dp
    @test sol.metadata[:K_grid] == 32
    @test sol.metadata[:dual_flow_coupling] === true
    @test isfinite(sol.objective)
    @test all(isfinite, sol.dual.λ)
    @test sol.dual.μ !== nothing
    @test all(isfinite, sol.dual.μ)
    # Primal trajectory shape: (T+1, nmod, n_scen)
    @test size(sol.primal.S_traj) == (5, 2, 4)
    @test size(sol.primal.q_traj) == (4, 2, 4)
end

@testset "C3.5 — DP validation errors" begin
    base = lagrangian_cascade(4)

    # K_grid < 2 → ArgumentError
    bad_K = LagrangianHydroSolver(;
        backend = CPUBackend(),
        step_size = ConstantStep(0.1),
        max_iter = 1, K_grid = 1, dispatch = :dp,
    )
    @test_throws ArgumentError solve(base, bad_K)

    # Missing stage_prices
    no_prices = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
        modules = base.modules, topology = base.topology,
        num_stages = base.num_stages, uncertainty = base.uncertainty,
        stage_prices = nothing,
    )
    dp_solver = LagrangianHydroSolver(;
        backend = CPUBackend(),
        step_size = ConstantStep(0.1),
        max_iter = 1, dispatch = :dp,
    )
    @test_throws ArgumentError solve(no_prices, dp_solver)
end

@testset "C3.5 — dispatch :auto resolves correctly" begin
    prob = lagrangian_cascade(4)
    # On CPUBackend, :auto resolves to :lp (preserves C3 behaviour).
    auto_cpu = LagrangianHydroSolver(;
        backend = CPUBackend(),
        subproblem_solver = JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)),
        step_size = ConstantStep(0.1),
        max_iter = 2,
    )
    sol = solve(prob, auto_cpu)
    @test sol.metadata[:dispatch] === :lp

    # Unknown dispatch symbol
    bad = LagrangianHydroSolver(;
        backend = CPUBackend(),
        step_size = ConstantStep(0.1),
        max_iter = 1, dispatch = :marsupial,
    )
    @test_throws ArgumentError solve(prob, bad)
end

if METAL_AVAILABLE
    @testset "C3.5 — DP dispatch on Metal" begin
        if Metal.functional()
            prob = lagrangian_cascade(4)
            metal_backend = HydroModelsCore.GPUBackend(MetalBackend())
            solver = LagrangianHydroSolver(;
                backend = metal_backend,
                step_size = DiminishingStep(1.0, 0.9),
                max_iter = 5,
                n_scenarios = 4,
                dispatch = :dp,
                K_grid = 32,
                dual_flow_coupling = false,
            )
            sol = solve(prob, solver)

            @test sol.metadata[:variant] === :LagrangianDP
            @test sol.metadata[:dispatch] === :dp
            @test sol.metadata[:ka_backend] === metal_backend
            @test isfinite(sol.objective)
            @test all(isfinite, sol.dual.λ)

            # With :auto and a Metal backend, :dp must be the resolved dispatch.
            auto_metal = LagrangianHydroSolver(;
                backend = metal_backend,
                step_size = DiminishingStep(1.0, 0.9),
                max_iter = 2,
                n_scenarios = 4,
                K_grid = 32,
            )
            sol_auto = solve(prob, auto_metal)
            @test sol_auto.metadata[:dispatch] === :dp
        else
            @test true
        end
    end
end
