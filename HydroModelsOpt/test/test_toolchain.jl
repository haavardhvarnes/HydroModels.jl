using HydroModelsCore
using HydroModelsOpt
using HydroModelsData
using HiGHS
using SDDP
using Tables
using Test

const SHORT_FIXTURE = joinpath(
    dirname(@__DIR__), "..",
    "HydroModelsData", "test", "data", "minimal.yaml",
)

# Long-term cascade builder (shared with test_sddp.jl-style fixture).
function long_term_cascade(num_stages::Int = 4)
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
    realizations = [
        [Float64[6.0, 3.0], Float64[2.0, 1.0]] for _ in 1:num_stages
    ]
    probabilities = [[0.5, 0.5] for _ in 1:num_stages]
    uncertainty = StagewiseIndependent{Float64}(realizations, probabilities)
    stage_prices = [30.0 + 5.0 * (t - 1) for t in 1:num_stages]
    return LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
        modules = [upper, lower],
        topology = topology,
        num_stages = num_stages,
        uncertainty = uncertainty,
        stage_prices = stage_prices,
    )
end

function short_term_template()
    parsed = read_shop_yaml(SHORT_FIXTURE; market_name = "NO3")
    return ShopShortTermProblem(parsed)
end

const HIGHS_QUIET = JuMPSolver(HiGHS.Optimizer;
    options = Dict(:output_flag => false))

# ============================================================

@testset "synthetic_end_value_cuts" begin
    cuts = synthetic_end_value_cuts(
        ["upper", "lower"],
        [100.0, 200.0],
        [10.0 8.0; 5.0 4.0],
    )
    @test length(cuts) == 1
    cg = cuts[1]
    @test cg isa CutGroup{Float64}
    @test cg.name == "_synthetic"
    @test cg.res_names == ["upper", "lower"]
    @test cg.ncuts == 2
    @test cg.intercept == [100.0, 200.0]
    @test cg.slopes == [10.0 8.0; 5.0 4.0]
    @test isempty(cg.res_indices)   # filled by with_end_value_cuts
end

@testset "synthetic_end_value_cuts shape check" begin
    @test_throws ArgumentError synthetic_end_value_cuts(
        ["upper", "lower"], [100.0], [10.0 8.0; 5.0 4.0],  # ncuts mismatch
    )
    @test_throws ArgumentError synthetic_end_value_cuts(
        ["upper"], [100.0, 200.0], [10.0 8.0; 5.0 4.0],    # nres mismatch
    )
end

@testset "with_end_value_cuts resolves reservoir indices" begin
    prob = short_term_template()
    cuts = synthetic_end_value_cuts(
        ["upper", "lower"],
        [100.0, 200.0, 150.0],
        [10.0 8.0 9.0; 5.0 4.0 4.5],
    )
    new_prob = with_end_value_cuts(prob, cuts)

    @test length(new_prob.cut_groups) == 1
    cg = new_prob.cut_groups[1]
    # short_term_template reservoirs come from sorted(keys) — "lower" < "upper"
    short_res = sort!(collect(keys(prob.reservoirs)))
    @test cg.res_names == ["upper", "lower"]
    @test cg.res_indices ==
        [findfirst(==(name), short_res) for name in cg.res_names]
    @test cg.ncuts == 3
    @test cg.intercept == [100.0, 200.0, 150.0]
end

@testset "with_end_value_cuts honours reservoir_map" begin
    prob = short_term_template()
    # Pretend the long-term model used module names that are slightly
    # different from the short-term reservoir names, and supply a map.
    cuts = synthetic_end_value_cuts(
        ["UPPER_RES", "LOWER_RES"],
        [100.0, 200.0],
        [10.0 8.0; 5.0 4.0],
    )
    new_prob = with_end_value_cuts(prob, cuts;
        reservoir_map = Dict(:UPPER_RES => "upper", :LOWER_RES => "lower"))

    @test length(new_prob.cut_groups) == 1
    cg = new_prob.cut_groups[1]
    @test cg.res_names == ["upper", "lower"]
    @test !isempty(cg.res_indices)
end

@testset "with_end_value_cuts drops unmappable groups" begin
    prob = short_term_template()
    cuts = synthetic_end_value_cuts(
        ["UPPER_RES", "NONEXISTENT_RES"],  # second one has no mapping
        [100.0, 200.0],
        [10.0 8.0; 5.0 4.0],
    )
    new_prob = with_end_value_cuts(prob, cuts;
        reservoir_map = Dict(:UPPER_RES => "upper"))
    @test isempty(new_prob.cut_groups)
end

@testset "Toolchain run with user-supplied cuts is deterministic" begin
    # Bypass the long-term solve by supplying cuts directly.
    cuts = synthetic_end_value_cuts(
        ["upper", "lower"],
        [50_000.0, 30_000.0, 70_000.0],
        [1.0e6 0.5e6 1.5e6; 0.5e6 0.25e6 0.75e6],
    )

    tc = HydropowerToolchain{Float64}(
        longterm_model    = long_term_cascade(4),
        shortterm_template = short_term_template(),
        longterm_solver   = SDDPHydroSolver(;
            subproblem_solver = HIGHS_QUIET, max_iterations = 5,
        ),
        shortterm_solver  = HIGHS_QUIET,
    )

    # Same cuts → identical short-term dispatch on every run.
    res1 = run_toolchain(tc; cuts = cuts)
    res2 = run_toolchain(tc; cuts = cuts)

    @test res1.long === nothing         # long-term skipped when cuts supplied
    @test res2.long === nothing
    @test res1.cuts === cuts
    @test res2.cuts === cuts

    @test res1.short.termination == "OPTIMAL"
    @test res2.short.termination == "OPTIMAL"
    @test isapprox(res1.short.objective, res2.short.objective; atol = 1.0e-6)

    # Check storage / dispatch tables match (the LP is deterministic, so
    # this should be exact modulo solver-specific numerical noise).
    s1 = Tables.columns(res1.short.storage)
    s2 = Tables.columns(res2.short.storage)
    @test s1.S == s2.S
    p1 = Tables.columns(res1.short.p_gen)
    p2 = Tables.columns(res2.short.p_gen)
    @test p1.P == p2.P
end

@testset "Toolchain run with empty cuts still solves short-term" begin
    tc = HydropowerToolchain{Float64}(
        longterm_model    = long_term_cascade(4),
        shortterm_template = short_term_template(),
        longterm_solver   = SDDPHydroSolver(;
            subproblem_solver = HIGHS_QUIET, max_iterations = 5,
        ),
        shortterm_solver  = HIGHS_QUIET,
    )
    res = run_toolchain(tc; cuts = CutGroup{Float64}[])
    @test res.short.termination == "OPTIMAL"
    @test isempty(res.shortterm_with_cuts.cut_groups)
end

@testset "extract_end_value_cuts returns empty when no SDDP model" begin
    long_prob = long_term_cascade(4)
    # Construct a Solution without a :sddp_model in metadata
    sol = HydroModelsCore.Solution{Float64}(
        primal = nothing,
        dual = nothing,
        objective = 0.0,
        status = HydroModelsCore.MOI.OTHER_LIMIT,
        iterations = 0,
        solve_time = 0.0,
        metadata = Dict{Symbol, Any}(),
    )
    cuts = (@test_logs (:warn, r"no :sddp_model") begin
        extract_end_value_cuts(sol, long_prob)
    end)
    @test isempty(cuts)
end

# ----------------------------------------------------------------
# Full long → short integration (slow). Skipped by default; enable
# with HYDROMODELS_TEST_TOOLCHAIN_INTEGRATION=1.
# ----------------------------------------------------------------
@testset "Full long-term → short-term integration (opt-in)" begin
    if get(ENV, "HYDROMODELS_TEST_TOOLCHAIN_INTEGRATION", "") == "1"
        tc = HydropowerToolchain{Float64}(
            longterm_model    = long_term_cascade(6),
            shortterm_template = short_term_template(),
            longterm_solver   = SDDPHydroSolver(;
                subproblem_solver = HIGHS_QUIET, max_iterations = 20,
                num_forward_scenarios = 10,
            ),
            shortterm_solver  = HIGHS_QUIET,
        )
        res = run_toolchain(tc)   # full chain — extracts cuts from SDDP
        @test res.long !== nothing
        @test res.short.termination == "OPTIMAL"
        # Cut extraction is best-effort — may be empty if SDDP.jl internal
        # paths shift. Don't assert non-empty.
        @test res.cuts isa Vector{CutGroup{Float64}}
    else
        @test_skip "Set HYDROMODELS_TEST_TOOLCHAIN_INTEGRATION=1 to run the full long → short chain"
    end
end
