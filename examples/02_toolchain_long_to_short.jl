# Example 2: long-term → short-term toolchain.
#
# Builds a 2-reservoir cascade as a long-term `LongTermHydroProblem`
# with StagewiseIndependent inflow uncertainty, parses a SHOP-style
# short-term YAML into a `ShopShortTermProblem`, and runs the toolchain
# that bridges them via end-of-horizon water-value cuts.
#
# Two paths are demonstrated:
#
# (a) Synthetic-cut path — bypasses the SDDP solve entirely. Useful for
#     development and for cases where cuts come from an external source
#     (a previous ProdRisk run, the literature, manual tuning).
#
# (b) Full-chain path — solves the long-term problem with SDDP.jl,
#     extracts cuts from the trained policy, and feeds them into the
#     short-term LP. Cut extraction is best-effort and may produce
#     empty cuts on some SDDP.jl versions; see HydroModelsOpt/CLAUDE.md
#     for the B3.5 follow-up plan.
#
# Run from the repo root with: julia --project=. examples/02_toolchain_long_to_short.jl

using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HiGHS
using SDDP
using Printf
using Tables

# ----------------------------------------------------------------
# Build the long-term cascade (HydroModule + StagewiseIndependent)
# ----------------------------------------------------------------
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

num_stages = 6
realizations = [
    [Float64[6.0, 3.0], Float64[2.0, 1.0]] for _ in 1:num_stages
]
probabilities = [[0.5, 0.5] for _ in 1:num_stages]
uncertainty = StagewiseIndependent{Float64}(realizations, probabilities)
stage_prices = [30.0 + 5.0 * (t - 1) for t in 1:num_stages]

long_term = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
    modules = [upper, lower],
    topology = topology,
    num_stages = num_stages,
    uncertainty = uncertainty,
    stage_prices = stage_prices,
)

# ----------------------------------------------------------------
# Build the short-term template from the minimal YAML fixture
# ----------------------------------------------------------------
short_yaml = joinpath(@__DIR__, "..", "HydroModelsData", "test", "data", "minimal.yaml")
parsed = read_shop_yaml(short_yaml; market_name = "NO3")
short_term_tmpl = ShopShortTermProblem(parsed)

# ----------------------------------------------------------------
# Wire the toolchain
# ----------------------------------------------------------------
const SOLVER = JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false))

tc = HydropowerToolchain{Float64}(
    longterm_model     = long_term,
    shortterm_template = short_term_tmpl,
    longterm_solver    = SDDPHydroSolver(;
        subproblem_solver = SOLVER,
        max_iterations    = 20,
        num_forward_scenarios = 10,
    ),
    shortterm_solver   = SOLVER,
    reservoir_map      = Dict{Symbol, String}(),   # names match directly
)

# ----------------------------------------------------------------
# (a) Synthetic-cut path
# ----------------------------------------------------------------
println("=== Path A: synthetic cuts (bypass long-term solve) ===")

# Three cuts on (upper, lower) end-of-horizon storage.
# Slopes are in EUR/Mm³ — kept high enough to actually bind in the LP.
synthetic = synthetic_end_value_cuts(
    ["upper", "lower"],
    [50_000.0, 30_000.0, 70_000.0],
    [1.0e6 0.5e6 1.5e6; 0.5e6 0.25e6 0.75e6],
)
res_a = run_toolchain(tc; cuts = synthetic)
@printf("  short-term termination: %s\n", res_a.short.termination)
@printf("  short-term objective:   %.4f EUR\n", res_a.short.objective)
@printf("  cut groups injected:    %d\n",
        length(res_a.shortterm_with_cuts.cut_groups))

# ----------------------------------------------------------------
# (b) Full-chain path (long-term SDDP solve → cut extraction → short-term)
# ----------------------------------------------------------------
println()
println("=== Path B: full long-term → cut extraction → short-term ===")

res_b = run_toolchain(tc)   # no cuts argument → extract from SDDP
@printf("  long-term termination:  %s\n", res_b.long === nothing ? "n/a" :
        string(res_b.long.status))
@printf("  long-term bound:        %.4f\n", res_b.long === nothing ? NaN :
        res_b.long.bound)
@printf("  cuts extracted:         %d cut group(s)\n", length(res_b.cuts))
if !isempty(res_b.cuts)
    cg = res_b.cuts[1]
    @printf("                          %d cuts × %d reservoir(s)\n",
            cg.ncuts, length(cg.res_names))
end
@printf("  short-term termination: %s\n", res_b.short.termination)
@printf("  short-term objective:   %.4f EUR\n", res_b.short.objective)
