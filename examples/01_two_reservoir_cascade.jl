"""
Example 1: Two-reservoir cascade — the canonical hydropower test problem.

Mirrors the hydro_valley test case from SDDP.jl but using HydroModels'
ProdRisk-style type system.

This example is currently a *type construction* demo — the solvers are
still stubs. Once `solve(::LongTermHydroProblem, ::SDDPHydroSolver)` is
implemented, this becomes a runnable end-to-end test.
"""

using HydroModels
using HydroModels: build_topology
using LinearAlgebra: I
using Random: randn

# Two-reservoir valley: water flows :upper → :lower → outflow
upper = HydroModule{Float64,RegulationReservoir}(
    name           = :upper,
    max_vol        = 200.0,         # Mm³
    initial_vol    = 200.0,
    rated_discharge = 70.0,         # m³/s
    energy_factor  = 1.1,           # kWh/m³
    discharge_to   = :lower,
)

lower = HydroModule{Float64,RegulationReservoir}(
    name           = :lower,
    max_vol        = 200.0,
    initial_vol    = 200.0,
    rated_discharge = 70.0,
    energy_factor  = 1.0,
)

topology = build_topology([upper, lower])

# ProdRisk-style uncertainty would normally be loaded from PRISMOD-format
# files. Here we construct minimal placeholder data.
T = 52  # weeks
n_scen = 30
n_gauge = 2
n_price_nodes = 5

uncertainty = ProdRiskUncertainty{Float64}(
    # PAR(1) for backward
    inflow_par_coeffs    = 0.3 * ones(T, n_gauge),
    inflow_residual_std  = 5.0 * ones(T, n_gauge),
    # Historical for forward
    historical_inflow    = 10.0 .+ 5.0 .* randn(T, n_scen, n_gauge),
    historical_price     = 30.0 .+ 10.0 .* randn(T, n_scen),
    # Markov price (PRISMOD-style)
    price_nodes          = [collect(range(20.0, 60.0; length=n_price_nodes)) for _ in 1:T],
    price_transitions    = [Matrix{Float64}(I, n_price_nodes, n_price_nodes) for _ in 1:T-1],
    price_profiles       = [ones(n_price_nodes, 1) for _ in 1:T],
    n_price_nodes        = n_price_nodes,
    n_min_scenarios      = 3,
)

problem = LongTermHydroProblem(
    modules     = [upper, lower],
    topology    = topology,
    num_stages  = T,
    uncertainty = uncertainty,
)

@info "Problem constructed" num_modules=length(problem.modules) num_stages=problem.num_stages

# Once the solver is implemented:
# solver = SDDPHydroSolver(
#     subproblem_solver = JuMPSolver(HiGHS.Optimizer),
#     max_iterations = 100,
# )
# solution = solve(problem, solver)
