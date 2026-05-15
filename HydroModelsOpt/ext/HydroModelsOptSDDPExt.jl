"""
SDDP.jl extension — delegates standard scenario-tree / stagewise-
independent / Markovian SDDP to the well-tested SDDP.jl package.

When this extension is loaded, calls to

    solve(prob::LongTermHydroProblem{T, StagewiseIndependent{T}}, solver::SDDPHydroSolver)

build an `SDDP.LinearPolicyGraph`, train it, and run a forward
simulation to estimate the in-sample objective.

`MarkovianUncertainty` and `ScenarioTree` paths and full
`WaterValueCuts` extraction are stubbed with informative errors for
now (see "Deferred" notes in `HydroModelsOpt/CLAUDE.md`).

`ProdRiskUncertainty` deliberately does **not** route through this
extension — its hybrid SDP/SDDP algorithm is implemented natively in
`src/solvers/sddp/prodrisk_hybrid.jl`.

## Unit convention

The bridge treats the LP coefficients as unitless. Revenue per stage
is computed as

    stage_prices[t] * sum(energy_factor[i] * turbine[i] for i)

with `turbine` in m³/s and `energy_factor` interpreted as the
conversion factor that produces *revenue units per (m³/s) per stage*.
For real weekly schedules in Nordic units (energy_factor in kWh/m³,
stage of one week), users should pre-multiply `energy_factor` by
`604_800 / 3.6e6 = 0.168` so that `energy_factor * stage_prices`
yields EUR. The bridge does not enforce a particular unit system;
consistency is the caller's responsibility.

## Mass balance

For each `RegulationReservoir` module `i` at stage `t`,

    S[i].out - S[i].in = inflow[i, t]
                       + sum(turbine[u] + spill[u] for u in upstream(i))
                       - turbine[i] - spill[i]

Storage is in Mm³ and flows in (m³/s) but the bridge treats the
conversion as 1.0 (i.e., one stage of 1 m³/s == 1 Mm³). For realistic
problems users should scale either `max_vol` / `initial_vol` or
`rated_discharge` so the same conversion holds.

`BufferReservoir` modules have no state and are constrained to a
pass-through balance (inflow + upstream == turbine + spill) at each
stage.
"""
module HydroModelsOptSDDPExt

using HydroModelsCore
using HydroModelsOpt
using JuMP
using SDDP

import HydroModelsCore: Solution
import HydroModelsCore: MOI

# ============================================================
# StagewiseIndependent bridge — the canonical SDDP.jl path
# ============================================================

"""
    solve(prob::LongTermHydroProblem{T, StagewiseIndependent{T}}, solver::SDDPHydroSolver)

Build an `SDDP.LinearPolicyGraph` from `prob`, train it with the
solver's configured optimizer, and simulate to produce a `Solution{T}`.

Requires `prob.stage_prices` to be set (deterministic price forecast).
"""
function HydroModelsOpt.solve(
        prob::LongTermHydroProblem{T, StagewiseIndependent{T}},
        solver::SDDPHydroSolver,
    ) where {T}

    _validate_stage_prices(prob)
    realizations, probabilities = _validate_stagewise(prob)

    optimizer = _extract_optimizer(solver)

    modules = prob.modules
    nmod = length(modules)

    is_reg = [m.reservoir_type isa RegulationReservoir for m in modules]
    reg_idx = findall(is_reg)
    isempty(reg_idx) && throw(ArgumentError(
        "SDDP bridge: at least one RegulationReservoir module is required"))

    # Build upstream index map: upstream_of[i] = vector of upstream module indices.
    upstream_of = [Int[] for _ in 1:nmod]
    for (u, v) in prob.topology.discharge_edges
        push!(upstream_of[v], u)
    end

    stage_prices = prob.stage_prices::Vector{T}
    nstages = prob.num_stages

    # Generous upper bound for SDDP's `sense = :Max`. Sum over stages of
    # (max revenue per stage = max_price × max_total_turbine). Pad by 10×.
    upper_bound = 10.0 * sum(p > 0 ? p : 0.0 for p in stage_prices) *
        sum(m.energy_factor * m.rated_discharge for m in modules)
    upper_bound = max(upper_bound, 1.0)

    sddp_model = SDDP.LinearPolicyGraph(
        stages = nstages,
        sense = :Max,
        upper_bound = Float64(upper_bound),
        optimizer = optimizer,
    ) do subproblem, t
        # State: storage per RegulationReservoir
        @variable(subproblem, S[i = reg_idx], SDDP.State,
            initial_value = Float64(modules[i].initial_vol),
            lower_bound = Float64(modules[i].min_vol),
            upper_bound = Float64(modules[i].max_vol),
        )

        # Controls — turbine and spill per module
        @variable(subproblem, 0 <= turbine[i = 1:nmod] <= Float64(modules[i].rated_discharge))
        @variable(subproblem, 0 <= spill[i = 1:nmod])

        # Parameterized inflow per module (fixed by SDDP.parameterize callback)
        @variable(subproblem, inflow_var[1:nmod])

        # Mass balance for RegulationReservoirs
        for i in reg_idx
            ups = upstream_of[i]
            @constraint(subproblem,
                S[i].out - S[i].in ==
                    inflow_var[i] +
                    sum(turbine[u] + spill[u] for u in ups; init = 0.0) -
                    turbine[i] - spill[i]
            )
        end

        # Mass balance for BufferReservoirs (pass-through, no storage state)
        for i in 1:nmod
            i in reg_idx && continue
            ups = upstream_of[i]
            @constraint(subproblem,
                inflow_var[i] +
                sum(turbine[u] + spill[u] for u in ups; init = 0.0) ==
                turbine[i] + spill[i]
            )
        end

        # Stage objective: revenue from generation
        @stageobjective(subproblem,
            stage_prices[t] *
            sum(Float64(modules[i].energy_factor) * turbine[i] for i in 1:nmod)
        )

        # Parameterize on stage-t inflow realizations
        SDDP.parameterize(subproblem, realizations[t], probabilities[t]) do ω
            for i in 1:nmod
                JuMP.fix(inflow_var[i], Float64(ω[i]); force = true)
            end
        end
    end

    # Train
    t_start = time()
    SDDP.train(sddp_model;
        iteration_limit = solver.max_iterations,
        print_level = 0,
    )
    t_elapsed = time() - t_start

    bound = SDDP.calculate_bound(sddp_model)
    # `:stage_objective` is auto-tracked by SDDP.simulate — no need to
    # list it as an additional recorded variable.
    sims = SDDP.simulate(sddp_model, solver.num_forward_scenarios)
    objective_estimate = isempty(sims) ? NaN :
        sum(sum(stage[:stage_objective] for stage in sim) for sim in sims) /
            length(sims)

    return Solution{T}(
        primal = sims,
        dual = nothing,
        objective = T(objective_estimate),
        bound = T(bound),
        status = MOI.OPTIMAL,
        iterations = solver.max_iterations,
        solve_time = t_elapsed,
        metadata = Dict{Symbol, Any}(
            :sddp_model => sddp_model,
            :variant => :StagewiseIndependent,
        ),
    )
end

# ============================================================
# Stubs for the other uncertainty variants
# ============================================================

function HydroModelsOpt.solve(
        ::LongTermHydroProblem{T, ScenarioTree{T}},
        ::SDDPHydroSolver,
    ) where {T}
    throw(ErrorException(
        "ScenarioTree → SDDP.PolicyGraph bridge not yet implemented. " *
        "Use StagewiseIndependent for now; see HydroModelsOpt/CLAUDE.md " *
        "for the B2.5 follow-up plan."))
end

function HydroModelsOpt.solve(
        ::LongTermHydroProblem{T, MarkovianUncertainty{T}},
        ::SDDPHydroSolver,
    ) where {T}
    throw(ErrorException(
        "MarkovianUncertainty → SDDP.MarkovianPolicyGraph bridge not yet " *
        "implemented. Use StagewiseIndependent for now; see " *
        "HydroModelsOpt/CLAUDE.md for the B2.5 follow-up plan."))
end

# ============================================================
# ProdRiskUncertainty bridge (milestone C2 — landed via SDDP.jl)
# ============================================================

"""
    solve(prob::LongTermHydroProblem{T, ProdRiskUncertainty{T}}, solver::SDDPHydroSolver)

Build an `SDDP.PolicyGraph` over the price-Markov outer structure
encoded in `prob.uncertainty`, train it, and return a `Solution{T}`.

# Mapping

- **Outer Markov structure** — `SDDP.MarkovianGraph(u.price_transitions)`
  produces one policy-graph node per `(stage, price_node)`. The price at
  stage `t`, node `k` is `u.price_nodes[t][k]`.
- **Inflow uncertainty** — per-stage independent draws from
  `u.historical_inflow[t, :, :]`. Each historical year contributes one
  scenario at each stage, weighted equally.
- **Modules** — `historical_inflow[t, scen, i]` is the inflow at stage
  `t`, scenario `scen`, **module index `i`**. Each module must be its
  own gauge for this MVP path.

# Deviation from the strict plan

The plan called for a *native* combined SDP/SDDP (Gjelsvik / Belsnes /
Haugstad 1999) with the asymmetric forward/backward sampling that is
the Nordic differentiator:

- Forward: historical scenarios (preserves inflow-price correlation)
- Backward: fitted PAR(1) + Markov

This MVP uses SDDP.jl's symmetric Markovian machinery for **both**
forward and backward (same historical-inflow distribution either way).
That delivers a working dispatch on `ProdRiskUncertainty` and a usable
research playground, at the cost of losing the ProdRisk-specific
asymmetry. The native implementation with PAR(1)-in-the-backward is
C2.5 in the roadmap.
"""
function HydroModelsOpt.solve(
        prob::LongTermHydroProblem{T, ProdRiskUncertainty{T}},
        solver::SDDPHydroSolver,
    ) where {T}

    u = prob.uncertainty
    _validate_prodrisk(prob, u)

    optimizer = _extract_optimizer(solver)

    modules = prob.modules
    nmod = length(modules)
    is_reg = [m.reservoir_type isa RegulationReservoir for m in modules]
    reg_idx = findall(is_reg)
    isempty(reg_idx) && throw(ArgumentError(
        "SDDP bridge: at least one RegulationReservoir module is required"))

    upstream_of = [Int[] for _ in 1:nmod]
    for (u_edge, v_edge) in prob.topology.discharge_edges
        push!(upstream_of[v_edge], u_edge)
    end

    nstages = prob.num_stages
    nscen = size(u.historical_inflow, 2)   # number of historical years

    # Build Markovian graph over price nodes. SDDP.MarkovianGraph wants
    # `nstages` transition matrices total, with the FIRST being shape
    # (1, K_1) representing the initial distribution from a deterministic
    # root. ProdRiskUncertainty only carries the inter-stage transitions
    # (`nstages - 1` matrices, each K_t × K_{t+1}). Prepend a uniform
    # initial distribution over the K_1 stage-1 price nodes.
    K1 = length(u.price_nodes[1])
    initial = reshape(fill(1.0 / K1, K1), 1, K1)
    transitions = Vector{Matrix{Float64}}(undef, nstages)
    transitions[1] = initial
    for t in 1:(nstages - 1)
        transitions[t + 1] = Matrix{Float64}(u.price_transitions[t])
    end
    graph = SDDP.MarkovianGraph(transitions)

    # Generous Max-sense upper bound: sum_t (max price at stage t) ×
    # total turbine power × 10× safety.
    max_price_per_stage = [maximum(u.price_nodes[t]) for t in 1:nstages]
    upper_bound = 10.0 * sum(max_price_per_stage) *
        sum(m.energy_factor * m.rated_discharge for m in modules)
    upper_bound = max(upper_bound, 1.0)

    sddp_model = SDDP.PolicyGraph(
        graph;
        sense = :Max,
        upper_bound = Float64(upper_bound),
        optimizer = optimizer,
    ) do subproblem, node_id
        t, k = node_id    # stage, price-node index
        price = Float64(u.price_nodes[t][k])

        @variable(subproblem, S[i = reg_idx], SDDP.State,
            initial_value = Float64(modules[i].initial_vol),
            lower_bound   = Float64(modules[i].min_vol),
            upper_bound   = Float64(modules[i].max_vol),
        )
        @variable(subproblem, 0 <= turbine[i = 1:nmod] <=
            Float64(modules[i].rated_discharge))
        @variable(subproblem, 0 <= spill[i = 1:nmod])
        @variable(subproblem, inflow_var[1:nmod])

        for i in reg_idx
            ups = upstream_of[i]
            @constraint(subproblem,
                S[i].out - S[i].in ==
                    inflow_var[i] +
                    sum(turbine[ue] + spill[ue] for ue in ups; init = 0.0) -
                    turbine[i] - spill[i]
            )
        end
        for i in 1:nmod
            i in reg_idx && continue
            ups = upstream_of[i]
            @constraint(subproblem,
                inflow_var[i] +
                sum(turbine[ue] + spill[ue] for ue in ups; init = 0.0) ==
                turbine[i] + spill[i]
            )
        end

        @stageobjective(subproblem,
            price * sum(Float64(modules[i].energy_factor) * turbine[i]
                        for i in 1:nmod)
        )

        # Inflow uncertainty: each historical year at stage `t` is one
        # equiprobable realization. Realization vector is one inflow per
        # module (assuming gauge == module index).
        realizations = [
            [Float64(u.historical_inflow[t, s, i]) for i in 1:nmod]
            for s in 1:nscen
        ]
        probs = fill(1.0 / nscen, nscen)
        SDDP.parameterize(subproblem, realizations, probs) do ω
            for i in 1:nmod
                JuMP.fix(inflow_var[i], ω[i]; force = true)
            end
        end
    end

    t_start = time()
    SDDP.train(sddp_model;
        iteration_limit = solver.max_iterations,
        print_level = 0,
    )
    t_elapsed = time() - t_start

    bound = SDDP.calculate_bound(sddp_model)
    sims = SDDP.simulate(sddp_model, solver.num_forward_scenarios)
    objective_estimate = isempty(sims) ? NaN :
        sum(sum(stage[:stage_objective] for stage in sim) for sim in sims) /
            length(sims)

    return Solution{T}(
        primal = sims,
        dual = nothing,
        objective = T(objective_estimate),
        bound = T(bound),
        status = MOI.OPTIMAL,
        iterations = solver.max_iterations,
        solve_time = t_elapsed,
        metadata = Dict{Symbol, Any}(
            :sddp_model => sddp_model,
            :variant => :ProdRiskUncertainty,
            :forward_backward_asymmetric => false,
        ),
    )
end

function _validate_prodrisk(prob::LongTermHydroProblem, u::ProdRiskUncertainty)
    nmod = length(prob.modules)
    nstages = prob.num_stages

    size(u.historical_inflow, 1) == nstages || throw(ArgumentError(
        "ProdRiskUncertainty: historical_inflow first dim " *
        "$(size(u.historical_inflow, 1)) must equal num_stages $(nstages)."))
    size(u.historical_inflow, 3) == nmod || throw(ArgumentError(
        "ProdRiskUncertainty: historical_inflow third dim (gauges) " *
        "$(size(u.historical_inflow, 3)) must equal num_modules $(nmod). " *
        "This MVP assumes one gauge per module; richer gauge↔module " *
        "mapping is a follow-up."))
    size(u.historical_inflow, 2) >= 1 || throw(ArgumentError(
        "ProdRiskUncertainty: historical_inflow needs >= 1 scenario."))

    length(u.price_nodes) == nstages || throw(ArgumentError(
        "ProdRiskUncertainty: price_nodes length " *
        "$(length(u.price_nodes)) must equal num_stages $(nstages)."))
    length(u.price_transitions) == nstages - 1 || throw(ArgumentError(
        "ProdRiskUncertainty: price_transitions length " *
        "$(length(u.price_transitions)) must equal num_stages-1 " *
        "$(nstages - 1)."))

    for (t, P) in enumerate(u.price_transitions)
        K_in = length(u.price_nodes[t])
        K_out = length(u.price_nodes[t + 1])
        size(P) == (K_in, K_out) || throw(ArgumentError(
            "ProdRiskUncertainty: price_transitions[$t] has shape " *
            "$(size(P)), expected ($K_in, $K_out)."))
        for i in 1:K_in
            row = sum(P[i, :])
            isapprox(row, 1.0; atol = 1.0e-6) || throw(ArgumentError(
                "ProdRiskUncertainty: price_transitions[$t] row $i sums " *
                "to $row, must be 1.0 (row-stochastic)."))
        end
    end

    prob.stage_prices === nothing || @warn(
        "ProdRiskUncertainty bridge ignores prob.stage_prices " *
        "(prices come from u.price_nodes / u.price_transitions). " *
        "Clear the field to silence this warning.")
    return nothing
end

# ============================================================
# Internal helpers
# ============================================================

function _validate_stage_prices(prob::LongTermHydroProblem)
    prob.stage_prices === nothing && throw(ArgumentError(
        "SDDP bridge: prob.stage_prices must be set for StagewiseIndependent " *
        "uncertainty (deterministic per-stage price forecast)."))
    length(prob.stage_prices) == prob.num_stages || throw(ArgumentError(
        "SDDP bridge: length(stage_prices) = $(length(prob.stage_prices)) " *
        "must equal num_stages = $(prob.num_stages)."))
    return prob.stage_prices
end

function _validate_stagewise(prob::LongTermHydroProblem)
    u = prob.uncertainty
    length(u.realizations) == prob.num_stages || throw(ArgumentError(
        "SDDP bridge: realizations vector length " *
        "$(length(u.realizations)) must equal num_stages $(prob.num_stages)."))
    length(u.probabilities) == prob.num_stages || throw(ArgumentError(
        "SDDP bridge: probabilities vector length must equal num_stages."))
    nmod = length(prob.modules)
    for (t, realizations_t) in enumerate(u.realizations)
        isempty(realizations_t) && throw(ArgumentError(
            "SDDP bridge: stage $t has zero realizations."))
        for (k, ω) in enumerate(realizations_t)
            length(ω) == nmod || throw(ArgumentError(
                "SDDP bridge: stage $t, realization $k has length $(length(ω)); " *
                "expected $(nmod) (one inflow value per module)."))
        end
    end
    return u.realizations, u.probabilities
end

function _extract_optimizer(solver::SDDPHydroSolver)
    sp = solver.subproblem_solver
    sp isa JuMPSolver || throw(ArgumentError(
        "SDDP bridge: subproblem_solver must be a JuMPSolver, " *
        "got $(typeof(sp))."))
    sp.optimizer === nothing && throw(ArgumentError(
        "SDDP bridge: JuMPSolver.optimizer is `nothing`; load a backend " *
        "(e.g. `using HiGHS`) and pass `JuMPSolver(HiGHS.Optimizer)`."))
    return sp.optimizer
end

end # module HydroModelsOptSDDPExt
