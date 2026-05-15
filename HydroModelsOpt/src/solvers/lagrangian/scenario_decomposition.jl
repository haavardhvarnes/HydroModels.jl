"""
Scenario-decomposed Lagrangian solver for `LongTermHydroProblem` with
`StagewiseIndependent` uncertainty.

The algorithm samples `N` scenarios from the per-stage realizations
(one realization per stage; whole-path sampling), builds a
deterministic LP per scenario, dualizes per-reservoir end-of-horizon
storage against a target, and iterates a subgradient method on the
multipliers. The expected revenue is estimated by KA-aggregated mean
across scenarios.

This is **research scaffold** rather than a production-grade
Lagrangian. The points it does establish:

- Per-scenario LP build that compiles and solves end-to-end
- A backend-agnostic aggregation kernel that runs on `CPUBackend()`
  today and would run on `GPUBackend(CUDABackend())` when a vendor
  extension is loaded
- A working `solve(::LongTermHydroProblem{T, StagewiseIndependent},
  ::LagrangianHydroSolver)` dispatch
- A subgradient step using `solver.step_size`

Deliberately deferred to C3.5+:

- Bundle method (vanilla subgradient is slow and oscillatory)
- Sophisticated primal recovery (here: simple averaging across
  scenarios)
- GPU benchmarks (the kernel is written for KA portability but
  exercised only on the CPU backend in CI)
- Dispatches for `MarkovianUncertainty`, `ProdRiskUncertainty`,
  `ScenarioTree` (stubbed with informative errors)
- Truly coupled multi-scenario problems (the current dispatch's
  Lagrangian coupling is the per-reservoir storage-target dual, which
  has a meaningful interpretation but is not a non-anticipativity
  enforcement)

References:
- Fisher (1985), Lagrangian relaxation classics.
- Helseth, Braaten (2015), parallelization of SDDP for hydro
  scheduling (Energies, open access).
"""

# ============================================================
# KA kernel for backend-portable mean aggregation
# ============================================================

# A pure-Julia per-index sum used by `_mean_with_backend`. KernelAbstractions
# generates the backend-specific implementation; the same source compiles
# on `CPU()`, `CUDABackend()`, etc.
@kernel function _accumulate_columnwise!(out, X)
    j = @index(Global)
    s = zero(eltype(out))
    @inbounds for i in 1:size(X, 1)
        s += X[i, j]
    end
    @inbounds out[j] = s
end

"""
    _backend_eltype(backend, T)

Return the element type to use on `backend` for an input of element type
`T`. Defaults to `T`. Vendor extensions can override this — e.g. the
Metal extension demotes `Float64` to `Float32` because Apple GPUs lack
native 64-bit float support.
"""
_backend_eltype(::ComputeBackend, ::Type{T}) where {T} = T

"""
    _mean_with_backend(backend, X)

Compute the column-mean of `X::AbstractMatrix` using a KernelAbstractions
kernel routed through `backend`. Returns a host `Vector{Float64}` of
length `size(X, 2)`. On `CPUBackend()` this runs threaded on the host;
on a loaded `GPUBackend(...)` it copies `X` to the device, launches the
kernel, and pulls the result back. The Metal extension demotes the
on-device element type to `Float32` (no native Float64 on Apple GPUs);
the host-side result is always promoted back to `Float64` so the outer
subgradient loop stays in Float64.
"""
function _mean_with_backend(backend::ComputeBackend, X::AbstractMatrix)
    ncols = size(X, 2)
    nrows = size(X, 1)
    T_dev = _backend_eltype(backend, eltype(X))
    if backend isa CPUBackend && T_dev === eltype(X)
        # Fast path — no copy, no conversion.
        out = HydroModelsCore.allocate(backend, T_dev, ncols)
        fill!(out, zero(T_dev))
        kernel! = _accumulate_columnwise!(ka_backend(backend))
        kernel!(out, X; ndrange = ncols)
        KernelAbstractions.synchronize(ka_backend(backend))
        return Float64.(out) ./ nrows
    end
    # GPU path (or CPU with eltype demotion): allocate on backend, copy
    # the matrix over, launch, copy result back.
    X_dev = HydroModelsCore.allocate(backend, T_dev, size(X)...)
    copyto!(X_dev, T_dev === eltype(X) ? X : T_dev.(X))
    out = HydroModelsCore.allocate(backend, T_dev, ncols)
    fill!(out, zero(T_dev))
    kernel! = _accumulate_columnwise!(ka_backend(backend))
    kernel!(out, X_dev; ndrange = ncols)
    KernelAbstractions.synchronize(ka_backend(backend))
    return Float64.(collect(out)) ./ nrows
end

# ============================================================
# Public dispatch — branch between LP path (C3) and DP path (C3.5)
# ============================================================

"""
    solve(prob::LongTermHydroProblem{T, StagewiseIndependent{T}}, solver::LagrangianHydroSolver)

Solve `prob` by scenario-decomposed Lagrangian relaxation. Branches
between the C3 LP path and the C3.5 KA Bellman-DP path according to
`_resolve_dispatch(solver)`:

- `:auto` (default) → `:dp` on any `GPUBackend`, `:lp` on `CPUBackend()`
- `:lp` → C3 path: per-scenario LP via `JuMPSolver` (HiGHS / etc.)
- `:dp` → C3.5 path: per-(scenario, reservoir) Bellman DP on the
  configured backend, with both end-of-horizon storage and (optionally)
  cross-reservoir mass balance dualised.
"""
function HydroModelsOpt.solve(
        prob::LongTermHydroProblem{T, StagewiseIndependent{T}},
        solver::LagrangianHydroSolver,
    ) where {T}
    dispatch = _resolve_dispatch(solver)
    if dispatch === :dp
        return _solve_dp_path(prob, solver)
    else
        return _solve_lp_path(prob, solver)
    end
end

"""
    _solve_lp_path(prob, solver)

C3's scenario-decomposed Lagrangian via per-scenario JuMP LP. The
public `solve` method delegates here when `solver.dispatch === :lp`.
"""
function _solve_lp_path(
        prob::LongTermHydroProblem{T, StagewiseIndependent{T}},
        solver::LagrangianHydroSolver,
    ) where {T}

    _validate_lagrangian_inputs(prob, solver)

    modules = prob.modules
    nmod = length(modules)
    is_reg = [m.reservoir_type isa RegulationReservoir for m in modules]
    reg_idx = findall(is_reg)
    nreg = length(reg_idx)
    isempty(reg_idx) && throw(ArgumentError(
        "Lagrangian solver: at least one RegulationReservoir module required"))

    upstream_of = [Int[] for _ in 1:nmod]
    for (u_edge, v_edge) in prob.topology.discharge_edges
        push!(upstream_of[v_edge], u_edge)
    end

    nstages = prob.num_stages
    stage_prices = prob.stage_prices::Vector{T}

    # ----------------------------------------------------------------
    # Sample N scenarios (one realization per stage, whole-path).
    # ----------------------------------------------------------------
    n_scen = max(1, solver.n_scenarios)
    scenarios = _enumerate_or_sample_scenarios(prob, n_scen)

    # ----------------------------------------------------------------
    # Per-reservoir end-of-horizon storage Lagrangian.
    # λ[i] is the multiplier on (S_end[i] - target_i) where target_i =
    # initial_vol_i (a simple closed-form target). Sub-objective per
    # scenario:  revenue_s − Σ_i λ[i] · S_end[i, s].
    # ----------------------------------------------------------------
    targets = Float64[Float64(modules[i].initial_vol) for i in reg_idx]
    λ = zeros(Float64, nreg)
    optimizer = _lagrangian_optimizer(solver)

    # Iteration state
    best_obj = -Inf
    last_scenario_objs = Float64[]
    last_scenario_S_end = Matrix{Float64}(undef, n_scen, nreg)

    t_start = time()
    for iter in 1:solver.max_iter
        scenario_objs = Vector{Float64}(undef, n_scen)
        S_end_matrix = Matrix{Float64}(undef, n_scen, nreg)

        for (s, inflow_path) in enumerate(scenarios)
            obj_s, S_end_s = _solve_lagrangian_subproblem(
                prob, modules, reg_idx, upstream_of,
                inflow_path, stage_prices, λ, optimizer,
            )
            scenario_objs[s] = obj_s
            S_end_matrix[s, :] = S_end_s
        end

        # KA-aggregated column means: average S_end per reservoir
        mean_S_end = _mean_with_backend(solver.backend, S_end_matrix)

        # Subgradient direction: average S_end deviation from target
        g = mean_S_end .- targets

        # Step using the configured rule
        step = compute_step(solver.step_size, iter, g,
                            Statistics.mean(scenario_objs))
        step = max(step, zero(step))   # subgradient updates use non-neg step
        λ .+= step .* g

        # The Lagrangian objective is the sum (or mean) of subproblem
        # objectives at the current multiplier; track the best seen.
        mean_obj = Statistics.mean(scenario_objs)
        if mean_obj > best_obj
            best_obj = mean_obj
        end

        last_scenario_objs = scenario_objs
        last_scenario_S_end = S_end_matrix

        # Convergence check: tiny step or tiny subgradient
        if maximum(abs, g) < solver.tol
            break
        end
    end
    t_elapsed = time() - t_start

    return Solution{T}(
        primal = (
            scenario_objectives = last_scenario_objs,
            scenario_S_end      = last_scenario_S_end,
        ),
        dual = (
            multipliers = copy(λ),
            targets     = targets,
        ),
        objective  = T(best_obj),
        bound      = T(NaN),   # no proper dual bound from vanilla subgradient
        status     = MOI.OTHER_LIMIT,
        iterations = solver.max_iter,
        solve_time = t_elapsed,
        metadata = Dict{Symbol, Any}(
            :variant     => :LagrangianStagewiseIndependent,
            :n_scenarios => n_scen,
            :ka_backend  => solver.backend,
            :dispatch    => :lp,
        ),
    )
end

# ============================================================
# Internal: per-scenario LP build + solve
# ============================================================

# Build and solve a single deterministic-scenario LP. Returns
# `(scenario_objective, S_end_per_reg_reservoir::Vector{Float64})`.
function _solve_lagrangian_subproblem(
        prob::LongTermHydroProblem,
        modules, reg_idx, upstream_of,
        inflow_path::Vector{Vector{Float64}},
        stage_prices, λ::Vector{Float64},
        optimizer,
    )

    nmod = length(modules)
    nstages = prob.num_stages
    nreg = length(reg_idx)
    reg_position = Dict(reg_idx[i] => i for i in 1:nreg)

    model = JuMP.Model(optimizer)
    try
        JuMP.set_optimizer_attribute(model, "output_flag", false)
    catch
        # not all optimizers accept this attribute; ignore
    end

    @variable(model, S[i = reg_idx, t = 1:nstages] >= Float64(modules[i].min_vol))
    @variable(model, S_end[reg_idx] >= 0)
    @variable(model, 0 <= turbine[i = 1:nmod, t = 1:nstages] <=
        Float64(modules[i].rated_discharge))
    @variable(model, 0 <= spill[i = 1:nmod, t = 1:nstages])

    # Upper bounds and initial conditions for storage
    for i in reg_idx, t in 1:nstages
        ub = Float64(modules[i].max_vol)
        isfinite(ub) && @constraint(model, S[i, t] <= ub)
    end
    for i in reg_idx
        @constraint(model, S[i, 1] == Float64(modules[i].initial_vol))
    end

    # Storage continuity per regulation reservoir
    for i in reg_idx, t in 1:(nstages - 1)
        ups = upstream_of[i]
        @constraint(model,
            S[i, t + 1] - S[i, t] ==
                inflow_path[t][i] +
                sum(turbine[u, t] + spill[u, t] for u in ups; init = 0.0) -
                turbine[i, t] - spill[i, t]
        )
    end

    # Final-stage mass-balance defines S_end
    for i in reg_idx
        ups = upstream_of[i]
        @constraint(model,
            S_end[i] == S[i, nstages] +
                inflow_path[nstages][i] +
                sum(turbine[u, nstages] + spill[u, nstages] for u in ups; init = 0.0) -
                turbine[i, nstages] - spill[i, nstages]
        )
    end

    # Pass-through mass balance for BufferReservoirs (no storage)
    for i in 1:nmod
        i in reg_idx && continue
        ups = upstream_of[i]
        for t in 1:nstages
            @constraint(model,
                inflow_path[t][i] +
                sum(turbine[u, t] + spill[u, t] for u in ups; init = 0.0) ==
                turbine[i, t] + spill[i, t]
            )
        end
    end

    # Lagrangian objective: revenue − Σ_i λ[i] · S_end[i]
    @objective(model, Max,
        sum(stage_prices[t] *
            sum(Float64(modules[i].energy_factor) * turbine[i, t]
                for i in 1:nmod) for t in 1:nstages)
        - sum(λ[reg_position[i]] * S_end[i] for i in reg_idx)
    )

    JuMP.optimize!(model)

    obj = JuMP.has_values(model) ? JuMP.objective_value(model) : -Inf
    S_end_vals = Vector{Float64}(undef, nreg)
    if JuMP.has_values(model)
        for i in reg_idx
            S_end_vals[reg_position[i]] = JuMP.value(S_end[i])
        end
    else
        fill!(S_end_vals, NaN)
    end
    return obj, S_end_vals
end

# ============================================================
# Internal: scenario enumeration / sampling
# ============================================================

# Sample one whole-path inflow scenario per realization slot at stage 1.
# (Simple Monte Carlo: at each stage independently choose realization
# index `k = mod1(slot, length(realizations[t]))`.) Returns a vector
# of `n_scen` paths, each path being `Vector{Vector{Float64}}` of
# length `nstages`.
function _enumerate_or_sample_scenarios(prob::LongTermHydroProblem, n_scen::Int)
    u = prob.uncertainty
    nstages = prob.num_stages
    paths = Vector{Vector{Vector{Float64}}}(undef, n_scen)
    for s in 1:n_scen
        path = Vector{Vector{Float64}}(undef, nstages)
        for t in 1:nstages
            K = length(u.realizations[t])
            k = mod1(s, K)
            path[t] = Float64.(u.realizations[t][k])
        end
        paths[s] = path
    end
    return paths
end

# ============================================================
# Validation
# ============================================================

function _validate_lagrangian_inputs(
        prob::LongTermHydroProblem{T, StagewiseIndependent{T}},
        solver::LagrangianHydroSolver,
    ) where {T}
    prob.stage_prices === nothing && throw(ArgumentError(
        "Lagrangian solver: prob.stage_prices must be set " *
        "(deterministic per-stage price forecast)."))
    length(prob.stage_prices) == prob.num_stages || throw(ArgumentError(
        "Lagrangian solver: length(stage_prices) must equal num_stages."))
    sp = solver.subproblem_solver
    sp isa JuMPSolver || throw(ArgumentError(
        "Lagrangian solver: subproblem_solver must be a JuMPSolver " *
        "(got $(typeof(sp)))."))
    sp.optimizer === nothing && throw(ArgumentError(
        "Lagrangian solver: subproblem_solver.optimizer is `nothing`; " *
        "load a backend (e.g. `using HiGHS`) and pass JuMPSolver(HiGHS.Optimizer)."))
    return nothing
end

function _lagrangian_optimizer(solver::LagrangianHydroSolver)
    return solver.subproblem_solver.optimizer
end

# ============================================================
# Stubs for non-StagewiseIndependent paths
# ============================================================

function HydroModelsOpt.solve(
        ::LongTermHydroProblem{T, U},
        ::LagrangianHydroSolver,
    ) where {T, U <: UncertaintyModel{T}}
    throw(ErrorException(
        "Lagrangian solver: dispatch for $(U) not yet implemented. " *
        "Currently only StagewiseIndependent is supported; see " *
        "HydroModelsOpt/CLAUDE.md for the C3.5 follow-up plan."))
end
