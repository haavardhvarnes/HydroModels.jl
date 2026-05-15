"""
GPU-native Lagrangian solver for `LongTermHydroProblem` with
`StagewiseIndependent` uncertainty (C3.5 milestone).

Builds on C3's scenario-decomposed Lagrangian by dualising **both**
the end-of-horizon storage targets (multipliers `λ`) and the
cross-reservoir mass-balance equalities (multipliers `μ`). After both
dualisations, each `(scenario, reservoir)` subproblem decouples to a
1-D finite-horizon optimal-control problem in storage `S`, solvable
by backward Bellman recursion on a discretised grid — implemented as
a KA kernel that runs natively on every backend.

The C3 LP path stays available via `solver.dispatch = :lp`; C3.5
becomes the default on any `GPUBackend`.

# Caveats (documented for the user)

1. The dual is non-smooth; vanilla subgradient is `O(1/√k)`. A bundle
   method (`solver.use_bundle = true`) is the natural follow-up. C3.5
   lands correctness, not convergence speed.
2. Relaxed DP solutions can violate mass balance until `μ` converges.
   The primal returned is the DP forward simulation along the optimal
   policy — not a feasible primal until the dual converges. Sophisticated
   primal recovery (e.g. per-scenario LP re-solve fixing `(λ, μ)`) is
   a C3.6 follow-up.
3. BufferReservoir support is deferred to C3.6. The validation
   function rejects problems with non-RegulationReservoir modules
   when `dispatch = :dp`.
"""

# ============================================================
# Public dispatch — picks LP or DP path based on solver.dispatch
# ============================================================

# Top-level dispatch is defined in `scenario_decomposition.jl` and
# branches to either `_solve_lp_path` (C3) or `_solve_dp_path` (this
# file) based on `_resolve_dispatch(solver)`.

# ============================================================
# Orchestrator
# ============================================================

function _solve_dp_path(
        prob::LongTermHydroProblem{T, StagewiseIndependent{T}},
        solver::LagrangianHydroSolver,
    ) where {T}

    _validate_dp_inputs(prob, solver)

    backend       = solver.backend
    ka            = ka_backend(backend)
    T_dev         = _backend_eltype(backend, Float64)
    modules       = prob.modules
    nmod          = length(modules)
    nstages       = prob.num_stages
    K             = solver.K_grid
    is_reg        = [m.reservoir_type isa RegulationReservoir for m in modules]
    n_reg         = count(is_reg)
    n_scen        = max(1, solver.n_scenarios)
    stage_prices  = prob.stage_prices::Vector{T}
    use_flow_dual = solver.dual_flow_coupling

    # ----------------------------------------------------------------
    # Build host-side flat arrays then push to device.
    # ----------------------------------------------------------------
    S_grid_host = Matrix{T_dev}(undef, K, nmod)
    for i in 1:nmod
        Smin = T_dev(modules[i].min_vol)
        Smax = T_dev(modules[i].max_vol)
        # Uniform K-point grid; degenerate to the single-point grid when
        # Smax == Smin (BufferReservoir or fixed reservoir).
        if Smax == Smin
            S_grid_host[:, i] .= Smin
        else
            for k in 1:K
                frac = T_dev(k - 1) / T_dev(K - 1)
                S_grid_host[k, i] = Smin + frac * (Smax - Smin)
            end
        end
    end

    η_host       = T_dev[modules[i].energy_factor for i in 1:nmod]
    qmax_host    = T_dev[modules[i].rated_discharge for i in 1:nmod]
    Smin_host    = T_dev[modules[i].min_vol for i in 1:nmod]
    Smax_host    = T_dev[modules[i].max_vol for i in 1:nmod]
    S0_host      = T_dev[modules[i].initial_vol for i in 1:nmod]
    reg_mask_h   = is_reg

    # Downstream module index: 0 means "outflows to nowhere"
    downstream_of_host = zeros(Int32, nmod)
    name_to_idx = Dict(modules[i].name => i for i in 1:nmod)
    for i in 1:nmod
        m = modules[i]
        if m.discharge_to !== nothing && haskey(name_to_idx, m.discharge_to)
            downstream_of_host[i] = Int32(name_to_idx[m.discharge_to])
        end
    end

    # Edges (u → v): used by the flow-subgradient kernel
    edge_u_host = Int32[]
    edge_v_host = Int32[]
    for (u, v) in prob.topology.discharge_edges
        push!(edge_u_host, Int32(u))
        push!(edge_v_host, Int32(v))
    end

    scenarios = _enumerate_or_sample_scenarios(prob, n_scen)
    inflow_host = Array{T_dev, 3}(undef, nstages, nmod, n_scen)
    for s in 1:n_scen, t in 1:nstages, i in 1:nmod
        inflow_host[t, i, s] = T_dev(scenarios[s][t][i])
    end

    price_host = T_dev.(stage_prices)

    # ----------------------------------------------------------------
    # Allocate device arrays.
    # ----------------------------------------------------------------
    S_grid_dev      = HydroModelsCore.allocate(backend, T_dev, K, nmod)
    copyto!(S_grid_dev, S_grid_host)
    η_dev           = _to_device(backend, η_host)
    qmax_dev        = _to_device(backend, qmax_host)
    Smin_dev        = _to_device(backend, Smin_host)
    Smax_dev        = _to_device(backend, Smax_host)
    S0_dev          = _to_device(backend, S0_host)
    reg_mask_dev    = _to_device(backend, reg_mask_h)
    downstream_dev  = _to_device(backend, downstream_of_host)
    edge_u_dev      = _to_device(backend, edge_u_host)
    edge_v_dev      = _to_device(backend, edge_v_host)
    inflow_dev      = _to_device(backend, inflow_host)
    price_dev       = _to_device(backend, price_host)

    # Per-stage μ_t and "downstream's μ_t" extracted from a (T_stg, nmod, n_scen) array.
    μ_dev = HydroModelsCore.allocate(backend, T_dev, nstages, nmod, n_scen)
    fill!(μ_dev, zero(T_dev))

    # λ as a (nmod,) device array (zeros at non-Reg positions for
    # simplicity; only the reg slots will move).
    λ_dev = HydroModelsCore.allocate(backend, T_dev, nmod)
    fill!(λ_dev, zero(T_dev))

    # Two-buffer ping-pong for V (only V_t and V_{t+1} are alive at a time).
    V_buf_a = HydroModelsCore.allocate(backend, T_dev, K, nmod, n_scen)
    V_buf_b = HydroModelsCore.allocate(backend, T_dev, K, nmod, n_scen)
    fill!(V_buf_a, zero(T_dev))
    fill!(V_buf_b, zero(T_dev))

    # Full 4D policy tables (memory limit is K · T · n_reg · n_scen · 4 B).
    π_q_dev    = HydroModelsCore.allocate(backend, T_dev,  K, nstages, nmod, n_scen)
    π_σ_dev    = HydroModelsCore.allocate(backend, T_dev,  K, nstages, nmod, n_scen)
    π_next_dev = HydroModelsCore.allocate(backend, Int32,  K, nstages, nmod, n_scen)
    fill!(π_q_dev,    zero(T_dev))
    fill!(π_σ_dev,    zero(T_dev))
    fill!(π_next_dev, Int32(1))

    # Trajectory arrays from forward simulation.
    S_traj_dev = HydroModelsCore.allocate(backend, T_dev, nstages + 1, nmod, n_scen)
    q_traj_dev = HydroModelsCore.allocate(backend, T_dev, nstages,     nmod, n_scen)
    σ_traj_dev = HydroModelsCore.allocate(backend, T_dev, nstages,     nmod, n_scen)
    g_μ_dev    = HydroModelsCore.allocate(backend, T_dev, nstages,     nmod, n_scen)

    targets = T_dev[modules[i].initial_vol for i in 1:nmod]   # host

    # ----------------------------------------------------------------
    # Outer subgradient loop.
    # ----------------------------------------------------------------
    last_objective = T(0)
    iters_done     = 0
    t_start        = time()

    for iter in 1:solver.max_iter
        iters_done = iter

        # ------- Backward Bellman sweep ----------------------------
        # Terminal: V_{T+1}(S) = −λ_i · S
        terminal_kernel = bellman_dp_terminal!(ka)
        terminal_kernel(V_buf_a, S_grid_dev, λ_dev, reg_mask_dev;
                        ndrange = (K, nmod, n_scen))
        KernelAbstractions.synchronize(ka)

        V_next_buf = V_buf_a
        V_curr_buf = V_buf_b
        bellman_kernel = bellman_dp_kernel!(ka)

        for t in nstages:-1:1
            # Extract slices of μ for the local and the downstream module.
            # Use device-resident temporary arrays of shape (nmod, n_scen).
            μ_t_dev   = _stage_slice(backend, μ_dev, t, nmod, n_scen)
            μ_d_t_dev = _downstream_slice(backend, μ_dev, t, downstream_of_host,
                                          nmod, n_scen)
            inflow_t_dev = _stage_slice(backend, inflow_dev, t, nmod, n_scen)

            bellman_kernel(
                V_curr_buf, V_next_buf,
                π_q_dev, π_σ_dev, π_next_dev,
                inflow_t_dev, T_dev(price_host[t]),
                μ_t_dev, μ_d_t_dev,
                S_grid_dev,
                η_dev, qmax_dev, Smin_dev, Smax_dev,
                reg_mask_dev,
                Int32(t);
                ndrange = (K, nmod, n_scen),
            )
            KernelAbstractions.synchronize(ka)
            V_next_buf, V_curr_buf = V_curr_buf, V_next_buf
        end

        # ------- Forward simulate from S0 along the policy --------
        forward_kernel = forward_simulate_kernel!(ka)
        forward_kernel(
            S_traj_dev, q_traj_dev, σ_traj_dev,
            π_q_dev, π_σ_dev, π_next_dev,
            S0_dev, S_grid_dev,
            reg_mask_dev;
            ndrange = (nmod, n_scen),
        )
        KernelAbstractions.synchronize(ka)

        # ------- Compute primal objective on host -----------------
        # We bring the trajectory back to the host for the objective sum
        # and primal record. For very large problems this is a host
        # bottleneck — a future C3.5+ revision would do the objective
        # reduction on device.
        S_traj_host = Array(S_traj_dev)
        q_traj_host = Array(q_traj_dev)
        σ_traj_host = Array(σ_traj_dev)

        # Lagrangian objective per scenario: revenue − Σ_i λ_i · S_end_i.
        # Matches the JuMP objective the C3 LP path returns, so the
        # two dispatches report comparable numbers.
        λ_host_iter = Float64.(Array(λ_dev))
        scenario_objs = Vector{Float64}(undef, n_scen)
        for s in 1:n_scen
            rev = 0.0
            for t in 1:nstages, i in 1:nmod
                rev += stage_prices[t] *
                       Float64(η_host[i]) * Float64(q_traj_host[t, i, s])
            end
            lagr = rev
            for i in 1:nmod
                if is_reg[i]
                    lagr -= λ_host_iter[i] *
                            Float64(S_traj_host[nstages + 1, i, s])
                end
            end
            scenario_objs[s] = lagr
        end
        mean_obj = Statistics.mean(scenario_objs)
        last_objective = T(mean_obj)

        # ------- Storage subgradient (host arithmetic) -------------
        # g_λ_i = mean_s S_end_{i,s} − target_i  (only over Reg modules)
        g_λ_host = zeros(Float64, nmod)
        for i in 1:nmod
            if is_reg[i]
                acc = 0.0
                for s in 1:n_scen
                    acc += Float64(S_traj_host[nstages + 1, i, s])
                end
                g_λ_host[i] = acc / n_scen - Float64(targets[i])
            end
        end

        # Update λ on device via the existing subgradient kernel.
        # Equality dualisation of `S_end == target` ⇒ λ ∈ R; this
        # matches the C3 LP path's behaviour.
        step_λ = compute_step(solver.step_size, iter, g_λ_host, mean_obj)
        step_λ = max(step_λ, zero(step_λ))
        g_λ_dev = _to_device(backend, T_dev.(g_λ_host))
        subgradient_update!(ka, λ_dev, g_λ_dev, T_dev(step_λ);
                            lower = T_dev(-Inf), upper = T_dev(Inf))

        # ------- Flow subgradient (device kernel) ------------------
        g_μ_max = 0.0
        if use_flow_dual
            flow_kernel = compute_subgradient_flow_kernel!(ka)
            flow_kernel(
                g_μ_dev,
                S_traj_dev, q_traj_dev, σ_traj_dev,
                inflow_dev,
                edge_u_dev, edge_v_dev;
                ndrange = (nstages, nmod, n_scen),
            )
            KernelAbstractions.synchronize(ka)

            # Update μ on device — flatten to 1D for the subgradient kernel.
            μ_flat_dev   = _flatten(μ_dev)
            g_μ_flat_dev = _flatten(g_μ_dev)
            step_μ = step_λ   # use the same step rule for both (simpler; ok for vanilla subgrad)
            subgradient_update!(ka, μ_flat_dev, g_μ_flat_dev, T_dev(step_μ);
                                lower = T_dev(-Inf), upper = T_dev(Inf))

            g_μ_max = maximum(abs, Array(g_μ_dev))
        end

        # ------- Convergence test ---------------------------------
        g_λ_max = maximum(abs, g_λ_host; init = 0.0)
        if max(g_λ_max, g_μ_max) < solver.tol
            break
        end
    end
    t_elapsed = time() - t_start

    # ----------------------------------------------------------------
    # Build the Solution record.
    # ----------------------------------------------------------------
    S_traj_host = Array(S_traj_dev)
    q_traj_host = Array(q_traj_dev)
    σ_traj_host = Array(σ_traj_dev)
    λ_host      = Float64.(Array(λ_dev))
    μ_host      = use_flow_dual ? Float64.(Array(μ_dev)) : nothing

    scenario_S_end = Matrix{Float64}(undef, n_scen, count(is_reg))
    reg_positions  = findall(is_reg)
    for s in 1:n_scen, j in eachindex(reg_positions)
        i = reg_positions[j]
        scenario_S_end[s, j] = Float64(S_traj_host[nstages + 1, i, s])
    end

    return Solution{T}(
        primal = (
            S_traj             = S_traj_host,
            q_traj             = q_traj_host,
            σ_traj             = σ_traj_host,
            scenario_S_end     = scenario_S_end,
        ),
        dual = (
            λ            = λ_host,
            μ            = μ_host,
            targets      = Float64.(targets),
        ),
        objective  = last_objective,
        bound      = T(NaN),
        status     = MOI.OTHER_LIMIT,
        iterations = iters_done,
        solve_time = t_elapsed,
        metadata = Dict{Symbol, Any}(
            :variant            => :LagrangianDP,
            :n_scenarios        => n_scen,
            :K_grid             => K,
            :dual_flow_coupling => use_flow_dual,
            :ka_backend         => backend,
            :dispatch           => :dp,
        ),
    )
end

# ============================================================
# Validation
# ============================================================

function _validate_dp_inputs(
        prob::LongTermHydroProblem{T, StagewiseIndependent{T}},
        solver::LagrangianHydroSolver,
    ) where {T}

    prob.stage_prices === nothing && throw(ArgumentError(
        "Lagrangian DP: prob.stage_prices must be set " *
        "(deterministic per-stage price forecast)."))
    length(prob.stage_prices) == prob.num_stages || throw(ArgumentError(
        "Lagrangian DP: length(stage_prices) must equal num_stages."))

    # C3.5 milestone scope — RegulationReservoir only.
    for m in prob.modules
        m.reservoir_type isa RegulationReservoir || throw(ArgumentError(
            "Lagrangian DP: BufferReservoir support is deferred to C3.6. " *
            "Module $(m.name) has reservoir_type = $(m.reservoir_type)."))
    end

    solver.K_grid >= 2 || throw(ArgumentError(
        "Lagrangian DP: K_grid must be ≥ 2 (got $(solver.K_grid))."))
    return nothing
end

# ============================================================
# Helpers: host → device transfer, slice extraction, flatten
# ============================================================

# Copy a host array to the active backend, respecting `_backend_eltype`
# for Float demotion. Integers are passed through unchanged.
function _to_device(backend::ComputeBackend, A::AbstractArray{T}) where {T}
    T_dev = _backend_eltype(backend, T)
    if backend isa CPUBackend && T_dev === T
        return A
    end
    dev = HydroModelsCore.allocate(backend, T_dev, size(A)...)
    copyto!(dev, T_dev === T ? A : T_dev.(A))
    return dev
end

# Extract a `(n_dim2, n_dim3)` slice from a `(nstages, n_dim2, n_dim3)`
# device array. We do this by allocating a fresh device buffer and
# copying — concrete-array semantics, no views (so KA dispatch works
# on every backend).
function _stage_slice(backend::ComputeBackend,
                     A::AbstractArray{T, 3}, t::Int,
                     n_dim2::Int, n_dim3::Int) where {T}
    out = HydroModelsCore.allocate(backend, T, n_dim2, n_dim3)
    if backend isa CPUBackend
        @inbounds for j in 1:n_dim3, i in 1:n_dim2
            out[i, j] = A[t, i, j]
        end
    else
        # On GPU, copy through the host. Acceptable for n_dim2 × n_dim3
        # in the low thousands; revisit when needed.
        host_slice = Array{T}(undef, n_dim2, n_dim3)
        host_full  = Array(A)
        @inbounds for j in 1:n_dim3, i in 1:n_dim2
            host_slice[i, j] = host_full[t, i, j]
        end
        copyto!(out, host_slice)
    end
    return out
end

# Like `_stage_slice` but reads the downstream module's row:
# `out[i, s] = A[t, downstream_of[i], s]` when downstream exists,
# else 0. `downstream_of_host` is a host `Vector{Int32}` indexed
# by source module.
function _downstream_slice(backend::ComputeBackend,
                          A::AbstractArray{T, 3}, t::Int,
                          downstream_of_host::Vector{Int32},
                          n_dim2::Int, n_dim3::Int) where {T}
    out = HydroModelsCore.allocate(backend, T, n_dim2, n_dim3)
    host_full = backend isa CPUBackend ? A : Array(A)
    host_slice = Array{T}(undef, n_dim2, n_dim3)
    @inbounds for j in 1:n_dim3, i in 1:n_dim2
        d = downstream_of_host[i]
        host_slice[i, j] = d == 0 ? zero(T) : host_full[t, Int(d), j]
    end
    if backend isa CPUBackend
        copyto!(out, host_slice)
    else
        copyto!(out, host_slice)
    end
    return out
end

# Flatten a device array to 1D in-place (alias if possible).
function _flatten(A::AbstractArray)
    return reshape(A, length(A))
end
