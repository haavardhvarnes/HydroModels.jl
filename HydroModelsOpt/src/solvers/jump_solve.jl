"""
JuMP solver: invoke the LP / MILP, extract values, and assemble the
output tables. Ported from `HydroModels_depr/src/solve.jl` with two
adaptations:

- All output tables are `NamedTuple`s that satisfy the Tables.jl
  interface, *not* `DataFrame`s. Users who want DataFrames convert at
  the boundary: `DataFrame(sol.storage)`.
- The solution wraps the parametric `HydroSolution{T}` rather than
  depr's `Float64`-only `HydroSolution`. Currently the LP works in
  `Float64`; `T` exists for forward compatibility.

Public entry point: [`solve(::ShopShortTermProblem, ::JuMPSolver)`](@ref).
"""

# ============================================================
# Solution container
# ============================================================

"""
    HydroSolution{T}

Solution of a `ShopShortTermProblem`. Holds a reference to the
originating problem, the JuMP termination status, the objective value,
and 11 long-format tables.

Every table field satisfies the **Tables.jl** interface — pass any of
them to `DataFrame`, `CSV.write`, Arrow, Makie, etc. without conversion.
The `problem` field is kept so downstream consumers (notably
`HydroModelsViz`) can access static data like reservoir bounds,
water-value cuts, and topology without re-parsing the YAML.

# Fields
- `problem::ShopShortTermProblem{T}` — originating problem.
- `termination::String` — JuMP termination status (e.g. `"OPTIMAL"`).
- `objective::T` — best known objective value.
- `storage` — columns `(time, reservoir, S, H)`.
- `spill` — columns `(time, reservoir, Q_spill)`.
- `q_gen` — columns `(time, generator, Q)`.
- `p_gen` — columns `(time, generator, P)`.
- `q_pump` — columns `(time, pump, Q)`.
- `e_pump` — columns `(time, pump, E)`.
- `q_tunnel` — columns `(time, tunnel, Q)`.
- `revenue` — columns `(time, price, gen_MWh, pump_MWh, revenue)`.
- `r_alloc` — columns `(time, generator, product, R_MW)`.
- `r_slack` — columns `(time, group, product, slack_MW)`.
- `water_value` — either `(group, W_EUR)` (cut-group form) or
  `(reservoir, W_EUR)` (per-reservoir form), depending on which cuts
  the problem carried.
- `junctions` — columns `(time, junction, Q_junc, R_min_slack)` —
  B1.7 typed-junction (KP) flow plus the soft-min slack. Empty when
  no junctions are declared.
"""
struct HydroSolution{T}
    problem::ShopShortTermProblem{T}
    termination::String
    objective::T
    storage::NamedTuple
    spill::NamedTuple
    q_gen::NamedTuple
    p_gen::NamedTuple
    q_pump::NamedTuple
    e_pump::NamedTuple
    q_tunnel::NamedTuple
    revenue::NamedTuple
    r_alloc::NamedTuple
    r_slack::NamedTuple
    water_value::NamedTuple
    junctions::NamedTuple
end

# ============================================================
# Long-format table builders
# ============================================================

# `_tidy2(names, times, X, valname, namecol)` builds a long table
# `(time, namecol, valname)` from a matrix `X[i, j]`.
function _tidy2(names::Vector{String}, times::Vector{DateTime},
                X::AbstractMatrix, valname::Symbol, namecol::Symbol)
    n_n = length(names)
    n_t = length(times)
    if n_n == 0 || n_t == 0
        return NamedTuple{(:time, namecol, valname)}((
            DateTime[], String[], Float64[],
        ))
    end
    total = n_n * n_t
    tcol = Vector{DateTime}(undef, total)
    ncol = Vector{String}(undef, total)
    vcol = Vector{Float64}(undef, total)
    k = 1
    for (i, nm) in enumerate(names), (j, t) in enumerate(times)
        tcol[k] = t
        ncol[k] = nm
        vcol[k] = X[i, j]
        k += 1
    end
    return NamedTuple{(:time, namecol, valname)}((tcol, ncol, vcol))
end

# Join two long tables `(time, name, valA)` and `(time, name, valB)` →
# `(time, name, valA, valB)`. Tables are produced from the same matrix
# layout so a positional zip is correct (both visit `(name_i, t_j)` in
# the same order).
function _join_on_time_name(
        a::NamedTuple, b::NamedTuple, namecol::Symbol,
        avalname::Symbol, bvalname::Symbol,
    )
    @assert getproperty(a, :time) == getproperty(b, :time)
    @assert getproperty(a, namecol) == getproperty(b, namecol)
    cols = (
        :time      => getproperty(a, :time),
        namecol    => getproperty(a, namecol),
        avalname   => getproperty(a, avalname),
        bvalname   => getproperty(b, bvalname),
    )
    return (; cols...)
end

function _empty_r_alloc()
    return (
        time = DateTime[],
        generator = String[],
        product = String[],
        R_MW = Float64[],
    )
end

function _empty_r_slack()
    return (
        time = DateTime[],
        group = String[],
        product = String[],
        slack_MW = Float64[],
    )
end

function _empty_water_value_per_res()
    return (reservoir = String[], W_EUR = Float64[])
end

function _empty_water_value_per_group()
    return (group = String[], W_EUR = Float64[])
end

function _empty_revenue(times::Vector{DateTime}, price::Vector{Float64})
    nt = length(times)
    return (
        time     = times,
        price    = price,
        gen_MWh  = fill(NaN, nt),
        pump_MWh = fill(NaN, nt),
        revenue  = fill(NaN, nt),
    )
end

function _build_r_alloc(lp::_ShortTermLPModel, model::JuMP.Model)
    (isempty(lp.PROD) || !haskey(model, :R)) && return _empty_r_alloc()
    R_val = JuMP.value.(model[:R])
    t_out = DateTime[]
    g_out = String[]
    p_out = String[]
    rv_out = Float64[]
    for u in 1:length(lp.GEN), j in 1:length(lp.T), p in 1:length(lp.PROD)
        lp.g_res_pmax[u, p] == 0 && continue
        push!(t_out, lp.T[j])
        push!(g_out, lp.GEN[u])
        push!(p_out, lp.PROD[p])
        push!(rv_out, R_val[u, j, p])
    end
    return (time = t_out, generator = g_out, product = p_out, R_MW = rv_out)
end

function _build_r_slack(lp::_ShortTermLPModel, model::JuMP.Model)
    (isempty(lp.RG) || !haskey(model, :Rslack)) && return _empty_r_slack()
    Rslack_val = JuMP.value.(model[:Rslack])
    t_out = DateTime[]
    g_out = String[]
    p_out = String[]
    s_out = Float64[]
    for g in 1:length(lp.RG), j in 1:length(lp.T)
        push!(t_out, lp.T[j])
        push!(g_out, lp.RG[g])
        push!(p_out, lp.PROD[lp.rg_prod[g]])
        push!(s_out, Rslack_val[g, j])
    end
    return (time = t_out, group = g_out, product = p_out, slack_MW = s_out)
end

function _build_water_value(lp::_ShortTermLPModel, model::JuMP.Model)
    if !isempty(lp.cut_groups) && haskey(model, :W_G)
        W_G = JuMP.value.(model[:W_G])
        return (
            group = [cg.name for cg in lp.cut_groups],
            W_EUR = [W_G[g] for g in 1:length(lp.cut_groups)],
        )
    end
    wv_res_indices = findall(>(0), lp.wv_ncuts)
    (isempty(wv_res_indices) || !haskey(model, :W)) && return _empty_water_value_per_res()
    W_val = JuMP.value.(model[:W])
    return (
        reservoir = [lp.R[i] for i in wv_res_indices],
        W_EUR     = [W_val[i] for i in wv_res_indices],
    )
end

# ============================================================
# Public entry point
# ============================================================

"""
    solve(prob::ShopShortTermProblem, solver::JuMPSolver)

Build the JuMP LP / MILP, optimize with the wrapped solver, and assemble
a `HydroSolution{Float64}`. The solver passes `optimizer` through to
JuMP; any MOI-compatible optimizer (HiGHS, Gurobi, CPLEX, …) works.

When `optimizer` is `nothing` (the default if none is wired), the model
is built but `optimize!` is not called; `termination` is `"NOT_CALLED"`
and the solution tables hold `NaN`. Useful for inspecting the model
without solving.
"""
function solve(prob::ShopShortTermProblem, solver::JuMPSolver)
    lp, model = _build_short_term_lp(prob, solver.optimizer)

    if solver.optimizer === nothing
        return _empty_solution(prob, lp, "NOT_CALLED", NaN)
    end

    # Apply user-provided solver options (overriding the defaults set in
    # _build_short_term_lp).
    for (k, v) in solver.options
        try
            JuMP.set_optimizer_attribute(model, string(k), v)
        catch
            # Solver may not support this attribute; non-fatal.
        end
    end

    JuMP.optimize!(model)
    term = string(JuMP.termination_status(model))
    obj = try
        JuMP.objective_value(model)
    catch
        NaN
    end

    if !JuMP.has_values(model)
        @warn "No primal solution available (termination=$term). Returning NaN solution."
        return _empty_solution(prob, lp, term, obj)
    end

    S = JuMP.value.(model[:S])
    H = JuMP.value.(model[:H])
    Qspill = JuMP.value.(model[:Qspill])
    Qg = JuMP.value.(model[:Qg])
    Pg = JuMP.value.(model[:Pg])
    Qpump = JuMP.value.(model[:Qpump])
    Ppump = JuMP.value.(model[:Ppump])
    Qtun = JuMP.value.(model[:Qtun])
    Qjunc = JuMP.value.(model[:Qjunc])
    Rj_min = JuMP.value.(model[:Rj_min])

    storageS = _tidy2(lp.R, lp.T, S, :S, :reservoir)
    storageH = _tidy2(lp.R, lp.T, H, :H, :reservoir)
    storage = _join_on_time_name(storageS, storageH, :reservoir, :S, :H)
    spill = _tidy2(lp.R, lp.T, Qspill, :Q_spill, :reservoir)
    q_gen = _tidy2(lp.GEN, lp.T, Qg, :Q, :generator)
    p_gen = _tidy2(lp.GEN, lp.T, Pg, :P, :generator)
    q_pump = _tidy2(lp.PUMP, lp.T, Qpump, :Q, :pump)
    e_pump = _tidy2(lp.PUMP, lp.T, Ppump, :E, :pump)
    q_tunnel = _tidy2(lp.TUN, lp.T, Qtun, :Q, :tunnel)

    nt = length(lp.T)
    gen_MWh = [sum(Pg[:, j]) * lp.dt_hours[j] for j in 1:nt]
    pump_MWh = [sum(Ppump[:, j]) * lp.dt_hours[j] for j in 1:nt]
    revenue_col = lp.price .* (gen_MWh .- pump_MWh)
    revenue = (
        time     = lp.T,
        price    = lp.price,
        gen_MWh  = gen_MWh,
        pump_MWh = pump_MWh,
        revenue  = revenue_col,
    )

    r_alloc = _build_r_alloc(lp, model)
    r_slack = _build_r_slack(lp, model)
    water_value = _build_water_value(lp, model)

    # B1.7 — junction flow + soft-min slack table.
    njunc = length(lp.JUNC)
    if njunc == 0
        junctions = (time = DateTime[], junction = String[],
                     Q_junc = Float64[], R_min_slack = Float64[])
    else
        n_pairs = njunc * nt
        j_time      = Vector{DateTime}(undef, n_pairs)
        j_name      = Vector{String}(undef, n_pairs)
        j_Q         = Vector{Float64}(undef, n_pairs)
        j_R         = Vector{Float64}(undef, n_pairs)
        idx = 1
        for k in 1:njunc, j in 1:nt
            j_time[idx] = lp.T[j]
            j_name[idx] = lp.JUNC[k]
            j_Q[idx]    = Qjunc[k, j]
            j_R[idx]    = Rj_min[k, j]
            idx += 1
        end
        junctions = (time = j_time, junction = j_name,
                     Q_junc = j_Q, R_min_slack = j_R)
    end

    return HydroSolution{Float64}(
        prob, term, obj,
        storage, spill, q_gen, p_gen, q_pump, e_pump, q_tunnel,
        revenue, r_alloc, r_slack, water_value, junctions,
    )
end

# Empty solution shell — used when no primal is available.
function _empty_solution(prob::ShopShortTermProblem, lp::_ShortTermLPModel,
                         term::String, obj)
    nr = length(lp.R)
    ngen = length(lp.GEN)
    np_ = length(lp.PUMP)
    nk = length(lp.TUN)
    nt = length(lp.T)
    zR = fill(NaN, nr, nt)
    zG = fill(NaN, ngen, nt)
    zP = fill(NaN, np_, nt)
    zK = fill(NaN, nk, nt)

    storageS = _tidy2(lp.R, lp.T, zR, :S, :reservoir)
    storageH = _tidy2(lp.R, lp.T, zR, :H, :reservoir)
    storage = _join_on_time_name(storageS, storageH, :reservoir, :S, :H)

    empty_junctions = (time = DateTime[], junction = String[],
                       Q_junc = Float64[], R_min_slack = Float64[])
    return HydroSolution{Float64}(
        prob, term, obj,
        storage,
        _tidy2(lp.R, lp.T, zR, :Q_spill, :reservoir),
        _tidy2(lp.GEN, lp.T, zG, :Q, :generator),
        _tidy2(lp.GEN, lp.T, zG, :P, :generator),
        _tidy2(lp.PUMP, lp.T, zP, :Q, :pump),
        _tidy2(lp.PUMP, lp.T, zP, :E, :pump),
        _tidy2(lp.TUN, lp.T, zK, :Q, :tunnel),
        _empty_revenue(lp.T, lp.price),
        _empty_r_alloc(),
        _empty_r_slack(),
        !isempty(lp.cut_groups) ?
            _empty_water_value_per_group() : _empty_water_value_per_res(),
        empty_junctions,
    )
end

# ============================================================
# Solver-side helper functions exposed for the HiGHS / MadNLP exts
# ============================================================

"""
    default_lp_solver()

Returns a `JuMPSolver` configured with a sensible default optimizer.
Declared here with no methods so that solver-providing package
extensions (`HydroModelsOptHiGHSExt`, `HydroModelsOptMadNLPExt`, …) can
add the implementation without method-overwrite warnings on Julia 1.12+.
Calling this without a backing extension loaded throws `MethodError` —
that's informative; load `HiGHS` (or similar) first.
"""
function default_lp_solver end

"""
    is_milp(prob::ShopShortTermProblem) -> Bool

Returns `true` if the LP built from `prob` will contain integer
variables (becoming a MILP rather than a pure LP). Currently the only
binary path is the **reversible pump-turbine** mode variable: a plant
that owns both a generator and a pump gets a `Bin` `mode[plant, t]`
variable that forbids simultaneous generation and pumping.

Future extensions (startup costs, min on/off times) will also flip
this to `true`. Use it to pick a MIP-capable optimizer or to choose
solver attributes that don't apply to pure LPs.

```julia
prob = ShopShortTermProblem(parsed)
solver = is_milp(prob) ? JuMPSolver(HiGHS.Optimizer) :
                          default_lp_solver()
sol = solve(prob, solver)
```
"""
function is_milp(prob::ShopShortTermProblem)
    plants_with_gens = Set(g.plant for g in prob.generators)
    plants_with_pumps = Set(p.plant for p in prob.pumps)
    return !isempty(intersect(plants_with_gens, plants_with_pumps))
end
