"""
JuMP LP/MILP baseline for `ShopShortTermProblem`. Ported from
`HydroModels_depr/src/build_model.jl`.

The formulation is a convex-relaxation LP for plants with only
generators (or only pumps), and a MILP when both are present at the
same plant (reversible pump-turbines need a binary mode variable).

Notable conventions:
- Flows in m³/s, storage in Mm³, power in MW, prices in EUR/MWh.
- Reservoir head is interpolated from `vol_head` breakpoints via a
  convex-combination representation.
- Generator hydraulic curves use a per-curve convex combination plus
  an outer convex combination over reference heads with head-slack
  variables to handle out-of-range operating points.
- Time-resolution-aware (`dt_hours[j]`) conversion: a mass-balance
  multiplier `dt_hours[j] * 3600 / 1e6` converts m³/s to Mm³ per
  timestep.

This file deliberately depends only on `JuMP`, `MathOptInterface`,
`OrderedCollections`, `Dates`, and the SHOP-style types from
`HydroModelsCore`. The HiGHS / MadNLP / etc. backends are wired up
elsewhere via package extensions.
"""

# ============================================================
# Internal flat LP-data layout
# ============================================================

"""
    _ShortTermLPModel

Internal flat-array representation of a parsed `ShopShortTermProblem`
ready for indexed LP construction. Built by `_build_short_term_lp` and
threaded through `solve` so the solution-assembly routines can index
results back to reservoir / generator / pump / tunnel names.

Not part of the public API.
"""
struct _ShortTermLPModel
    # Sets
    R::Vector{String}
    T::Vector{DateTime}
    GEN::Vector{String}
    PUMP::Vector{String}
    TUN::Vector{String}

    # Lookup maps
    ridx::Dict{String, Int}
    gidx::Dict{String, Int}
    pidx::Dict{String, Int}
    tidx::Dict{String, Int}
    tmap::Dict{DateTime, Int}

    # Reservoir data
    Vbreaks::Vector{Vector{Float64}}
    Hbreaks::Vector{Vector{Float64}}
    Smin::Vector{Float64}
    Smax::Vector{Float64}
    S0::Vector{Float64}
    Qin::Matrix{Float64}
    dt_hours::Vector{Float64}

    # Spill routing — `spill_to_idx[i] = 0` means terminal (sink);
    # else it is the destination reservoir index. `spill_in[d]` is the
    # reverse adjacency: source reservoir indices whose spill flows into
    # reservoir `d`. See the B1.5 plan ("Spill routing") for the
    # inference rule applied by the parser.
    spill_to_idx::Vector{Int}
    spill_in::Vector{Vector{Int}}

    # B1.7 — typed Junction (KP) flow constraints. Each junction has a
    # hard `max_flow` upper bound and a soft `min_flow` lower bound
    # (slack with high penalty). `junc_spill_in[k]` lists reservoir
    # indices whose `spill_to_reservoir` resolves to junction `k`.
    JUNC::Vector{String}
    jidx::Dict{String, Int}
    junc_max_flow::Matrix{Float64}
    junc_min_flow::Matrix{Float64}
    junc_spill_in::Vector{Vector{Int}}
    junc_min_penalty::Float64

    # Generator data
    # B1.6: `g_from` is now ragged — one inner vector per generator, each
    # containing the reservoir indices the generator draws water from.
    # `g_head_ref` is the index whose head defines `Hg` (defaults to
    # `g_from[u][1]`). `g_source_qmax` is the per-source tunnel-chain
    # capacity, parallel to `g_from`.
    g_from::Vector{Vector{Int}}
    g_head_ref::Vector{Int}
    g_source_qmax::Vector{Vector{Float64}}
    g_to::Vector{Int}
    g_outlet_line::Vector{Float64}
    g_href::Vector{Vector{Float64}}
    g_Qcurves::Vector{Vector{Float64}}
    g_Pcurves::Vector{Vector{Float64}}
    g_curve_offsets::Vector{Vector{Int}}
    g_curve_sizes::Vector{Vector{Int}}
    g_qmin::Vector{Float64}
    g_qmax::Vector{Float64}
    g_pmin::Vector{Float64}
    g_pmax::Vector{Float64}
    g_eff_x::Vector{Vector{Float64}}
    g_eff_y::Vector{Vector{Float64}}
    g_time_pcap::Matrix{Float64}
    g_time_maint::Matrix{Int}

    # Pump data — B1.6 mirrors the generator multi-source layout.
    p_from::Vector{Vector{Int}}
    p_head_ref::Vector{Int}
    p_source_qmax::Vector{Vector{Float64}}
    p_to::Vector{Int}
    p_outlet_line::Vector{Float64}
    p_href::Vector{Vector{Float64}}
    p_Qcurves::Vector{Vector{Float64}}
    p_Ecurves::Vector{Vector{Float64}}
    p_curve_offsets::Vector{Vector{Int}}
    p_curve_sizes::Vector{Vector{Int}}
    p_qmin::Vector{Float64}
    p_qmax::Vector{Float64}
    p_pmin::Vector{Float64}
    p_pmax::Vector{Float64}
    p_time_pcap::Matrix{Float64}
    p_time_maint::Matrix{Int}

    # Tunnel data
    t_from::Vector{Int}
    t_to::Vector{Int}
    t_qmax::Vector{Float64}

    # Market prices
    price::Vector{Float64}

    # Reserve products
    PROD::Vector{String}
    prodidx::Dict{String, Int}
    prod_dir::Vector{Int}
    g_res_pmin::Matrix{Float64}
    g_res_pmax::Matrix{Float64}
    g_res_sched::Array{Float64, 3}

    # Reserve groups
    RG::Vector{String}
    rg_prod::Vector{Int}
    rg_oblig::Matrix{Float64}
    rg_penalty::Matrix{Float64}

    # End-of-horizon water value (per-reservoir scalar form)
    wv_ncuts::Vector{Int}
    wv_ref::Vector{Vector{Float64}}
    wv_slope::Vector{Vector{Float64}}

    # Group-level Benders cuts (resolved indices)
    cut_groups::Vector{CutGroup{Float64}}

    # Reserve market sale prices [nprod × nt]
    prod_price::Matrix{Float64}
end

# ============================================================
# Helpers
# ============================================================

"""Build a union time grid from all active time series."""
function _build_timegrid(prob::ShopShortTermProblem)
    ts = Set{DateTime}()
    for r in values(prob.reservoirs)
        union!(ts, keys(r.inflow.by_time))
    end
    union!(ts, keys(prob.market.by_time))
    for g in prob.generators
        union!(ts, keys(g.max_p_constr))
        union!(ts, keys(g.maintenance_flag))
    end
    for p in prob.pumps
        union!(ts, keys(p.maintenance_flag))
    end
    for pl in values(prob.plants)
        union!(ts, keys(pl.max_p_constr))
        union!(ts, keys(pl.maintenance_flag))
    end
    return sort!(collect(ts))
end

"""Map `(from, to)` reservoir indices to per-reservoir incoming/outgoing arc lists."""
function _in_out_maps(
        from_vec::Vector{Int}, to_vec::Vector{Int}, nres::Int;
        require_both::Bool = false,
    )
    incoming = [Int[] for _ in 1:nres]
    outgoing = [Int[] for _ in 1:nres]
    for (idx, (f, t)) in enumerate(zip(from_vec, to_vec))
        if require_both
            (f in 1:nres && t in 1:nres) || continue
            push!(outgoing[f], idx)
            push!(incoming[t], idx)
        else
            f in 1:nres && push!(outgoing[f], idx)
            t in 1:nres && push!(incoming[t], idx)
        end
    end
    return incoming, outgoing
end

"""
    _in_out_maps_ragged(from_ragged, to_vec, nres) -> (incoming, outgoing_pairs)

Variant of `_in_out_maps` for the B1.6 multi-source case where each
unit (generator / pump) has a *vector* of source reservoir indices.

- `incoming[i] :: Vector{Int}` — unit indices whose `to_vec[u] == i`
  (single destination — unchanged from the scalar case).
- `outgoing_pairs[i] :: Vector{Tuple{Int, Int}}` — `(u, s)` pairs
  where `from_ragged[u][s] == i`. The slot index `s` lets the
  mass-balance loop subtract the right `Qg_from[u, s, j]`.
"""
function _in_out_maps_ragged(
        from_ragged::Vector{Vector{Int}}, to_vec::Vector{Int}, nres::Int,
    )
    incoming = [Int[] for _ in 1:nres]
    outgoing_pairs = [Tuple{Int, Int}[] for _ in 1:nres]
    for u in eachindex(to_vec)
        t = to_vec[u]
        t in 1:nres && push!(incoming[t], u)
        for (s, f) in enumerate(from_ragged[u])
            f in 1:nres && push!(outgoing_pairs[f], (u, s))
        end
    end
    return incoming, outgoing_pairs
end

"""Flatten a curve list `[(href_c, PiecewiseLinear)]` into packed arrays."""
function _pack_curves(curves::AbstractVector)
    href = Float64[]
    qflat = Float64[]
    yflat = Float64[]
    offsets = Int[]
    sizes = Int[]
    pos = 1
    for (h, pwl) in curves
        push!(href, h)
        push!(offsets, pos)
        n = length(pwl.x)
        append!(qflat, pwl.x)
        append!(yflat, pwl.y)
        push!(sizes, n)
        pos += n
    end
    return href, qflat, yflat, offsets, sizes
end

# ============================================================
# LP construction (port of build_model from depr)
# ============================================================

"""
    _is_highs_optimizer(optimizer) -> Bool

Internal predicate. Returns `true` iff the parent module of the
optimizer constructor is `HiGHS`. Used to gate vendor-specific
attribute defaults (`mip_rel_gap`, `primal_feasibility_tolerance`,
`mip_feasibility_tolerance`) inside `_build_short_term_lp` so they do
not silently leak into other LP backends (e.g. `CoolPDLP.Optimizer`)
that would then fail at `optimize!`.

Handles both bare optimizer constructors (`HiGHS.Optimizer`) and
`MOI.OptimizerWithAttributes` wrappers.
"""
function _is_highs_optimizer(optimizer)
    optimizer === nothing && return false
    base = optimizer isa MOI.OptimizerWithAttributes ?
           optimizer.optimizer_constructor : optimizer
    try
        return nameof(parentmodule(base)) === :HiGHS
    catch
        return false
    end
end

"""
    _build_short_term_lp(prob, optimizer)

Construct the JuMP model for a `ShopShortTermProblem`. Returns
`(lp_data::_ShortTermLPModel, model::JuMP.Model)`. Internal — users go
through `solve(::ShopShortTermProblem, ::JuMPSolver)`.
"""
function _build_short_term_lp(prob::ShopShortTermProblem, optimizer)
    Tg = _build_timegrid(prob)
    tmap = Dict(t => i for (i, t) in pairs(Tg))
    nt = length(Tg)

    # Per-timestep duration in hours from the variable-resolution schedule.
    dt_schedule = isempty(prob.dt_schedule) ?
        [(DateTime(0), 60.0)] : prob.dt_schedule
    dt_hours = Vector{Float64}(undef, nt)
    sched_times = [s[1] for s in dt_schedule]
    sched_mins = [s[2] for s in dt_schedule]
    for (j, t) in enumerate(Tg)
        idx = max(searchsortedlast(sched_times, t), 1)
        dt_hours[j] = sched_mins[idx] / 60.0
    end

    # Reservoir flat layout
    R = sort!(collect(keys(prob.reservoirs)))
    ridx = Dict(r => i for (i, r) in pairs(R))
    nr = length(R)
    Vbreaks = [prob.reservoirs[r].vol_breaks for r in R]
    Hbreaks = [prob.reservoirs[r].head_breaks for r in R]
    Smin = [prob.reservoirs[r].s_min for r in R]
    Smax = [prob.reservoirs[r].s_max for r in R]
    S0 = [prob.reservoirs[r].s0 for r in R]

    # Inflow with LOCF (last observation carried forward).
    Qin = zeros(nr, nt)
    for (rname, rr) in prob.reservoirs
        i = ridx[rname]
        isempty(rr.inflow.by_time) && continue
        inf_times = sort!(collect(keys(rr.inflow.by_time)))
        inf_vals = [rr.inflow.by_time[t] for t in inf_times]
        for (j, t) in enumerate(Tg)
            idx = searchsortedlast(inf_times, t)
            idx == 0 && continue
            Qin[i, j] = inf_vals[idx]
        end
    end

    # Market price with LOCF
    price = zeros(nt)
    if !isempty(prob.market.by_time)
        mkt_times = sort!(collect(keys(prob.market.by_time)))
        mkt_vals = [prob.market.by_time[t] for t in mkt_times]
        for (j, t) in enumerate(Tg)
            idx = searchsortedlast(mkt_times, t)
            idx == 0 && continue
            price[j] = mkt_vals[idx]
        end
    end
    all(iszero, price) && @warn "All market prices are zero — revenue objective will be flat. Check market_name selection."

    # Per-reservoir water value
    wv_ncuts = zeros(Int, nr)
    wv_ref = [Float64[] for _ in 1:nr]
    wv_slope = [Float64[] for _ in 1:nr]
    for (rname, rr) in prob.reservoirs
        i = get(ridx, rname, 0)
        i == 0 && continue
        if rr.water_value !== nothing
            wv_ncuts[i] = length(rr.water_value.ref)
            wv_ref[i] = copy(rr.water_value.ref)
            wv_slope[i] = copy(rr.water_value.slope)
        end
    end

    # B1.7 — typed Junction (KP) book-keeping. Builds the index space,
    # densifies the per-junction `max_flow` / `flow` schedules, and
    # the reverse-adjacency `junc_spill_in[k]` (reservoirs whose
    # `spill_to_reservoir` resolves to junction `k`).
    JUNC = sort!(collect(keys(prob.junctions)))
    jidx = Dict(n => i for (i, n) in pairs(JUNC))
    njunc = length(JUNC)
    junc_max_flow = fill(Inf, njunc, nt)
    junc_min_flow = zeros(njunc, nt)
    for (k, jname) in pairs(JUNC)
        j = prob.junctions[jname]
        for (t, v) in j.max_flow
            tt = get(tmap, t, nothing)
            tt === nothing && continue
            junc_max_flow[k, tt] = min(junc_max_flow[k, tt], Float64(v))
        end
        for (t, v) in j.flow
            tt = get(tmap, t, nothing)
            tt === nothing && continue
            junc_min_flow[k, tt] = max(junc_min_flow[k, tt], Float64(v))
        end
    end
    junc_spill_in = [Int[] for _ in 1:njunc]

    # Spill routing — resolve each reservoir's `spill_to_reservoir`
    # name to an index in `R` (reservoir destination) OR in `JUNC`
    # (junction destination). `spill_to_idx[i] = 0` means terminal
    # (the existing sink behaviour). For reservoir destinations,
    # `Qspill[i, j]` appears as an inflow term in reservoir
    # `spill_to_idx[i]`'s mass balance via `spill_in[d]`. For junction
    # destinations, `Qspill[i, j]` is collected into `junc_spill_in[k]`
    # and the LP enforces `Qjunc[k, j] == sum_i Qspill[i, j]` plus
    # the max/min flow constraints.
    spill_to_idx = zeros(Int, nr)
    for (rname, rr) in prob.reservoirs
        i = get(ridx, rname, 0)
        i == 0 && continue
        dest = rr.spill_to_reservoir
        dest === nothing && continue
        # First try reservoir destination.
        r_idx = get(ridx, dest, 0)
        if r_idx > 0
            spill_to_idx[i] = r_idx
            continue
        end
        # Then try junction destination.
        k_idx = get(jidx, dest, 0)
        if k_idx > 0
            push!(junc_spill_in[k_idx], i)
            continue
        end
        @warn "Reservoir \"$rname\" has spill_to_reservoir=\"$dest\" but " *
              "no such reservoir or junction in problem; treating as terminal."
    end
    spill_in = [Int[] for _ in 1:nr]
    for i in 1:nr
        d = spill_to_idx[i]
        d > 0 && push!(spill_in[d], i)
    end
    wv_res_indices = findall(>(0), wv_ncuts)
    has_water_value = !isempty(wv_res_indices)

    # Cut-group reservoir indices
    resolved_cut_groups = CutGroup{Float64}[]
    for cg in prob.cut_groups
        ri = [get(ridx, rn, 0) for rn in cg.res_names]
        push!(resolved_cut_groups, CutGroup{Float64}(
            name        = cg.name,
            res_names   = cg.res_names,
            res_indices = ri,
            ncuts       = cg.ncuts,
            intercept   = cg.intercept,
            slopes      = cg.slopes,
        ))
    end
    has_cut_groups = !isempty(resolved_cut_groups)

    # Generator flat layout
    GEN = [g.name for g in prob.generators]
    gidx = Dict(g => i for (i, g) in pairs(GEN))
    ngen = length(GEN)
    g_from = Vector{Int}[]            # B1.6: ragged — per-generator source list
    g_head_ref = Int[]                 # B1.6: head-reference reservoir per gen
    g_source_qmax = Vector{Float64}[]  # B1.6: per-source tunnel capacity
    g_to = Int[]
    g_outlet_line = Float64[]
    g_href = Vector{Vector{Float64}}()
    g_Qcurves = Vector{Vector{Float64}}()
    g_Pcurves = Vector{Vector{Float64}}()
    g_offsets = Vector{Vector{Int}}()
    g_sizes = Vector{Vector{Int}}()
    g_qmin = Float64[]
    g_qmax = Float64[]
    g_pmin = Float64[]
    g_pmax = Float64[]
    g_eff_x = Vector{Vector{Float64}}()
    g_eff_y = Vector{Vector{Float64}}()
    g_time_pcap = fill(Inf, ngen, nt)
    g_time_maint = zeros(Int, ngen, nt)

    for (u, g) in enumerate(prob.generators)
        # B1.6: resolve every source-reservoir name to its index and
        # build the per-source caps. Single-source generators end up
        # with one-element vectors; multi-source plants (e.g. Hemsil2)
        # produce 2–5 entries each.
        src_idx = Int[]
        src_cap = Float64[]
        for src_name in g.from_res
            i = get(ridx, src_name, 0)
            i == 0 && continue
            push!(src_idx, i)
            cap = get(g.source_qmax, src_name, g.qmax)
            push!(src_cap, isfinite(cap) ? cap : g.qmax)
        end
        push!(g_from, src_idx)
        push!(g_source_qmax, src_cap)
        head_name = g.head_reference !== nothing ? g.head_reference :
                    (isempty(g.from_res) ? "" : first(g.from_res))
        push!(g_head_ref, get(ridx, head_name, 0))
        push!(g_to, get(ridx, g.to_res, 0))
        outlet = haskey(prob.plants, g.plant) ?
            prob.plants[g.plant].outlet_line : 0.0
        push!(g_outlet_line, outlet)
        href, qflat, pflat, offs, sizes = _pack_curves(g.curves)
        push!(g_href, href)
        push!(g_Qcurves, qflat)
        push!(g_Pcurves, pflat)
        push!(g_offsets, offs)
        push!(g_sizes, sizes)
        push!(g_qmin, g.qmin)
        push!(g_qmax, g.qmax)
        push!(g_pmin, g.pmin)
        push!(g_pmax, g.pmax)

        if g.gen_eff === nothing
            push!(g_eff_x, Float64[])
            push!(g_eff_y, Float64[])
        else
            push!(g_eff_x, g.gen_eff.x)
            push!(g_eff_y, g.gen_eff.y)
        end

        for j in 1:nt
            g_time_pcap[u, j] = g.pmax
        end
        for (t, cap) in g.max_p_constr
            j = get(tmap, t, nothing)
            j === nothing && continue
            g_time_pcap[u, j] = min(g_time_pcap[u, j], cap)
        end
        for (t, flag) in g.maintenance_flag
            j = get(tmap, t, nothing)
            j === nothing && continue
            g_time_maint[u, j] = flag
            if flag == 1
                g_time_pcap[u, j] = 0.0
            end
        end
    end

    # Pump flat layout
    PUMP = [p.name for p in prob.pumps]
    pidx = Dict(p => i for (i, p) in pairs(PUMP))
    npump = length(PUMP)
    p_from = Vector{Int}[]           # B1.6
    p_head_ref = Int[]
    p_source_qmax = Vector{Float64}[]
    p_to = Int[]
    p_outlet_line = Float64[]
    p_href = Vector{Vector{Float64}}()
    p_Qcurves = Vector{Vector{Float64}}()
    p_Ecurves = Vector{Vector{Float64}}()
    p_offsets = Vector{Vector{Int}}()
    p_sizes = Vector{Vector{Int}}()
    p_qmin = Float64[]
    p_qmax = Float64[]
    p_pmin = Float64[]
    p_pmax = Float64[]
    p_time_pcap = fill(Inf, npump, nt)
    p_time_maint = zeros(Int, npump, nt)

    for (v, p) in enumerate(prob.pumps)
        # B1.6: pump intake can be multi-source (rare in practice).
        src_idx = Int[]
        src_cap = Float64[]
        for src_name in p.from_res
            i = get(ridx, src_name, 0)
            i == 0 && continue
            push!(src_idx, i)
            cap = get(p.source_qmax, src_name, p.qmax)
            push!(src_cap, isfinite(cap) ? cap : p.qmax)
        end
        push!(p_from, src_idx)
        push!(p_source_qmax, src_cap)
        head_name = p.head_reference !== nothing ? p.head_reference :
                    (isempty(p.from_res) ? "" : first(p.from_res))
        push!(p_head_ref, get(ridx, head_name, 0))
        push!(p_to, get(ridx, p.to_res, 0))
        outlet = haskey(prob.plants, p.plant) ?
            prob.plants[p.plant].outlet_line : 0.0
        push!(p_outlet_line, outlet)
        href, qflat, eflat, offs, sizes = _pack_curves(p.curves)
        push!(p_href, href)
        push!(p_Qcurves, qflat)
        push!(p_Ecurves, eflat)
        push!(p_offsets, offs)
        push!(p_sizes, sizes)
        push!(p_qmin, p.qmin)
        push!(p_qmax, p.qmax)
        push!(p_pmin, p.pmin)
        push!(p_pmax, p.pmax)

        for j in 1:nt
            p_time_pcap[v, j] = p.pmax
        end
        for (t, flag) in p.maintenance_flag
            j = get(tmap, t, nothing)
            j === nothing && continue
            p_time_maint[v, j] = flag
            if flag == 1
                p_time_pcap[v, j] = 0.0
            end
        end
    end

    # Tunnel flat layout
    TUN = [t.name for t in prob.tunnels]
    tidx = Dict(t => i for (i, t) in pairs(TUN))
    nk = length(TUN)
    t_from = [get(ridx, t.from_node, 0) for t in prob.tunnels]
    t_to = [get(ridx, t.to_node, 0) for t in prob.tunnels]
    t_qmax = [t.qmax for t in prob.tunnels]

    # Reserve products (union over generators + groups)
    prod_set = String[]
    for g in prob.generators, rs in g.reserves
        rs.product in prod_set || push!(prod_set, rs.product)
    end
    for rg in prob.reserve_groups
        rg.product in prod_set || push!(prod_set, rg.product)
    end
    sort!(prod_set)
    PROD = prod_set
    prodidx = Dict(p => i for (i, p) in pairs(PROD))
    nprod = length(PROD)
    _up_products = Set(["FCR_N_UP", "FCR_D_UP", "FRR_UP", "RR_UP"])
    prod_dir = [p in _up_products ? 1 : -1 for p in PROD]

    g_res_pmin = zeros(ngen, nprod)
    g_res_pmax = zeros(ngen, nprod)
    g_res_sched = fill(Inf, ngen, nprod, nt)
    for (u, g) in enumerate(prob.generators), rs in g.reserves
        p = get(prodidx, rs.product, 0)
        p == 0 && continue
        g_res_pmin[u, p] = rs.pmin
        g_res_pmax[u, p] = rs.pmax
        isempty(rs.schedule) && continue
        obs_times = sort!(collect(keys(rs.schedule)))
        obs_vals = [rs.schedule[t] for t in obs_times]
        for (j, t) in enumerate(Tg)
            idx = searchsortedlast(obs_times, t)
            idx == 0 && continue
            g_res_sched[u, p, j] = obs_vals[idx]
        end
    end

    RG = [rg.name * "/" * rg.product for rg in prob.reserve_groups]
    nrg = length(RG)
    rg_prod = [get(prodidx, rg.product, 0) for rg in prob.reserve_groups]
    rg_oblig = zeros(nrg, nt)
    rg_penalty = zeros(nrg, nt)
    for (g_idx, rg) in enumerate(prob.reserve_groups)
        for (t, v) in rg.obligation
            j = get(tmap, t, nothing)
            j === nothing && continue
            rg_oblig[g_idx, j] = v
        end
        for (t, v) in rg.penalty_cost
            j = get(tmap, t, nothing)
            j === nothing && continue
            rg_penalty[g_idx, j] = v
        end
    end

    prod_price = zeros(nprod, nt)
    for (p_name, price_ts) in prob.reserve_prices
        p = get(prodidx, p_name, 0)
        p == 0 && continue
        isempty(price_ts) && continue
        obs_times = sort!(collect(keys(price_ts)))
        obs_vals = [price_ts[t] for t in obs_times]
        for (j, t) in enumerate(Tg)
            idx = searchsortedlast(obs_times, t)
            idx == 0 && continue
            prod_price[p, j] = obs_vals[idx]
        end
    end

    lp = _ShortTermLPModel(
        R, Tg, GEN, PUMP, TUN,
        ridx, gidx, pidx, tidx, tmap,
        Vbreaks, Hbreaks, Smin, Smax, S0, Qin, dt_hours,
        spill_to_idx, spill_in,
        JUNC, jidx, junc_max_flow, junc_min_flow, junc_spill_in,
        1.0e6,                                  # junc_min_penalty (EUR per m³/s·h)
        g_from, g_head_ref, g_source_qmax, g_to,
        g_outlet_line, g_href, g_Qcurves, g_Pcurves, g_offsets, g_sizes,
        g_qmin, g_qmax, g_pmin, g_pmax, g_eff_x, g_eff_y, g_time_pcap, g_time_maint,
        p_from, p_head_ref, p_source_qmax, p_to,
        p_outlet_line, p_href, p_Qcurves, p_Ecurves, p_offsets, p_sizes,
        p_qmin, p_qmax, p_pmin, p_pmax, p_time_pcap, p_time_maint,
        t_from, t_to, t_qmax,
        price,
        PROD, prodidx, prod_dir,
        g_res_pmin, g_res_pmax, g_res_sched,
        RG, rg_prod, rg_oblig, rg_penalty,
        wv_ncuts, wv_ref, wv_slope,
        resolved_cut_groups,
        prod_price,
    )

    model = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)

    # MIP gap and feasibility tolerance — see depr/build_model.jl notes.
    # Water-value slopes (EUR/Mm³) can hit ~5e8 while flow vars are O(1e-3),
    # spanning 11 orders of magnitude. The default 1e-7 tolerance is too
    # tight; bump to 1e-4 so HiGHS doesn't reject otherwise-optimal solutions.
    # These attribute names are HiGHS-specific; CoolPDLP and other vendors
    # accept the attribute store silently and then fail at `optimize!` when
    # the constructor sees unknown kwargs. Gate on optimizer module name so
    # the defaults apply only when HiGHS is actually in use.
    if optimizer !== nothing && _is_highs_optimizer(optimizer)
        try
            JuMP.set_optimizer_attribute(model, "mip_rel_gap", 0.001)
            JuMP.set_optimizer_attribute(model, "primal_feasibility_tolerance", 1e-4)
            JuMP.set_optimizer_attribute(model, "mip_feasibility_tolerance", 1e-4)
        catch
            # Defensive — non-fatal if the optimizer rejects them.
        end
    end

    # ----------------------------------------------------------------------
    # Reservoir state, head, spill
    # ----------------------------------------------------------------------
    maxK = maximum(length.(lp.Vbreaks); init = 0)
    @variable(model, S[1:nr, 1:nt])
    @variable(model, H[1:nr, 1:nt])
    @variable(model, λ_res[1:nr, 1:nt, 1:maxK] >= 0)
    @variable(model, Qspill[1:nr, 1:nt] >= 0)

    for i in 1:nr, j in 1:nt
        Ki = length(lp.Vbreaks[i])
        if Ki == 0
            @constraint(model, S[i, j] == lp.S0[i])
            @constraint(model, H[i, j] == 0.0)
        else
            @constraint(model, sum(λ_res[i, j, k] for k in 1:Ki) == 1)
            @constraint(model, S[i, j] == sum(lp.Vbreaks[i][k] * λ_res[i, j, k] for k in 1:Ki))
            @constraint(model, H[i, j] == sum(lp.Hbreaks[i][k] * λ_res[i, j, k] for k in 1:Ki))
        end
        isfinite(lp.Smin[i]) && @constraint(model, S[i, j] >= lp.Smin[i])
        isfinite(lp.Smax[i]) && @constraint(model, S[i, j] <= lp.Smax[i])
    end
    for i in 1:nr
        @constraint(model, S[i, 1] == lp.S0[i])
    end

    # ----------------------------------------------------------------------
    # Generator hydraulic PQ + electrical efficiency
    # ----------------------------------------------------------------------
    @variable(model, Qg[1:ngen, 1:nt] >= 0)
    @variable(model, Pmech[1:ngen, 1:nt] >= 0)
    @variable(model, Pg[1:ngen, 1:nt] >= 0)
    @variable(model, Hg[1:ngen, 1:nt])

    # B1.6 — per-(generator, source) flow split. For each generator `u`
    # with `length(lp.g_from[u]) = n_src`, we introduce `n_src · nt`
    # auxiliary variables that sum to `Qg[u, j]`. Single-source generators
    # collapse to `Qg_from[u, 1, j] == Qg[u, j]` and the LP presolver
    # removes the trivial alias.
    Qg_from = @variable(model,
        Qg_from[u = 1:ngen, s = 1:length(lp.g_from[u]), j = 1:nt] >= 0)
    for u in 1:ngen, j in 1:nt
        n_src = length(lp.g_from[u])
        n_src == 0 && continue
        @constraint(model, Qg[u, j] == sum(Qg_from[u, s, j] for s in 1:n_src))
        for s in 1:n_src
            cap = lp.g_source_qmax[u][s]
            isfinite(cap) && @constraint(model, Qg_from[u, s, j] <= cap)
        end
    end

    for u in 1:ngen
        # B1.6 — head difference uses the head-reference reservoir
        # (defaults to first source). For multi-source plants the
        # physical intake head is at the merging KP; the LP picks the
        # dominant source as a stable approximation.
        from_idx = lp.g_head_ref[u]
        to_idx = lp.g_to[u]
        for j in 1:nt
            if from_idx in 1:nr && to_idx in 1:nr
                @constraint(model, Hg[u, j] == H[from_idx, j] - H[to_idx, j])
            elseif from_idx in 1:nr
                @constraint(model, Hg[u, j] == H[from_idx, j] - lp.g_outlet_line[u])
            else
                @constraint(model, Hg[u, j] == 0.0)
            end
        end

        Cu = length(lp.g_href[u])
        has_head = from_idx in 1:nr
        if Cu == 0 || (Cu > 1 && !has_head)
            for j in 1:nt
                @constraint(model, Pmech[u, j] == 0.0)
                @constraint(model, Pg[u, j] == 0.0)
                @constraint(model, Qg[u, j] == 0.0)
            end
        else
            sizes = lp.g_curve_sizes[u]
            offs = lp.g_curve_offsets[u]
            if Cu == 1
                lambda_u = @variable(model, [k = 1:sizes[1], j = 1:nt],
                    lower_bound = 0, base_name = "lambda_g_$(u)_c1")
                for j in 1:nt
                    @constraint(model, sum(lambda_u[k, j] for k in 1:sizes[1]) == 1)
                    @constraint(model, Qg[u, j] == sum(lp.g_Qcurves[u][offs[1] + k - 1] * lambda_u[k, j] for k in 1:sizes[1]))
                    @constraint(model, Pmech[u, j] == sum(lp.g_Pcurves[u][offs[1] + k - 1] * lambda_u[k, j] for k in 1:sizes[1]))
                end
            else
                alpha_u = @variable(model, [c = 1:Cu, j = 1:nt],
                    lower_bound = 0, base_name = "alpha_g_$(u)")
                lambda_u = [@variable(model, [k = 1:sizes[c], j = 1:nt],
                    lower_bound = 0, base_name = "lambda_g_$(u)_c$(c)") for c in 1:Cu]
                href_min_u = minimum(lp.g_href[u])
                href_max_u = maximum(lp.g_href[u])
                hg_lo_u = @variable(model, [j = 1:nt],
                    lower_bound = 0, base_name = "hg_lo_g_$(u)")
                hg_hi_u = @variable(model, [j = 1:nt],
                    lower_bound = 0, base_name = "hg_hi_g_$(u)")
                for j in 1:nt
                    @constraint(model, sum(alpha_u[c, j] for c in 1:Cu) == 1)
                    @constraint(model, hg_lo_u[j] >= href_min_u - Hg[u, j])
                    @constraint(model, hg_hi_u[j] >= Hg[u, j] - href_max_u)
                    @constraint(model, sum(lp.g_href[u][c] * alpha_u[c, j] for c in 1:Cu) ==
                                       Hg[u, j] + hg_lo_u[j] - hg_hi_u[j])
                    @constraint(model, Qg[u, j] == sum(sum(lp.g_Qcurves[u][offs[c] + k - 1] * lambda_u[c][k, j] for k in 1:sizes[c]) for c in 1:Cu))
                    @constraint(model, Pmech[u, j] == sum(sum(lp.g_Pcurves[u][offs[c] + k - 1] * lambda_u[c][k, j] for k in 1:sizes[c]) for c in 1:Cu))
                    for c in 1:Cu
                        @constraint(model, sum(lambda_u[c][k, j] for k in 1:sizes[c]) == alpha_u[c, j])
                    end
                end
            end

            K_eff = length(lp.g_eff_x[u])
            if K_eff > 0
                sigma_u = @variable(model, [k = 1:K_eff, j = 1:nt],
                    lower_bound = 0, base_name = "sigma_g_$(u)")
                for j in 1:nt
                    @constraint(model, sum(sigma_u[k, j] for k in 1:K_eff) == 1)
                    @constraint(model, Pmech[u, j] == sum(lp.g_eff_x[u][k] * sigma_u[k, j] for k in 1:K_eff))
                    @constraint(model, Pg[u, j] == sum(lp.g_eff_x[u][k] * lp.g_eff_y[u][k] * sigma_u[k, j] for k in 1:K_eff))
                end
            else
                for j in 1:nt
                    @constraint(model, Pg[u, j] == Pmech[u, j])
                end
            end
        end

        for j in 1:nt
            isfinite(lp.g_qmax[u]) && @constraint(model, Qg[u, j] <= lp.g_qmax[u])
            isfinite(lp.g_pmax[u]) && @constraint(model, Pg[u, j] <= lp.g_pmax[u])
            isfinite(lp.g_time_pcap[u, j]) && @constraint(model, Pg[u, j] <= lp.g_time_pcap[u, j])
        end
    end

    # ----------------------------------------------------------------------
    # Pumps
    # ----------------------------------------------------------------------
    @variable(model, Qpump[1:npump, 1:nt] >= 0)
    @variable(model, Ppump[1:npump, 1:nt] >= 0)
    @variable(model, Hpump[1:npump, 1:nt])

    # B1.6 — per-(pump, source) flow split; same structure as Qg_from
    # above. Single-source pumps (the only kind in NO5) collapse to
    # `Qpump_from[v, 1, j] == Qpump[v, j]`.
    Qpump_from = @variable(model,
        Qpump_from[v = 1:npump, s = 1:length(lp.p_from[v]), j = 1:nt] >= 0)
    for v in 1:npump, j in 1:nt
        n_src = length(lp.p_from[v])
        n_src == 0 && continue
        @constraint(model, Qpump[v, j] == sum(Qpump_from[v, s, j] for s in 1:n_src))
        for s in 1:n_src
            cap = lp.p_source_qmax[v][s]
            isfinite(cap) && @constraint(model, Qpump_from[v, s, j] <= cap)
        end
    end

    for v in 1:npump
        from_idx = lp.p_head_ref[v]
        to_idx = lp.p_to[v]
        for j in 1:nt
            if from_idx in 1:nr && to_idx in 1:nr
                @constraint(model, Hpump[v, j] == H[to_idx, j] - H[from_idx, j])
            elseif to_idx in 1:nr
                @constraint(model, Hpump[v, j] == H[to_idx, j] - lp.p_outlet_line[v])
            else
                @constraint(model, Hpump[v, j] == 0.0)
            end
        end

        Cv = length(lp.p_href[v])
        if Cv == 0
            for j in 1:nt
                @constraint(model, Qpump[v, j] == 0.0)
                @constraint(model, Ppump[v, j] == 0.0)
            end
        else
            sizes = lp.p_curve_sizes[v]
            offs = lp.p_curve_offsets[v]
            if Cv == 1
                mu_v = @variable(model, [k = 1:sizes[1], j = 1:nt],
                    lower_bound = 0, base_name = "mu_p_$(v)_c1")
                for j in 1:nt
                    @constraint(model, sum(mu_v[k, j] for k in 1:sizes[1]) == 1)
                    @constraint(model, Qpump[v, j] == sum(lp.p_Qcurves[v][offs[1] + k - 1] * mu_v[k, j] for k in 1:sizes[1]))
                    @constraint(model, Ppump[v, j] == sum(lp.p_Ecurves[v][offs[1] + k - 1] * mu_v[k, j] for k in 1:sizes[1]))
                end
            else
                beta_v = @variable(model, [c = 1:Cv, j = 1:nt],
                    lower_bound = 0, base_name = "beta_p_$(v)")
                mu_v = [@variable(model, [k = 1:sizes[c], j = 1:nt],
                    lower_bound = 0, base_name = "mu_p_$(v)_c$(c)") for c in 1:Cv]
                href_min_v = minimum(lp.p_href[v])
                href_max_v = maximum(lp.p_href[v])
                hp_lo_v = @variable(model, [j = 1:nt],
                    lower_bound = 0, base_name = "hp_lo_p_$(v)")
                hp_hi_v = @variable(model, [j = 1:nt],
                    lower_bound = 0, base_name = "hp_hi_p_$(v)")
                for j in 1:nt
                    @constraint(model, sum(beta_v[c, j] for c in 1:Cv) == 1)
                    @constraint(model, hp_lo_v[j] >= href_min_v - Hpump[v, j])
                    @constraint(model, hp_hi_v[j] >= Hpump[v, j] - href_max_v)
                    @constraint(model, sum(lp.p_href[v][c] * beta_v[c, j] for c in 1:Cv) ==
                                       Hpump[v, j] + hp_lo_v[j] - hp_hi_v[j])
                    @constraint(model, Qpump[v, j] == sum(sum(lp.p_Qcurves[v][offs[c] + k - 1] * mu_v[c][k, j] for k in 1:sizes[c]) for c in 1:Cv))
                    @constraint(model, Ppump[v, j] == sum(sum(lp.p_Ecurves[v][offs[c] + k - 1] * mu_v[c][k, j] for k in 1:sizes[c]) for c in 1:Cv))
                    for c in 1:Cv
                        @constraint(model, sum(mu_v[c][k, j] for k in 1:sizes[c]) == beta_v[c, j])
                    end
                end
            end
        end

        for j in 1:nt
            isfinite(lp.p_qmax[v]) && @constraint(model, Qpump[v, j] <= lp.p_qmax[v])
            isfinite(lp.p_pmax[v]) && @constraint(model, Ppump[v, j] <= lp.p_pmax[v])
            isfinite(lp.p_time_pcap[v, j]) && @constraint(model, Ppump[v, j] <= lp.p_time_pcap[v, j])
        end
    end

    # ----------------------------------------------------------------------
    # Binary mode for reversible plants (pump-turbines)
    # ----------------------------------------------------------------------
    plant_gen_units = Dict{String, Vector{Int}}()
    plant_pump_units = Dict{String, Vector{Int}}()
    for (u, g) in enumerate(prob.generators)
        push!(get!(plant_gen_units, g.plant, Int[]), u)
    end
    for (v, p) in enumerate(prob.pumps)
        push!(get!(plant_pump_units, p.plant, Int[]), v)
    end
    reversible_plants = sort!(collect(intersect(keys(plant_gen_units), keys(plant_pump_units))))
    if !isempty(reversible_plants)
        nrev = length(reversible_plants)
        @variable(model, mode[1:nrev, 1:nt], Bin)
        for (i, pl) in enumerate(reversible_plants)
            for u in plant_gen_units[pl], j in 1:nt
                M = isfinite(lp.g_pmax[u]) ? lp.g_pmax[u] : 0.0
                M > 0 && @constraint(model, Pg[u, j] <= M * mode[i, j])
            end
            for v in plant_pump_units[pl], j in 1:nt
                M = isfinite(lp.p_pmax[v]) ? lp.p_pmax[v] : 0.0
                M > 0 && @constraint(model, Ppump[v, j] <= M * (1 - mode[i, j]))
            end
        end
    end

    # ----------------------------------------------------------------------
    # Tunnels (passive flow with capacity)
    # ----------------------------------------------------------------------
    @variable(model, Qtun[1:nk, 1:nt] >= 0)
    for k in 1:nk, j in 1:nt
        isfinite(lp.t_qmax[k]) && @constraint(model, Qtun[k, j] <= lp.t_qmax[k])
    end

    # ----------------------------------------------------------------------
    # Junctions (B1.7) — typed KP nodes with hard max-flow cap and a
    # soft min-flow lower bound. `Qjunc[k, j]` is the total flow through
    # junction k at stage j, defined as the sum of spills from every
    # reservoir whose `spill_to_reservoir` resolves to this junction.
    # The soft min-flow is enforced via slack `Rj_min[k, j]` plus a
    # large penalty in the objective.
    # ----------------------------------------------------------------------
    njunc = length(lp.JUNC)
    @variable(model, Qjunc[1:njunc, 1:nt] >= 0)
    @variable(model, Rj_min[1:njunc, 1:nt] >= 0)
    for k in 1:njunc, j in 1:nt
        @constraint(model, Qjunc[k, j] ==
            sum(Qspill[i, j] for i in lp.junc_spill_in[k]; init = 0.0))
        cap = lp.junc_max_flow[k, j]
        isfinite(cap) && @constraint(model, Qjunc[k, j] <= cap)
        minv = lp.junc_min_flow[k, j]
        minv > 0 && @constraint(model, Qjunc[k, j] + Rj_min[k, j] >= minv)
    end

    # ----------------------------------------------------------------------
    # Mass balance — m³/s × dt_hours × 3600 / 1e6 → Mm³
    # ----------------------------------------------------------------------
    conv = [lp.dt_hours[j] * (3600.0 / 1.0e6) for j in 1:nt]
    # B1.6 — generator / pump source side is now ragged. `*_out_pairs[i]`
    # is a list of `(unit, source_slot)` tuples for the mass balance to
    # subtract `Qg_from[u, s, j]` / `Qpump_from[v, s, j]` per source.
    g_in, g_out_pairs = _in_out_maps_ragged(lp.g_from, lp.g_to, nr)
    p_in, p_out_pairs = _in_out_maps_ragged(lp.p_from, lp.p_to, nr)
    t_in, t_out = _in_out_maps(lp.t_from, lp.t_to, nr; require_both = true)

    for j in 1:(nt - 1), i in 1:nr
        @constraint(model,
            S[i, j + 1] == S[i, j]
                + lp.Qin[i, j] * conv[j]
                + sum(Qg[u, j] * conv[j] for u in g_in[i])
                + sum(Qpump[v, j] * conv[j] for v in p_in[i])
                + sum(Qtun[k, j] * conv[j] for k in t_in[i])
                + sum(Qspill[u, j] * conv[j] for u in lp.spill_in[i]; init = 0.0)
                - sum(Qg_from[u, s, j] * conv[j] for (u, s) in g_out_pairs[i]; init = 0.0)
                - sum(Qpump_from[v, s, j] * conv[j] for (v, s) in p_out_pairs[i]; init = 0.0)
                - sum(Qtun[k, j] * conv[j] for k in t_out[i])
                - Qspill[i, j] * conv[j]
        )
    end

    # Post-horizon storage for the end-value evaluation.
    @variable(model, S_end[1:nr])
    for i in 1:nr
        @constraint(model,
            S_end[i] == S[i, nt]
                + lp.Qin[i, nt] * conv[nt]
                + sum(Qg[u, nt] * conv[nt] for u in g_in[i])
                + sum(Qpump[v, nt] * conv[nt] for v in p_in[i])
                + sum(Qtun[k, nt] * conv[nt] for k in t_in[i])
                + sum(Qspill[u, nt] * conv[nt] for u in lp.spill_in[i]; init = 0.0)
                - sum(Qg_from[u, s, nt] * conv[nt] for (u, s) in g_out_pairs[i]; init = 0.0)
                - sum(Qpump_from[v, s, nt] * conv[nt] for (v, s) in p_out_pairs[i]; init = 0.0)
                - sum(Qtun[k, nt] * conv[nt] for k in t_out[i])
                - Qspill[i, nt] * conv[nt]
        )
        isfinite(lp.Smin[i]) && @constraint(model, S_end[i] >= lp.Smin[i])
        isfinite(lp.Smax[i]) && @constraint(model, S_end[i] <= lp.Smax[i])
    end

    # ----------------------------------------------------------------------
    # Reserve headroom (FOS §8 joint up/down constraints)
    # ----------------------------------------------------------------------
    if nprod > 0
        @variable(model, R[1:ngen, 1:nt, 1:nprod] >= 0)
        @variable(model, Rslack[1:nrg, 1:nt] >= 0)

        for u in 1:ngen, j in 1:nt, p in 1:nprod
            lp.g_res_pmax[u, p] == 0 && continue
            @constraint(model, R[u, j, p] <= lp.g_res_pmax[u, p])
            sched_cap = lp.g_res_sched[u, p, j]
            if isfinite(sched_cap)
                @constraint(model, R[u, j, p] <= sched_cap)
            end
        end

        _sym_products = Set(["FCR_N_UP", "FCR_N_DOWN"])
        up_prods = [p for p in 1:nprod if lp.prod_dir[p] == 1]
        dn_prods = [p for p in 1:nprod if lp.prod_dir[p] == -1]
        for u in 1:ngen, j in 1:nt
            up_capable = [p for p in up_prods if lp.g_res_pmax[u, p] > 0]
            sym_dn = [p for p in dn_prods if lp.PROD[p] in _sym_products && lp.g_res_pmax[u, p] > 0]
            up_all = vcat(up_capable, sym_dn)
            if !isempty(up_all)
                @constraint(model,
                    Pg[u, j] + sum(R[u, j, p] for p in up_all) <= lp.g_time_pcap[u, j])
            end
            dn_capable = [p for p in dn_prods if lp.g_res_pmax[u, p] > 0]
            sym_up = [p for p in up_prods if lp.PROD[p] in _sym_products && lp.g_res_pmax[u, p] > 0]
            dn_all = vcat(dn_capable, sym_up)
            if !isempty(dn_all)
                @constraint(model,
                    Pg[u, j] - sum(R[u, j, p] for p in dn_all) >= 0.0)
            end
        end

        for g in 1:nrg
            p = lp.rg_prod[g]
            p == 0 && continue
            for j in 1:nt
                @constraint(model,
                    sum(R[u, j, p] for u in 1:ngen if lp.g_res_pmax[u, p] > 0; init = 0.0) + Rslack[g, j] >=
                        lp.rg_oblig[g, j]
                )
            end
        end
    end

    # ----------------------------------------------------------------------
    # End-of-horizon water value
    # ----------------------------------------------------------------------
    if has_cut_groups
        ncg = length(lp.cut_groups)
        @variable(model, W_G[1:ncg])
        for (g, cg) in enumerate(lp.cut_groups)
            for c in 1:cg.ncuts
                @constraint(model,
                    W_G[g] <= cg.intercept[c] + sum(
                        cg.slopes[i, c] * S_end[cg.res_indices[i]]
                        for i in 1:length(cg.res_names) if cg.res_indices[i] > 0
                    )
                )
            end
        end
    elseif has_water_value
        @variable(model, W[wv_res_indices])
        for i in wv_res_indices, c in 1:lp.wv_ncuts[i]
            @constraint(model, W[i] <= lp.wv_ref[i][c] + lp.wv_slope[i][c] * S_end[i])
        end
    end

    wv_obj = if has_cut_groups
        sum(model[:W_G])
    elseif has_water_value
        sum(model[:W][i] for i in wv_res_indices)
    else
        0.0
    end

    # ----------------------------------------------------------------------
    # Objective: revenue − reserve shortfall − spill anti-degeneracy
    # ----------------------------------------------------------------------
    @objective(model, Max,
        sum(lp.price[j] * lp.dt_hours[j] *
                (sum(Pg[u, j] for u in 1:ngen) - sum(Ppump[v, j] for v in 1:npump))
            for j in 1:nt)
        + (nprod > 0 ?
            sum(lp.prod_price[p, j] * lp.dt_hours[j] * model[:R][u, j, p]
                for u in 1:ngen, j in 1:nt, p in 1:nprod if lp.g_res_pmax[u, p] > 0)
           : 0.0)
        - (nprod > 0 ?
            sum(lp.rg_penalty[g, j] * lp.dt_hours[j] * model[:Rslack][g, j]
                for g in 1:nrg, j in 1:nt if lp.rg_prod[g] > 0)
           : 0.0)
        + wv_obj
        - sum(1e-3 * lp.dt_hours[j] * Qspill[i, j] for i in 1:nr, j in 1:nt)
        - (njunc > 0 ?
            sum(lp.junc_min_penalty * lp.dt_hours[j] * Rj_min[k, j]
                for k in 1:njunc, j in 1:nt)
           : 0.0)
    )

    return lp, model
end
