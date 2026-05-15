"""
SHOP / Harmonie YAML reader. Ported from
`HydroModels_depr/src/parsing.jl` and adapted to:

- Produce the parametric Core types (`Plant{Float64}`, `Reservoir{Float64}`,
  …) rather than depr's `Float64`-only structs.
- Use Core's validated `PiecewiseLinear{Float64}` everywhere a hydraulic
  or efficiency curve appears. Curves with `length < 2` are silently
  dropped — depr fabricated a single-point dummy that our type cannot
  represent.
- Use `ReservoirWaterValue{Float64}` for per-reservoir end-of-horizon
  cuts (depr called this `WaterValueCuts`; renamed in A1 to avoid clash
  with Core's richer multi-stage `WaterValueCuts`).

The public entry point is [`read_shop_yaml`](@ref).
"""

"""
Canonical reserve-product table for unit-level participation:
`(product_name, pmin_keys, pmax_keys, schedule_key, direction)`.
`direction` is `true` for up-reserve (headroom above dispatch), `false`
for down-reserve. `pmin_keys` / `pmax_keys` are tried in order; the
first non-`nothing` match wins, mirroring SHOP's loose key conventions
(`p_fcr_max` vs `p_fcr_n_up_max`, `rr_up_max` vs `p_rr_up_max`, …).
"""
const _RESERVE_UNIT_PRODUCTS = [
    ("FCR_N_UP",   ["fcr_n_up_min", "p_fcr_min"],   ["p_fcr_n_up_max", "p_fcr_max"],   "fcr_n_up_schedule",   true),
    ("FCR_N_DOWN", ["fcr_n_down_min", "p_fcr_min"], ["p_fcr_n_down_max", "p_fcr_max"], "fcr_n_down_schedule", false),
    ("FCR_D_UP",   ["fcr_d_up_min", "p_fcr_min"],   ["p_fcr_d_up_max", "p_fcr_max"],   "fcr_d_up_schedule",   true),
    ("FCR_D_DOWN", ["fcr_d_down_min", "p_fcr_min"], ["p_fcr_d_down_max", "p_fcr_max"], "fcr_d_down_schedule", false),
    ("FRR_UP",     ["frr_up_min"],                   ["frr_up_max", "p_frr_up_max"],     "frr_up_schedule",     true),
    ("FRR_DOWN",   ["frr_down_min"],                 ["frr_down_max", "p_frr_down_max"], "frr_down_schedule",   false),
    ("RR_UP",      ["rr_up_min", "p_rr_min"],       ["rr_up_max", "p_rr_up_max"],       "rr_up_schedule",      true),
    ("RR_DOWN",    ["rr_down_min", "p_rr_min"],     ["rr_down_max", "p_rr_down_max"],   "rr_down_schedule",    false),
]

"""
Canonical reserve-product table for area-level groups:
`(product_name, obligation_key, penalty_key)`.
"""
const _RESERVE_GROUP_PRODUCTS = [
    ("FCR_N_UP",   "fcr_n_up_obligation",  "fcr_n_penalty_cost"),
    ("FCR_N_DOWN", "fcr_n_down_obligation", "fcr_n_penalty_cost"),
    ("FCR_D_UP",   "fcr_d_up_obligation",  "fcr_d_penalty_cost"),
    ("FCR_D_DOWN", "fcr_d_down_obligation", "fcr_d_penalty_cost"),
    ("FRR_UP",     "frr_up_obligation",    "frr_penalty_cost"),
    ("FRR_DOWN",   "frr_down_obligation",  "frr_penalty_cost"),
    ("RR_UP",      "rr_up_obligation",     "rr_penalty_cost"),
    ("RR_DOWN",    "rr_down_obligation",   "rr_penalty_cost"),
]

"""
Convert turbine efficiency curves to mechanical PQ curves at each
reference head. Power output is `K * href * η * Q` where `K = 0.00981`
(specific weight of water × g over a unit conversion factor),
assuming `Q` is in m³/s and `href` in m. Efficiency `y` may be in
percent or fraction; both are accepted.
"""
function _effcurves_to_PQ(curves_any)
    K = 0.00981
    lst = curves_any isa AbstractVector ? curves_any : [curves_any]
    out = Tuple{Float64, PiecewiseLinear{Float64}}[]
    for c in lst
        href = Float64(_get(c, "ref", _get(c, "href", 0.0)))
        Q = Float64.(_get(c, "x", Float64[]))
        y = Float64.(_get(c, "y", Float64[]))
        η = maximum(y; init = 0.0) > 1.0 ? y ./ 100.0 : y
        P = K .* href .* η .* Q
        pwl = _try_piecewise(Q, P)
        pwl === nothing && continue
        push!(out, (href, _ensure_origin(pwl)))
    end
    return out
end

"""
Extract direct PQ curves if present, otherwise derive them from
turbine efficiency curves. Returns `Vector{Tuple{Float64,
PiecewiseLinear{Float64}}}`, empty when no usable curves are found.
"""
function _extract_curves(unit::AbstractDict; pump::Bool = false)
    rawpq = _get(unit, "pq_curve", nothing)
    rawpq = rawpq === nothing ? _get(unit, "p_q_curve", nothing) : rawpq
    if rawpq !== nothing
        lst = rawpq isa AbstractVector ? rawpq : [rawpq]
        out = Tuple{Float64, PiecewiseLinear{Float64}}[]
        for c in lst
            href = Float64(_get(c, "ref", _get(c, "href", 0.0)))
            Q = Float64.(_get(c, "x", Float64[]))
            Y = Float64.(_get(c, "y", Float64[]))
            pwl = _try_piecewise(Q, Y)
            pwl === nothing && continue
            push!(out, (href, _ensure_origin(pwl)))
        end
        return out
    end

    rawη = _get(unit, "turb_eff_curves", nothing)
    return rawη === nothing ? Tuple{Float64, PiecewiseLinear{Float64}}[] :
        _effcurves_to_PQ(rawη)
end

# ============================================================
# Spill-routing inference (B1.5)
#
# Each reservoir's dam spillway physically flows into whatever module
# lies downstream of the dam. SHOP YAML doesn't encode this directly,
# but the generator topology does — generators take water from one
# reservoir and discharge it into another, and spillway water follows
# the same route by physical default. The inference rule picks the
# most common `g.to_res` among generators originating from each
# reservoir; lexicographic tie-break for determinism.
# ============================================================

"""
    _infer_spill_destinations(generators, tunnels, fwd_graph, reservoir_names)
        -> Dict{String, Union{Nothing,String}}

For each reservoir name in `reservoir_names`, return the destination
reservoir for surplus spill. Three-tier fallback:

1. **Most common `g.to_res` among generators with `g.from_res == r`.**
   The standard cascade case (a reservoir feeds one plant which
   discharges to a specific downstream reservoir).
2. **Most common `t.to_node` among parser-emitted virtual tunnels with
   `t.from_node == r`.** Handles cascades where an intake reservoir
   feeds the plant only through tunnels — the parser's river-resolution
   code at this point has already chased through declared `river`
   blocks to produce reservoir-to-reservoir virtual tunnels.
3. **BFS through `fwd_graph` to the nearest reservoir.** Walks the
   full connection graph (rivers, tunnels, KP junction nodes — every
   node referenced from `connections:` at the YAML root). Catches the
   case where the river chain `R → river → KP → river → R'` was not
   declared as a `river` block in `model.river:` and therefore not
   resolved by the parser's tunnel-emission code. Uses the same
   `_bfs_nearest` helper that `incoming_by_plant` / `outgoing_by_plant`
   use for plant inference, so the routing rule is consistent across
   plant- and reservoir-keyed inferences.

Lexicographic tie-break on destination name in tiers 1 and 2. BFS
returns the topologically nearest reservoir, deterministic given the
insertion order of `fwd_graph`.
"""
function _infer_spill_destinations(generators, tunnels,
                                  fwd_graph::Dict{String, Vector{String}},
                                  reservoir_names::AbstractSet{String},
                                  junction_names::AbstractSet{String} =
                                      Set{String}())
    gen_counts = Dict{String, Dict{String, Int}}()
    for g in generators
        # B1.6: `g.from_res` is `Vector{String}` — iterate over every
        # source. Single-source generators (most cases today) behave
        # identically because the vector has one entry.
        to = g.to_res
        isempty(to) && continue
        to in reservoir_names || continue
        for from in g.from_res
            isempty(from) && continue
            from in reservoir_names || continue
            to == from && continue
            d = get!(gen_counts, from, Dict{String, Int}())
            d[to] = get(d, to, 0) + 1
        end
    end

    tun_counts = Dict{String, Dict{String, Int}}()
    for t in tunnels
        from = t.from_node
        to   = t.to_node
        isempty(from) && continue
        isempty(to)   && continue
        from in reservoir_names || continue
        to   in reservoir_names || continue
        to == from && continue
        d = get!(tun_counts, from, Dict{String, Int}())
        d[to] = get(d, to, 0) + 1
    end

    function _pick(c)
        c === nothing && return nothing
        isempty(c)    && return nothing
        ranked = sort!(collect(c); by = pair -> (-pair[2], pair[1]))
        return ranked[1][1]
    end

    # B1.7: the BFS target set widens to reservoirs ∪ junctions so
    # terminal reservoirs (Limarka, Eikrebekken, Strandefjorden,
    # Vassbygdvatn in NO5) can resolve to a typed Junction (BergheimKP,
    # VangenSeaLevelKP) instead of falling through to `nothing`. The
    # earlier generator + tunnel tiers still only target reservoirs;
    # only the final BFS fallback opens up to junctions.
    bfs_targets = isempty(junction_names) ? reservoir_names :
                  union(reservoir_names, junction_names)
    result = Dict{String, Union{Nothing, String}}()
    for r in reservoir_names
        dest = _pick(get(gen_counts, r, nothing))
        if dest === nothing
            dest = _pick(get(tun_counts, r, nothing))
        end
        if dest === nothing
            # Tier 3: BFS through the full connection graph. Crosses
            # KP nodes and any other intermediate-node type the parser
            # might not have explicit handling for.
            bfs_hit = _bfs_nearest(r, fwd_graph, bfs_targets)
            dest = isempty(bfs_hit) ? nothing : bfs_hit
        end
        result[r] = dest
    end
    return result
end

"""
    _break_spill_cycles!(spill_dest)

Detect cycles in the spill-destination graph and break the back-edge
with a warning. The chain follows generator `from_res → to_res` which
is acyclic for any physically-realisable cascade, so this is purely
defensive.
"""
function _break_spill_cycles!(spill_dest::Dict{String, Union{Nothing, String}})
    for start in collect(keys(spill_dest))
        visited = String[]
        node = start
        while node !== nothing && !(node in visited)
            push!(visited, node)
            node = get(spill_dest, node, nothing)
        end
        if node !== nothing && node in visited
            @warn "Spill routing cycle detected starting at \"$start\"; " *
                  "breaking the back-edge at \"$node\"."
            spill_dest[node] = nothing
        end
    end
    return spill_dest
end

"""
    read_shop_yaml(path; market_name = "NO3",
                   generator_overrides = Dict(),
                   pump_overrides = Dict())

Parse a SHOP / Harmonie YAML file into typed `HydroModelsCore` objects.

# Keyword arguments
- `market_name`: preferred market block key, e.g. `"NO3"` or `"NO5"`.
  Emits a warning and falls back to the longest available `sale_price`
  series when the key is not found.
- `generator_overrides`, `pump_overrides`: `Dict(unit_name => (from_res,
  to_res))` overrides for the automatically inferred unit-reservoir
  connections.

# Returns
A `NamedTuple` with these keys:
- `plants::Dict{String, Plant{Float64}}`
- `reservoirs::Dict{String, Reservoir{Float64}}`
- `generators::Vector{Generator{Float64}}`
- `pumps::Vector{Pump{Float64}}`
- `tunnels::Vector{Tunnel{Float64}}`
- `market::MarketSeries{Float64}` — day-ahead price series.
- `dt_schedule::Vector{Tuple{DateTime, Float64}}` — `(from_time,
  dt_minutes)` breakpoints, sorted by time.
- `reserve_groups::Vector{ReserveGroup{Float64}}` — area-level
  obligations.
- `reserve_prices::Dict{String, OrderedDict{DateTime, Float64}}` —
  per-product market sale prices.
- `cut_groups::Vector{CutGroup{Float64}}` — group-level Benders cuts.
"""
function read_shop_yaml(
        path::AbstractString;
        market_name::AbstractString = "NO3",
        generator_overrides::AbstractDict = Dict(),
        pump_overrides::AbstractDict = Dict(),
    )

    gen_map = _normalize_map(generator_overrides)
    pump_map = _normalize_map(pump_overrides)

    raw = YAML.load_file(path)
    model = _get(raw, "model", Dict())

    # ----------------------------------------------------------------------
    # Market
    # ----------------------------------------------------------------------
    market_block = _get(model, "market", Dict())
    market_block isa AbstractDict || (market_block = Dict())
    market_ts = OrderedDict{DateTime, Float64}()
    selected_market = ""

    preferred = _get(market_block, String(market_name), nothing)
    if preferred isa AbstractDict
        sp = _get(preferred, "sale_price", nothing)
        if sp !== nothing
            market_ts = _ordered_datetime_float(sp)
            selected_market = market_name
        end
    end

    if isempty(market_ts)
        available = [
            string(k) for (k, v) in market_block
            if v isa AbstractDict && _get(v, "sale_price", nothing) !== nothing
        ]
        if !isempty(available)
            if preferred === nothing || !(preferred isa AbstractDict)
                @warn "Market \"$market_name\" not found in YAML. Available markets: [$(join(available, ", "))]. Falling back to longest series."
            else
                @warn "Market \"$market_name\" has no sale_price data. Available markets: [$(join(available, ", "))]. Falling back to longest series."
            end
        end
        best_len = -1
        for (mname, mkt) in market_block
            mkt isa AbstractDict || continue
            sp = _get(mkt, "sale_price", nothing)
            sp === nothing && continue
            cand = _ordered_datetime_float(sp)
            if length(cand) > best_len
                market_ts = cand
                best_len = length(cand)
                selected_market = string(mname)
            end
        end
        isempty(market_ts) || @info "Using market \"$selected_market\" with $(length(market_ts)) timesteps."
    else
        @info "Using market \"$selected_market\" with $(length(market_ts)) timesteps."
    end

    isempty(market_ts) && @warn "No market price series found. All prices will be zero."

    market = MarketSeries(market_ts)

    # Reserve market sale_price series, keyed by canonical product name.
    _known_reserve_products = Set(p for (p, _, _, _, _) in _RESERVE_UNIT_PRODUCTS)
    reserve_prices = Dict{String, OrderedDict{DateTime, Float64}}()
    for (_mname, mkt) in market_block
        mkt isa AbstractDict || continue
        mt = _get(mkt, "market_type", nothing)
        mt === nothing && continue
        mt_s = String(mt)
        mt_s in _known_reserve_products || continue
        sp = _get(mkt, "sale_price", nothing)
        sp === nothing && continue
        ts = _ordered_datetime_float(sp)
        isempty(ts) && continue
        reserve_prices[mt_s] = ts
    end
    isempty(reserve_prices) || @info "Parsed reserve market prices for: [$(join(sort(collect(keys(reserve_prices))), ", "))]"

    # ----------------------------------------------------------------------
    # Plants
    # ----------------------------------------------------------------------
    plants = Dict{String, Plant{Float64}}()
    plant_block = _get(model, "plant", Dict())
    for (pname, p) in plant_block
        pname_s = String(pname)
        plants[pname_s] = Plant{Float64}(
            name             = pname_s,
            outlet_line      = _as_float(_get(p, "outlet_line", 0.0), 0.0),
            main_loss        = Float64.(_get(p, "main_loss", Float64[])),
            penstock_loss    = Float64.(_get(p, "penstock_loss", Float64[])),
            maintenance_flag = _ordered_datetime_int(_get(p, "maintenance_flag", Dict())),
            max_p_constr     = _ordered_datetime_float(_get(p, "max_p_constr", Dict())),
        )
    end

    # ----------------------------------------------------------------------
    # Reservoirs
    # ----------------------------------------------------------------------
    reservoirs = Dict{String, Reservoir{Float64}}()
    res_block = _get(model, "reservoir", Dict())
    for (rname, r) in res_block
        rname_s = String(rname)

        vol_head = _get(r, "vol_head", Dict())
        V = Float64.(_get(vol_head, "x", Float64[]))
        H = Float64.(_get(vol_head, "y", Float64[]))

        inflow = InflowSeries(_ordered_datetime_float(_get(r, "inflow", Dict())))

        s0 = if _has(r, "start_vol")
            Float64(_get(r, "start_vol", 0.0))
        elseif _has(r, "start_head")
            v = _vol_from_head(V, H, Float64(_get(r, "start_head", 0.0)))
            isfinite(v) ? v : 0.0
        else
            0.0
        end

        smin = if _has(r, "lrl")
            v = _vol_from_head(V, H, Float64(_get(r, "lrl", 0.0)))
            isfinite(v) ? v : 0.0
        else
            0.0
        end

        smax = if _has(r, "hrl")
            v = _vol_from_head(V, H, Float64(_get(r, "hrl", 0.0)))
            isfinite(v) ? v : (isempty(V) ? Inf : maximum(V))
        else
            Float64(_get(r, "max_vol", isempty(V) ? Inf : maximum(V)))
        end

        # Flood start: s0 may legitimately exceed the hrl-derived smax.
        # Relax smax to the physical curve maximum so the LP initial
        # condition stays feasible.
        if s0 > smax && !isempty(V)
            smax_phys = maximum(V)
            new_smax = s0 <= smax_phys ? smax_phys : s0
            @warn "Reservoir $rname_s: s0=$(round(s0, sigdigits = 5)) > hrl-smax=$(round(smax, sigdigits = 5)). Relaxing smax to $(round(new_smax, sigdigits = 5))."
            smax = new_smax
        end

        # Per-reservoir water value cuts: each SDDP cut has
        #   ref  = intercept (EUR)
        #   x[s] = end-of-horizon volume for scenario s (Mm³)
        #   y[s] = marginal water value for scenario s (EUR/Mm³)
        # We use the mean over scenarios as the deterministic slope.
        wv_raw = _get(r, "water_value_input", nothing)
        water_value = if wv_raw isa AbstractVector && !isempty(wv_raw)
            wv_ref_vec = Float64[]
            wv_slope_vec = Float64[]
            for entry in wv_raw
                ref_val = _as_float(_get(entry, "ref", nothing), 0.0)
                ys = _get(entry, "y", [])
                isempty(ys) && continue
                slope_eur_per_mm3 = sum(Float64, ys) / length(ys)
                push!(wv_ref_vec, ref_val)
                push!(wv_slope_vec, slope_eur_per_mm3)
            end
            isempty(wv_slope_vec) ? nothing :
                ReservoirWaterValue(wv_ref_vec, wv_slope_vec)
        else
            nothing
        end

        reservoirs[rname_s] = Reservoir{Float64}(
            name        = rname_s,
            vol_breaks  = V,
            head_breaks = H,
            s_min       = smin,
            s_max       = smax,
            s0          = s0,
            inflow      = inflow,
            water_value = water_value,
        )
    end

    plant_names = Set(keys(plants))
    reservoir_names = Set(keys(reservoirs))

    # ----------------------------------------------------------------------
    # Tunnels and routing inference
    # ----------------------------------------------------------------------
    tunnels = Tunnel{Float64}[]
    tunnel_block = _get(model, "tunnel", Dict())
    tunnel_block isa AbstractDict || (tunnel_block = Dict())

    _tunnel_names = Set(String(t) for t in keys(tunnel_block))
    _tun_upstream = Dict{String, Vector{String}}()
    _tun_downstream = Dict{String, Vector{String}}()
    for conn in _get(raw, "connections", [])
        conn isa AbstractDict || continue
        from_s = String(_get(conn, "from", ""))
        to_s = String(_get(conn, "to", ""))
        ft = String(_get(conn, "from_type", ""))
        tt = String(_get(conn, "to_type", ""))

        if (tt == "tunnel" || isempty(tt)) && to_s in _tunnel_names
            push!(get!(_tun_upstream, to_s, String[]), from_s)
        end
        if (ft == "tunnel" || isempty(ft)) && from_s in _tunnel_names
            push!(get!(_tun_downstream, from_s, String[]), to_s)
        end
    end

    # Resolve each tunnel's reachable sources/targets through KP-waypoint
    # chains via DFS with memoization.
    function _resolve_sources(
            tun, upstream_map,
            memo = Dict{String, Vector{String}}(),
        )
        haskey(memo, tun) && return memo[tun]
        result = String[]
        for src in get(upstream_map, tun, String[])
            if src in plant_names || src in reservoir_names
                push!(result, src)
            elseif src in _tunnel_names
                append!(result, _resolve_sources(src, upstream_map, memo))
            end
        end
        memo[tun] = result
        return result
    end

    function _resolve_targets(
            tun, downstream_map,
            memo = Dict{String, Vector{String}}(),
        )
        haskey(memo, tun) && return memo[tun]
        result = String[]
        for tgt in get(downstream_map, tun, String[])
            if tgt in plant_names || tgt in reservoir_names
                push!(result, tgt)
            elseif tgt in _tunnel_names
                append!(result, _resolve_targets(tgt, downstream_map, memo))
            end
        end
        memo[tun] = result
        return result
    end

    fwd_graph = Dict{String, Vector{String}}()
    rev_graph = Dict{String, Vector{String}}()
    # B1.7 — `full_conn_graph` includes every (from, to) edge from
    # the YAML `connections:` block, regardless of node type. Used
    # by the spill-inference BFS so terminal reservoirs can chase
    # through river chains and KP junctions to reach the typed
    # `Junction{T}` targets (e.g. Limarka → w_Limarka_LioddenKP →
    # w_LioddenKP_BergheimKP → BergheimKP). The narrower
    # `fwd_graph` / `rev_graph` (tunnel-resolved) remain for
    # plant-keyed inference where we deliberately do NOT want to
    # cross river chains.
    full_conn_graph = Dict{String, Vector{String}}()
    for conn in _get(raw, "connections", [])
        conn isa AbstractDict || continue
        fs = String(_get(conn, "from", ""))
        ts = String(_get(conn, "to", ""))
        (isempty(fs) || isempty(ts)) && continue
        push!(get!(full_conn_graph, fs, String[]), ts)
    end
    src_memo = Dict{String, Vector{String}}()
    tgt_memo = Dict{String, Vector{String}}()
    for tun in _tunnel_names
        sources = _resolve_sources(tun, _tun_upstream, src_memo)
        targets = _resolve_targets(tun, _tun_downstream, tgt_memo)
        for s in sources, t in targets
            push!(get!(fwd_graph, s, String[]), t)
            push!(get!(rev_graph, t, String[]), s)
        end
    end

    # BFS from each plant: upstream uses capacity-weighted choice (prefer
    # large headwater over small forebay through the same KP chain);
    # downstream uses nearest-first.
    res_capacity = Dict{String, Float64}(
        name => (isfinite(reservoirs[name].s_max) ? reservoirs[name].s_max : 0.0)
        for name in reservoir_names
    )
    incoming_by_plant = Dict{String, String}()
    outgoing_by_plant = Dict{String, String}()
    # B1.6 — `incoming_by_plant_all` captures every reservoir reachable
    # upstream of the plant via the tunnel-resolved graph. Multi-source
    # plants (Pattern B/C cascades like Hemsil2 ← Eikrebekken/Logga/Ruståni)
    # get ≥ 2 entries here even though `incoming_by_plant` (singular)
    # keeps the dominant-capacity choice for legacy callers.
    incoming_by_plant_all = Dict{String, Vector{String}}()
    for plant in plant_names
        from = _bfs_nearest(plant, rev_graph, reservoir_names; capacity = res_capacity)
        isempty(from) || (incoming_by_plant[plant] = from)
        to = _bfs_nearest(plant, fwd_graph, reservoir_names)
        isempty(to) || (outgoing_by_plant[plant] = to)
        # All upstream reservoirs, sorted by capacity desc.
        ups = _bfs_all_reservoirs(plant, rev_graph, reservoir_names;
                                  capacity = res_capacity)
        isempty(ups) || (incoming_by_plant_all[plant] = ups)
    end

    # Terminal-tunnel emission with bottleneck-tracking source resolution.
    _intermediate_tuns = Set{String}()
    for tn in _tunnel_names
        dests = get(_tun_downstream, tn, String[])
        if !isempty(dests) && all(d -> d in _tunnel_names, dests)
            push!(_intermediate_tuns, tn)
        end
    end

    function _tunnel_effective_cap(td)
        td isa AbstractDict || return Inf
        cap = Float64(_get(td, "q_max", _get(td, "qmax", Inf)))
        # Closed gate (all schedule values zero) zeros the cap.
        gos = _get(td, "gate_opening_schedule", nothing)
        if gos isa AbstractDict
            vals = collect(values(gos))
            if !isempty(vals) && all(v -> v isa Number && v == 0, vals)
                cap = 0.0
            end
        elseif gos isa Number && gos == 0
            cap = 0.0
        end
        return cap
    end

    _chain_src_memo = Dict{String, Vector{Tuple{String, Float64}}}()
    _chain_src_ip = Set{String}()
    function _chain_sources_cap(tun)
        haskey(_chain_src_memo, tun) && return _chain_src_memo[tun]
        tun in _chain_src_ip && return Tuple{String, Float64}[]
        push!(_chain_src_ip, tun)
        td = get(tunnel_block, tun, nothing)
        my_cap = _tunnel_effective_cap(td)
        result = Tuple{String, Float64}[]
        for src in get(_tun_upstream, tun, String[])
            if src in _tunnel_names
                for (s, cap) in _chain_sources_cap(src)
                    push!(result, (s, min(my_cap, cap)))
                end
            else
                push!(result, (src, my_cap))
            end
        end
        _chain_src_memo[tun] = result
        delete!(_chain_src_ip, tun)
        return result
    end

    # B1.6 — accumulate per-(plant, source) tunnel-chain capacity. For each
    # tunnel that terminates at a plant `p`, `source_qmax_by_plant[p][r]`
    # gets the sum of chain_qmax across every penstock connecting source
    # reservoir `r` to `p`. Used by the LP to bound per-source flow when
    # generators withdraw from multiple intake reservoirs simultaneously.
    source_qmax_by_plant = Dict{String, Dict{String, Float64}}()

    for (tname, t) in tunnel_block
        tname_s = String(tname)
        tname_s in _intermediate_tuns && continue

        tgts = get(tgt_memo, tname_s, String[])
        isempty(tgts) && continue
        sources_caps = _chain_sources_cap(tname_s)
        isempty(sources_caps) && continue

        # Pass 1 — record per-source capacity contribution to each plant
        # endpoint, BEFORE the plant→reservoir remapping below.
        for b_raw in tgts
            b_raw in plant_names || continue
            plant_dict = get!(source_qmax_by_plant, b_raw, Dict{String, Float64}())
            for (a_raw, chain_qmax) in sources_caps
                a_raw in reservoir_names || continue
                plant_dict[a_raw] = get(plant_dict, a_raw, 0.0) + chain_qmax
            end
        end

        for b_raw in tgts
            b = b_raw
            if b in plant_names
                # B1.6: for multi-source plants, the LP per-source
                # `Qg_from[u, s, j]` machinery already debits each
                # upstream reservoir directly. The phantom virtual
                # reservoir-to-reservoir tunnel (which remaps a
                # non-dominant source to the dominant one) would
                # introduce a degenerate alternative routing path
                # and a misleading inflow into the dominant reservoir.
                # Skip emission for these plants.
                if length(get(incoming_by_plant_all, b, String[])) > 1
                    continue
                end
                b = get(incoming_by_plant, b, b)
            end
            isempty(b) && continue

            for (a_raw, chain_qmax) in sources_caps
                a = a_raw
                if a in plant_names
                    a = get(outgoing_by_plant, a, a)
                end
                a == b && !isempty(a) && continue
                isempty(a) && continue

                push!(tunnels, Tunnel{Float64}(
                    name         = tname_s,
                    from_node    = a,
                    to_node      = b,
                    qmax         = chain_qmax,
                    start_height = Float64(_get(t, "start_height", 0.0)),
                    end_height   = Float64(_get(t, "end_height", 0.0)),
                    loss_factor  = Float64(_get(t, "loss_factor", 0.0)),
                ))
            end
        end
    end

    # ----------------------------------------------------------------------
    # River routing → virtual tunnels with bottleneck max_flow
    # ----------------------------------------------------------------------
    _river_names = Set{String}()
    river_block = _get(model, "river", Dict())
    if river_block isa AbstractDict
        for rn in keys(river_block)
            push!(_river_names, String(rn))
        end
    end

    _river_maxflow = Dict{String, Float64}()
    if river_block isa AbstractDict
        for (rn, rd) in river_block
            rd isa AbstractDict || continue
            mf = _get(rd, "max_flow", nothing)
            mf === nothing && continue
            _river_maxflow[String(rn)] = _scalar_or_first(mf, Inf)
        end
    end

    # ----------------------------------------------------------------------
    # B1.7 — Junction (KP) parsing. A river block becomes a typed
    # Junction only when:
    # (a) the name does NOT have the river-segment prefix `b_/f_/w_`
    #     (those are reserved for river *edges* between two endpoints —
    #     b_X_Y = bottom outlet, f_X_Y = free spill, w_X_Y = water
    #     flow), and
    # (b) it carries at least one constraint field (`max_flow` /
    #     `flow` time-series).
    # In NO5: BergheimKP and VangenSeaLevelKP qualify; the ~97 river-
    # edge blocks with `upstream_elevation` are correctly excluded.
    # Their min-flow constraints (per-segment regulatory release) are
    # a separate, more involved feature — they'd need the LP to keep
    # per-river-segment variables alive instead of collapsing chains
    # to reservoir-to-reservoir virtual tunnels. Out of scope for B1.7.
    # ----------------------------------------------------------------------
    function _is_junction_name(s::AbstractString)
        # River-segment prefixes (length-2 with underscore): "b_", "f_", "w_".
        length(s) >= 2 || return true
        return !(s[1] in ('b', 'f', 'w') && s[2] == '_')
    end
    junctions = Dict{String, Junction{Float64}}()
    if river_block isa AbstractDict
        for (rn, rd) in river_block
            rd isa AbstractDict || continue
            name_s = String(rn)
            _is_junction_name(name_s) || continue
            has_max  = haskey(rd, "max_flow") &&
                       _get(rd, "max_flow", nothing) isa AbstractDict
            has_min  = haskey(rd, "flow") &&
                       _get(rd, "flow", nothing) isa AbstractDict
            (has_max || has_min) || continue
            junctions[name_s] = Junction{Float64}(
                name               = name_s,
                upstream_elevation = Float64(_get(rd, "upstream_elevation", 0.0)),
                max_flow           = has_max ?
                    _ordered_datetime_float(_get(rd, "max_flow", Dict())) :
                    OrderedDict{DateTime, Float64}(),
                flow               = has_min ?
                    _ordered_datetime_float(_get(rd, "flow", Dict())) :
                    OrderedDict{DateTime, Float64}(),
            )
        end
    end
    isempty(junctions) ||
        @info "Parsed $(length(junctions)) junction(s): [$(join(sort!(collect(keys(junctions))), ", "))]"

    _riv_upstream = Dict{String, Vector{String}}()
    _riv_downstream = Dict{String, Vector{String}}()
    for conn in _get(raw, "connections", [])
        conn isa AbstractDict || continue
        from_s = String(_get(conn, "from", ""))
        to_s = String(_get(conn, "to", ""))
        ft = String(_get(conn, "from_type", ""))
        tt = String(_get(conn, "to_type", ""))

        if tt == "river" && to_s in _river_names
            push!(get!(_riv_upstream, to_s, String[]), from_s)
        end
        if ft == "river" && from_s in _river_names
            push!(get!(_riv_downstream, from_s, String[]), to_s)
        end
    end

    function _resolve_river_sources_cap(
            riv, cap_so_far,
            memo = Dict{String, Vector{Tuple{String, Float64}}}(),
        )
        haskey(memo, riv) && return [(s, min(cap_so_far, c)) for (s, c) in memo[riv]]
        result = Tuple{String, Float64}[]
        for src in get(_riv_upstream, riv, String[])
            if src in reservoir_names || src in plant_names
                push!(result, (src, cap_so_far))
            elseif src in _river_names
                src_cap = get(_river_maxflow, src, Inf)
                bn = min(cap_so_far, src_cap)
                append!(result, _resolve_river_sources_cap(src, bn, memo))
            end
        end
        memo[riv] = result
        return [(s, min(cap_so_far, c)) for (s, c) in result]
    end

    function _resolve_river_targets_cap(
            riv, cap_so_far,
            memo = Dict{String, Vector{Tuple{String, Float64}}}(),
        )
        haskey(memo, riv) && return [(t, min(cap_so_far, c)) for (t, c) in memo[riv]]
        result = Tuple{String, Float64}[]
        for tgt in get(_riv_downstream, riv, String[])
            if tgt in reservoir_names
                push!(result, (tgt, cap_so_far))
            elseif tgt in _river_names
                tgt_cap = get(_river_maxflow, tgt, Inf)
                bn = min(cap_so_far, tgt_cap)
                append!(result, _resolve_river_targets_cap(tgt, bn, memo))
            end
        end
        memo[riv] = result
        return [(t, min(cap_so_far, c)) for (t, c) in result]
    end

    n_river_tunnels = 0
    riv_src_cap_memo = Dict{String, Vector{Tuple{String, Float64}}}()
    riv_tgt_cap_memo = Dict{String, Vector{Tuple{String, Float64}}}()
    for riv in _river_names
        riv_cap = get(_river_maxflow, riv, Inf)
        sources = _resolve_river_sources_cap(riv, riv_cap, riv_src_cap_memo)
        targets = _resolve_river_targets_cap(riv, riv_cap, riv_tgt_cap_memo)
        for (s, s_cap) in sources, (t, t_cap) in targets
            s in reservoir_names && t in reservoir_names || continue
            s == t && continue
            qmax = min(s_cap, t_cap)
            qmax <= 0 && continue
            push!(tunnels, Tunnel{Float64}(
                name         = "_river_$(riv)",
                from_node    = s,
                to_node      = t,
                qmax         = qmax,
                start_height = 0.0,
                end_height   = 0.0,
                loss_factor  = 0.0,
            ))
            n_river_tunnels += 1
        end
    end

    n_river_tunnels > 0 && @info "Added $n_river_tunnels virtual river tunnels for water routing."

    # ----------------------------------------------------------------------
    # Generators
    # ----------------------------------------------------------------------
    generators = Generator{Float64}[]
    for (gname, g) in _get(model, "generator", Dict())
        gname_s = String(gname)
        plant = String(_unit_plant(gname_s))
        curves = _extract_curves(g; pump = false)

        ge = _get(g, "gen_eff_curve", nothing)
        gen_eff = nothing
        if ge !== nothing
            x = Float64.(_get(ge, "x", Float64[]))
            y = Float64.(_get(ge, "y", Float64[]))
            η = maximum(y; init = 0.0) > 1.0 ? y ./ 100.0 : y
            pwl_eff = _try_piecewise(x, η)
            gen_eff = pwl_eff === nothing ? nothing : _ensure_origin(pwl_eff)
        end

        # B1.6: `default_from` is now `Vector{String}` (every upstream
        # reservoir reachable via tunnels). User-supplied `gen_map`
        # overrides may still pass a scalar — wrap to a 1-vector below.
        default_from, default_to = _infer_generator_connection(
            plant, incoming_by_plant_all, outgoing_by_plant)
        from_res_raw, to_res = get(gen_map, gname_s, (default_from, default_to))
        from_res = from_res_raw isa AbstractString ?
            (isempty(from_res_raw) ? String[] : String[String(from_res_raw)]) :
            Vector{String}(from_res_raw)

        max_p = _ordered_datetime_float(_get(g, "max_p_constr", Dict()))
        if haskey(plants, plant)
            for (t, v) in plants[plant].max_p_constr
                max_p[t] = haskey(max_p, t) ? min(max_p[t], v) : v
            end
        end

        maint = _ordered_datetime_int(_get(g, "maintenance_flag", Dict()))
        if haskey(plants, plant)
            for (t, v) in plants[plant].maintenance_flag
                maint[t] = max(get(maint, t, 0), v)
            end
        end

        reserves = ReserveSpec{Float64}[]
        for (prod, pmin_keys, pmax_keys, sched_key, _) in _RESERVE_UNIT_PRODUCTS
            pmax_v = 0.0
            for k in pmax_keys
                v = _get(g, k, nothing)
                v === nothing && continue
                pmax_v = _scalar_or_first(v, 0.0)
                break
            end
            pmax_v > 0 || continue
            pmin_v = 0.0
            for k in pmin_keys
                v = _get(g, k, nothing)
                v === nothing && continue
                pmin_v = _scalar_or_first(v, 0.0)
                break
            end
            sched = _ordered_datetime_float(_get(g, sched_key, Dict()))
            push!(reserves, ReserveSpec{Float64}(
                product      = prod,
                pmin         = pmin_v,
                pmax         = pmax_v,
                schedule     = sched,
                penalty_cost = OrderedDict{DateTime, Float64}(),
            ))
        end

        # B1.6: from_res arrives as Vector{String} from
        # _infer_generator_connection. For multi-source plants the
        # vector has ≥ 2 entries; for single-source ones it has 1.
        # `source_qmax` is the per-source tunnel-chain capacity used by
        # the LP to bound `Qg_from[u, s, j]`.
        gen_source_qmax = get(source_qmax_by_plant, plant, Dict{String, Float64}())

        push!(generators, Generator{Float64}(
            name             = gname_s,
            plant            = plant,
            from_res         = from_res,
            source_qmax      = gen_source_qmax,
            to_res           = to_res,
            curves           = curves,
            qmin             = Float64(_get(g, "q_min", _get(g, "qmin", 0.0))),
            qmax             = Float64(_get(g, "q_max", _get(g, "qmax", Inf))),
            pmin             = Float64(_get(g, "p_min", 0.0)),
            pmax             = Float64(_get(g, "p_max", Inf)),
            p_nom            = Float64(_get(g, "p_nom", 0.0)),
            penstock         = Int(round(_scalar_or_first(_get(g, "penstock", 0), 0))),
            gen_eff          = gen_eff,
            max_p_constr     = max_p,
            maintenance_flag = maint,
            startcost        = _scalar_or_first(_get(g, "startcost", 0.0), 0.0),
            initial_state    = Int(round(_scalar_or_first(_get(g, "initial_state", 0), 0))),
            reserves         = reserves,
        ))
    end

    # ----------------------------------------------------------------------
    # Spill routing — rebuild reservoirs with `spill_to_reservoir`
    # inferred from the generator topology. The SHOP YAML schema has no
    # explicit spill-destination field; physically, dam spillway flow
    # follows the same route as turbine discharge, so the inference
    # picks the most common `g.to_res` per origin reservoir.
    # See `_infer_spill_destinations` for the rule.
    # ----------------------------------------------------------------------
    junction_names_set = Set(keys(junctions))
    # B1.7 — use the full connection graph (includes river edges)
    # so the tier-3 BFS can chase through river chains to typed
    # junctions; falls back to the tunnel-resolved `fwd_graph` when
    # no junctions are declared (preserves B1.5 behaviour bit-
    # identically on the existing fixtures).
    spill_bfs_graph = isempty(junction_names_set) ? fwd_graph : full_conn_graph
    spill_dests = _infer_spill_destinations(generators, tunnels,
                                           spill_bfs_graph,
                                           reservoir_names, junction_names_set)
    _break_spill_cycles!(spill_dests)
    for (rname, r) in collect(reservoirs)
        reservoirs[rname] = Reservoir{Float64}(
            name               = r.name,
            vol_breaks         = r.vol_breaks,
            head_breaks        = r.head_breaks,
            s_min              = r.s_min,
            s_max              = r.s_max,
            s0                 = r.s0,
            inflow             = r.inflow,
            water_value        = r.water_value,
            spill_to_reservoir = spill_dests[rname],
        )
    end
    n_routed = count(!isnothing, values(spill_dests))
    @info "Inferred spill routing: $n_routed of $(length(spill_dests)) " *
          "reservoirs have a downstream destination."

    # ----------------------------------------------------------------------
    # Pumps
    # ----------------------------------------------------------------------
    pumps = Pump{Float64}[]
    for (pname, p) in _get(model, "pump", Dict())
        pname_s = String(pname)
        plant = String(_unit_plant(pname_s))
        curves = _extract_curves(p; pump = true)

        default_from, default_to = _infer_pump_connection(
            plant, incoming_by_plant, outgoing_by_plant)
        from_res_raw, to_res = get(pump_map, pname_s, (default_from, default_to))
        from_res = from_res_raw isa AbstractString ?
            (isempty(from_res_raw) ? String[] : String[String(from_res_raw)]) :
            Vector{String}(from_res_raw)

        maint = _ordered_datetime_int(_get(p, "maintenance_flag", Dict()))
        if haskey(plants, plant)
            for (t, v) in plants[plant].maintenance_flag
                maint[t] = max(get(maint, t, 0), v)
            end
        end

        # B1.6: Pump.source_qmax stays empty for single-source pumps.
        # Multi-source pumps are not produced by the current parser
        # path; the schema supports them but no test case exercises it.
        push!(pumps, Pump{Float64}(
            name             = pname_s,
            plant            = plant,
            from_res         = from_res,
            to_res           = to_res,
            curves           = curves,
            qmin             = Float64(_get(p, "q_min", _get(p, "qmin", 0.0))),
            qmax             = Float64(_get(p, "q_max", _get(p, "qmax", Inf))),
            pmin             = Float64(_get(p, "p_min", 0.0)),
            pmax             = Float64(_get(p, "p_max", Inf)),
            p_nom            = Float64(_get(p, "p_nom", 0.0)),
            penstock         = Int(round(_scalar_or_first(_get(p, "penstock", 0), 0))),
            maintenance_flag = maint,
            startcost        = _scalar_or_first(_get(p, "startcost", 0.0), 0.0),
            initial_state    = Int(round(_scalar_or_first(_get(p, "initial_state", 0), 0))),
        ))
    end

    # ----------------------------------------------------------------------
    # Timestep schedule (variable resolution)
    # ----------------------------------------------------------------------
    time_block = _get(raw, "time", Dict())
    timeres = _get(time_block, "timeresolution", nothing)
    dt_schedule = Tuple{DateTime, Float64}[]
    if timeres isa AbstractDict && !isempty(timeres)
        for (t, m) in timeres
            dt = _parse_dt(t)
            dt === nothing && continue
            push!(dt_schedule, (dt, Float64(m)))
        end
        sort!(dt_schedule, by = first)
    end
    if isempty(dt_schedule)
        dt_schedule = [(DateTime(0), 60.0)]
    end

    # ----------------------------------------------------------------------
    # Reserve groups
    # ----------------------------------------------------------------------
    reserve_groups = ReserveGroup{Float64}[]
    rg_block = _get(model, "reserve_group", Dict())
    if rg_block isa AbstractDict
        for (rgname, rg) in rg_block
            rg isa AbstractDict || continue
            rgname_s = String(rgname)
            for (prod, oblig_key, penalty_key) in _RESERVE_GROUP_PRODUCTS
                oblig_raw = _get(rg, oblig_key, nothing)
                oblig_raw === nothing && continue
                oblig = _ordered_datetime_float(oblig_raw)
                isempty(oblig) && continue
                penalty = _ordered_datetime_float(_get(rg, penalty_key, Dict()))
                push!(reserve_groups, ReserveGroup{Float64}(
                    name         = rgname_s,
                    product      = prod,
                    obligation   = oblig,
                    penalty_cost = penalty,
                ))
            end
        end
    end
    isempty(reserve_groups) || @info "Parsed $(length(reserve_groups)) reserve group obligations."

    # ----------------------------------------------------------------------
    # Cut groups (group-level Benders cuts from SDDP)
    # ----------------------------------------------------------------------
    cut_groups = CutGroup{Float64}[]
    cg_section = _get(model, "cut_group", Dict())
    if cg_section isa AbstractDict && !isempty(cg_section)
        cg_members = Dict{String, Vector{Tuple{Int, String}}}()
        for conn in _get(raw, "connections", [])
            conn isa AbstractDict || continue
            _get(conn, "to_type", "") == "cut_group" || continue
            _get(conn, "from_type", "") == "reservoir" || continue
            group = String(_get(conn, "to", ""))
            order = Int(_get(conn, "order", -1))
            push!(get!(cg_members, group, Tuple{Int, String}[]), (order, String(_get(conn, "from", ""))))
        end
        for v in values(cg_members)
            sort!(v, by = first)
        end

        for (gname, gdata) in cg_section
            gname_s = String(gname)
            members = get(cg_members, gname_s, nothing)
            members === nothing && continue
            res_names_cg = [m[2] for m in members]

            rhs_list = _get(gdata, "rhs", [])
            rhs_list isa AbstractVector || continue
            ncuts = length(rhs_list)
            nres = length(res_names_cg)
            ncuts == 0 && continue

            intercept = zeros(ncuts)
            slopes = zeros(nres, ncuts)

            for (c, rhs_entry) in enumerate(rhs_list)
                rhs_y_raw = _get(rhs_entry, "y", Float64[])
                rhs_y = Float64.(rhs_y_raw)
                nscen = length(rhs_y)
                nscen == 0 && continue

                sum_pi_x = zeros(nscen)
                for (ri, rname) in enumerate(res_names_cg)
                    r_raw = _get(res_block, rname, nothing)
                    r_raw === nothing && continue
                    wvi = _get(r_raw, "water_value_input", nothing)
                    wvi === nothing && continue
                    wvi isa AbstractVector || continue
                    c > length(wvi) && continue

                    wvi_entry = wvi[c]
                    ys = _get(wvi_entry, "y", [])
                    xs = _get(wvi_entry, "x", [])
                    isempty(ys) && continue

                    slopes[ri, c] = sum(Float64, ys) / length(ys)

                    ns = min(nscen, length(ys), length(xs))
                    for s in 1:ns
                        sum_pi_x[s] += Float64(ys[s]) * Float64(xs[s])
                    end
                end

                intercept[c] = sum(rhs_y[s] - sum_pi_x[s] for s in 1:nscen) / nscen
            end

            push!(cut_groups, CutGroup{Float64}(
                name        = gname_s,
                res_names   = res_names_cg,
                res_indices = Int[],
                ncuts       = ncuts,
                intercept   = intercept,
                slopes      = slopes,
            ))
        end
        isempty(cut_groups) || @info "Parsed $(length(cut_groups)) cut groups: [$(join([cg.name for cg in cut_groups], ", "))]"
    end

    return (;
        plants,
        reservoirs,
        generators,
        pumps,
        tunnels,
        junctions,
        market,
        dt_schedule,
        reserve_groups,
        reserve_prices,
        cut_groups,
    )
end
