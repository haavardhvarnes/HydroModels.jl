"""
YAML-parsing helpers used by `read_shop_yaml`. Ported from
`HydroModels_depr/src/utils.jl` with adaptations for `HydroModelsCore`'s
validated `PiecewiseLinear` type (which requires `length(x) >= 2` and
sorted abscissae, unlike depr's bare `PWL` struct).

The helpers are intentionally module-internal — they take `Any`-shaped
YAML data and produce typed Julia values. They are not part of the
public API.
"""

const _TF = DateFormat("yyyy-mm-dd HH:MM:SS")

_has(d, k::AbstractString) = haskey(d, k) || haskey(d, Symbol(k))
_get(d, k::AbstractString, default = nothing) =
    haskey(d, k) ? d[k] :
    (haskey(d, Symbol(k)) ? d[Symbol(k)] : default)

_parse_dt(x) = x isa DateTime ? x : tryparse(DateTime, String(x), _TF)

_as_float(x, default = 0.0) = x === nothing ? default : Float64(x)

function _scalar_or_first(x, default = 0.0)
    if x === nothing
        return default
    elseif x isa Number
        return Float64(x)
    elseif x isa AbstractDict
        isempty(x) && return default
        return Float64(first(values(x)))
    else
        try
            return Float64(x)
        catch
            return default
        end
    end
end

_as_pair(v) =
    v isa Tuple && length(v) == 2 ? (string(v[1]), string(v[2])) :
    v isa AbstractVector && length(v) == 2 ? (string(v[1]), string(v[2])) :
    v isa NamedTuple && all(haskey(v, x) for x in (:from, :to)) ?
        (string(v[:from]), string(v[:to])) :
    ("", "")

_normalize_map(d::AbstractDict) =
    Dict{String, Tuple{String, String}}(string(k) => _as_pair(v) for (k, v) in d)

function _ordered_datetime_float(raw)
    out = OrderedDict{DateTime, Float64}()
    if raw isa AbstractDict
        for (k, v) in raw
            t = _parse_dt(k)
            t === nothing && continue
            out[t] = Float64(v)
        end
    end
    return out
end

function _ordered_datetime_int(raw)
    out = OrderedDict{DateTime, Int}()
    if raw isa AbstractDict
        for (k, v) in raw
            t = _parse_dt(k)
            t === nothing && continue
            out[t] = Int(v)
        end
    end
    return out
end

"""
    _try_piecewise(x, y)

Build a `PiecewiseLinear` from raw `x`/`y` vectors, returning `nothing`
when the data cannot satisfy the type's invariants (`length >= 2`,
sorted, equal-length). Degenerate single-point curves are silently
dropped — the caller decides whether to warn.
"""
function _try_piecewise(x::AbstractVector, y::AbstractVector)
    length(x) >= 2 || return nothing
    length(x) == length(y) || return nothing
    xv = Float64.(x)
    yv = Float64.(y)
    issorted(xv) || return nothing
    return PiecewiseLinear(xv, yv)
end

"""
    _ensure_origin(curve)

If `curve.x[1] != 0`, prepend a `(0, 0)` breakpoint so curve integrations
that start at zero discharge remain well-defined. The depr signature
took a bare `PWL`; here the input is `PiecewiseLinear{Float64}` and the
returned curve is guaranteed to satisfy the parent type's invariants
(sortedness still holds because we prepend `0 < x[1]`).
"""
function _ensure_origin(curve::PiecewiseLinear{Float64})
    iszero(curve.x[1]) && return curve
    curve.x[1] > 0 || return curve  # already includes negative origin or starts at 0
    return PiecewiseLinear(vcat(0.0, curve.x), vcat(0.0, curve.y))
end

"""Linear interpolation of storage volume at absolute head `h`."""
function _vol_from_head(V::Vector{Float64}, H::Vector{Float64}, h::Float64)
    n = length(V)
    if n == 0 || length(H) != n
        return NaN
    end
    for k in 1:(n - 1)
        Hk, Hk1 = H[k], H[k + 1]
        if (Hk <= h <= Hk1) || (Hk1 <= h <= Hk)
            θ = (h - Hk) / (Hk1 - Hk)
            return V[k] + θ * (V[k + 1] - V[k])
        end
    end
    h <= minimum(H) && return V[argmin(H)]
    h >= maximum(H) && return V[argmax(H)]
    return NaN
end

"""Extract the plant name from a unit name, split on `_|_` (SHOP convention)."""
_unit_plant(name::AbstractString) = String(split(String(name), "_|_")[1])

function _infer_generator_connection(
        plant::AbstractString,
        incoming_by_plant_all::AbstractDict{String, Vector{String}},
        outgoing_by_plant::AbstractDict{String, String},
    )
    p = String(plant)
    # B1.6: `from_res` is a vector of all upstream reservoirs reachable
    # through tunnel chains (multi-source plants have ≥ 2 entries; the
    # vector is sorted by reservoir capacity descending, lex tie-break).
    from_res = copy(get(incoming_by_plant_all, p, String[]))
    to_res = get(outgoing_by_plant, p, "")
    return from_res, to_res
end

function _infer_pump_connection(
        plant::AbstractString,
        incoming_by_plant::AbstractDict{String, String},
        outgoing_by_plant::AbstractDict{String, String},
    )
    p = String(plant)
    # Pumps at reversible plants flow opposite to generators: pump intake
    # is the generator's tailrace, discharge is the generator's headrace.
    # We keep pumps single-source for B1.6 — no real SHOP case in NO5 has
    # a multi-source pump, and the schema-symmetric vector form would
    # complicate the LP without operational benefit.
    from_str = get(outgoing_by_plant, p, "")
    from_res = isempty(from_str) ? String[] : String[from_str]
    to_res = get(incoming_by_plant, p, "")
    return from_res, to_res
end

"""
    _bfs_all_reservoirs(start, graph, targets; capacity = empty)

Like `_bfs_nearest`, but returns **every** reachable reservoir in
`targets`. With `capacity` given, the result is sorted by capacity
descending with lexicographic tie-break (so the first entry is the
dominant intake — useful as the default head reference). Without
`capacity`, the result is sorted lexicographically.

Returns `String[]` if no target is reachable.
"""
function _bfs_all_reservoirs(
        start::String,
        graph::Dict{String, Vector{String}},
        targets::AbstractSet{String};
        capacity::Dict{String, Float64} = Dict{String, Float64}(),
    )
    visited = Set{String}([start])
    queue = String[start]
    found = String[]
    while !isempty(queue)
        node = popfirst!(queue)
        for nb in get(graph, node, String[])
            nb in visited && continue
            push!(visited, nb)
            if nb in targets
                push!(found, nb)
                # Do not continue BFS through a reservoir — reservoirs
                # are mass-balance endpoints, same as in `_bfs_nearest`.
                continue
            end
            push!(queue, nb)
        end
    end
    isempty(found) && return String[]
    if isempty(capacity)
        sort!(found)
    else
        sort!(found; by = r -> (-get(capacity, r, 0.0), r))
    end
    return found
end

"""
    _bfs_nearest(start, graph, targets; capacity = empty)

BFS on `graph` (adjacency dict) starting from `start`.

When `capacity` is empty: returns the first reservoir found (nearest).
When `capacity` is provided: explores all reachable reservoirs and
returns the one with the highest capacity — avoids picking a tiny
forebay over a large headwater when both are reachable via KP
waypoints. Returns `""` if no target is reachable.
"""
function _bfs_nearest(
        start::String,
        graph::Dict{String, Vector{String}},
        targets::AbstractSet{String};
        capacity::Dict{String, Float64} = Dict{String, Float64}(),
    )
    use_capacity = !isempty(capacity)
    visited = Set{String}([start])
    queue = String[start]
    found = String[]
    while !isempty(queue)
        node = popfirst!(queue)
        for nb in get(graph, node, String[])
            nb in visited && continue
            push!(visited, nb)
            if nb in targets
                use_capacity || return nb
                push!(found, nb)
                # Do NOT continue BFS through reservoirs — they are mass-
                # balance endpoints. Only continue through intermediate
                # nodes (KP waypoints, etc.).
                continue
            end
            push!(queue, nb)
        end
    end
    isempty(found) && return ""
    return argmax(r -> get(capacity, r, 0.0), found)
end
