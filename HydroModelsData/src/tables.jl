"""
Tables.jl interface for `HydroModelsCore` time-series and reserve types.

Implements the column-access interface on:

- `InflowSeries{T}` → columns `(:time, :inflow)`.
- `MarketSeries{T}` → columns `(:time, :price)`.

Plus convenience functions for *wide* tables that combine multiple
series:

- `inflow_table(reservoirs)` → long table `(time, reservoir, inflow)`
  built from a `Dict{String, Reservoir{T}}`.
- `market_table(market)` → equivalent of
  `Tables.columntable(market)`.
- `reserve_obligation_table(groups)` → long table
  `(time, group, product, obligation)`.

Adding `DataFrames` to the user environment lets them call
`DataFrame(inflow_table(reservoirs))` directly — no DataFrames hard
dependency in this package.
"""

# ============================================================
# Tables.jl interface on InflowSeries
# ============================================================

Tables.istable(::Type{<:InflowSeries}) = true
Tables.columnaccess(::Type{<:InflowSeries}) = true

function Tables.columns(s::InflowSeries{T}) where {T}
    return (
        time = collect(keys(s.by_time)),
        inflow = collect(values(s.by_time)),
    )
end

function Tables.schema(::InflowSeries{T}) where {T}
    return Tables.Schema((:time, :inflow), (DateTime, T))
end

# ============================================================
# Tables.jl interface on MarketSeries
# ============================================================

Tables.istable(::Type{<:MarketSeries}) = true
Tables.columnaccess(::Type{<:MarketSeries}) = true

function Tables.columns(s::MarketSeries{T}) where {T}
    return (
        time = collect(keys(s.by_time)),
        price = collect(values(s.by_time)),
    )
end

function Tables.schema(::MarketSeries{T}) where {T}
    return Tables.Schema((:time, :price), (DateTime, T))
end

# ============================================================
# Convenience long-table builders
# ============================================================

"""
    inflow_table(reservoirs)

Build a long table `(time::Vector{DateTime}, reservoir::Vector{String},
inflow::Vector{T})` from a `Dict{String, Reservoir{T}}`. Reservoirs are
emitted in iteration order; missing inflow series contribute zero rows.
Returns a `NamedTuple` of columns that satisfies `Tables.istable`.
"""
function inflow_table(reservoirs::AbstractDict{String, <:Reservoir{T}}) where {T}
    times = DateTime[]
    names = String[]
    vals = T[]
    for (rname, r) in reservoirs
        for (t, v) in r.inflow.by_time
            push!(times, t)
            push!(names, rname)
            push!(vals, v)
        end
    end
    return (time = times, reservoir = names, inflow = vals)
end

"""
    market_table(market)

Materialize a `MarketSeries{T}` into its two-column representation.
Identical to `Tables.columntable(market)`.
"""
function market_table(market::MarketSeries{T}) where {T}
    return Tables.columntable(market)
end

"""
    reserve_obligation_table(groups)

Build a long table `(time, group, product, obligation)` from a
`Vector{ReserveGroup{T}}`.
"""
function reserve_obligation_table(groups::AbstractVector{<:ReserveGroup{T}}) where {T}
    times = DateTime[]
    gnames = String[]
    prods = String[]
    vals = T[]
    for g in groups
        for (t, v) in g.obligation
            push!(times, t)
            push!(gnames, g.name)
            push!(prods, g.product)
            push!(vals, v)
        end
    end
    return (time = times, group = gnames, product = prods, obligation = vals)
end
