# `HydroModelsData` — API reference

The data layer reads SHOP / Harmonie YAML into `HydroModelsCore`
types and exposes a [Tables.jl](https://github.com/JuliaData/Tables.jl)
interface on forcing / market / reserve outputs.

## SHOP YAML parser

```@docs
read_shop_yaml
```

The parser returns a `NamedTuple` with fields:

| Field | Type | Description |
|---|---|---|
| `plants` | `Vector{Plant{Float64}}` | one per `model.plant.*` block |
| `reservoirs` | `Dict{String, Reservoir{Float64}}` | one per `model.reservoir.*` block |
| `generators` | `Vector{Generator{Float64}}` | one per `model.generator.*` block |
| `pumps` | `Vector{Pump{Float64}}` | one per `model.pump.*` block |
| `tunnels` | `Vector{Tunnel{Float64}}` | virtual edges from `model.tunnel.*` + river-resolved edges |
| `junctions` | `Dict{String, Junction{Float64}}` | declared `model.river.<name>` KP blocks |
| `reserve_groups` | `Vector{ReserveGroup{Float64}}` | one per `model.reserve_group.*` block |
| `market_series` | `MarketSeries{Float64}` | day-ahead price series |
| `dt_schedule` | `OrderedDict{DateTime, Period}` | variable timestep map |

Passing this `NamedTuple` to the
[`ShopShortTermProblem(...)`](@ref ShopShortTermProblem) constructor
produces a Core-level problem ready for `solve(...)`.

## Long-format Tables.jl builders

```@docs
inflow_table
market_table
reserve_obligation_table
```

These return `NamedTuple` of `Vector` columns. Convert to a
`DataFrame` with `using DataFrames; DataFrame(t)`.

The Tables.jl interface is also defined directly on
`InflowSeries{T}` and `MarketSeries{T}` from `HydroModelsCore`, so
`Tables.columns(s)` and `Tables.schema(s)` work without going
through a builder function.
