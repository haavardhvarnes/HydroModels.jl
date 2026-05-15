"""
    HydroModelsData

I/O layer of the HydroModels meta-package. Reads SHOP / Harmonie YAML
into `HydroModelsCore` types, and exposes a Tables.jl interface on
forcing / market / reserve outputs.

Public entry points:

- [`read_shop_yaml`](@ref) — parse a SHOP YAML file.
- `inflow_table`, `market_table`, `reserve_obligation_table` — build
  long-format tables (Tables.jl–compatible `NamedTuple`s) for downstream
  consumers (DataFrames, CSV, Makie, …).

The Tables.jl interface is also defined directly on `InflowSeries{T}`
and `MarketSeries{T}` from `HydroModelsCore`, so `Tables.columns(s)`
and `Tables.schema(s)` work without going through a builder function.

EMPS v10 reading and HDF5 results I/O are planned but not yet
implemented; see `CLAUDE.md` for the scope outline.
"""
module HydroModelsData

using Dates
using OrderedCollections
using Tables
using YAML

using HydroModelsCore

include("parsing/utils.jl")
include("parsing/shop_yaml.jl")
include("tables.jl")

export read_shop_yaml
export inflow_table, market_table, reserve_obligation_table

end # module HydroModelsData
