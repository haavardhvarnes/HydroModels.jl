"""
SHOP-style operational types — `Plant`, `Reservoir`, `Generator`,
`Pump`, `Tunnel`. These model the physical system at the granularity
needed for short-term operational scheduling (the SHOP algorithm of
Skjelbred, Kong, Fosso, et al.).

Distinct from the ProdRisk-style `HydroModule` / `TurbineUnit` types
in `physical.jl`, which model the system at the coarser granularity
used by long-term SDDP scheduling. Both type families coexist and a
single problem instance uses one or the other depending on which
scheduling horizon is being modelled.

All types are parametric on `T` (the numeric scalar type) to support
`Float64`, `Float32`, autodiff `Dual`, `BigFloat`, etc. Where time
series appear, they use `OrderedDict{DateTime,...}` to preserve
insertion order through the parsing → solving → reporting pipeline.

References:
- Skjelbred, Kong, Fosso (2019), IJEPES — SHOP dynamic MILP formulation.
- Kong, Skjelbred, Fosso (2020), EPSR — unit-based STHS overview.
- SINTEF SHOP docs: <https://docs.shop.sintef.energy>
"""

"""
    Plant{T}

Plant-level metadata used for outlet inference, head-loss accounting,
and time-varying capacity caps. A plant aggregates one or more
generator and/or pump units that share a penstock and outlet.

# Fields
- `name::String` — plant identifier.
- `outlet_line::T` — tailrace elevation (m above reference).
- `main_loss::Vector{T}` — main waterway head-loss coefficients.
- `penstock_loss::Vector{T}` — per-penstock head-loss coefficients.
- `maintenance_flag::OrderedDict{DateTime,Int}` — `0 = available`,
  `1 = on maintenance`. Integer-valued regardless of `T`.
- `max_p_constr::OrderedDict{DateTime,T}` — plant-level power cap (MW).
"""
Base.@kwdef struct Plant{T}
    name::String
    outlet_line::T
    main_loss::Vector{T} = T[]
    penstock_loss::Vector{T} = T[]
    maintenance_flag::OrderedDict{DateTime, Int} = OrderedDict{DateTime, Int}()
    max_p_constr::OrderedDict{DateTime, T} = OrderedDict{DateTime, T}()
end

"""
    Reservoir{T}

Reservoir with a storage-volume → absolute-head relation, storage
bounds, inflow time series, and an optional end-of-horizon water-value
description.

# Fields
- `name::String` — reservoir identifier.
- `vol_breaks::Vector{T}`, `head_breaks::Vector{T}` — paired breakpoints
  of the V→H curve. `length(vol_breaks) == length(head_breaks)`.
- `s_min::T`, `s_max::T`, `s0::T` — storage bounds (Mm³) and initial
  storage.
- `inflow::InflowSeries{T}` — time-indexed inflow.
- `water_value::Union{Nothing, ReservoirWaterValue{T}}` — per-reservoir
  end-of-horizon Benders cuts; `nothing` when the YAML omits them.
- `spill_to_reservoir::Union{Nothing, String}` — destination reservoir
  for surplus spill (dam overtopping). `nothing` means terminal: spill
  exits the modelled system. Inferred by `read_shop_yaml` from the
  generator topology when not explicitly set.

Note: this is the SHOP-style data struct. It is distinct from the
ProdRisk-style type tags `RegulationReservoir` / `BufferReservoir`
defined in `physical.jl`, which live as type parameters on
`HydroModule`.
"""
Base.@kwdef struct Reservoir{T}
    name::String
    vol_breaks::Vector{T}
    head_breaks::Vector{T}
    s_min::T
    s_max::T
    s0::T
    inflow::InflowSeries{T} = InflowSeries{T}()
    water_value::Union{Nothing, ReservoirWaterValue{T}} = nothing
    spill_to_reservoir::Union{Nothing, String} = nothing
end

"""
    Generator{T}

Generator (turbine) unit with hydraulic curves, efficiency curve,
time-varying operational limits, and reserve-product capacity
specifications.

# Fields
- `name::String` — full unit name from the SHOP YAML.
- `plant::String` — owning plant, inferred from `name`.
- `from_res::Vector{String}` — upstream reservoir name(s). One entry
  for single-source plants, multiple entries (sorted by capacity desc,
  lex tie-break) for plants whose intake tunnels fan in at a KP
  junction (e.g. Hemsil2 in NO5 has `["Eikrebekken", "Logga", "Ruståni"]`).
- `head_reference::Union{Nothing, String}` — reservoir whose head
  defines the head-difference in `Hg`. Defaults to `from_res[1]` when
  `nothing`. For multi-source plants the physical intake head is at
  the merging KP; the LP approximates it by the dominant source.
- `to_res::String` — downstream reservoir name (singular; multi-
  destination plants don't appear in real SHOP cases).
- `curves::Vector{Tuple{T, PiecewiseLinear{T}}}` — hydraulic curves as
  `(head_reference, PiecewiseLinear(Q, P_mech))`.
- `qmin::T`, `qmax::T` — discharge limits (m³/s).
- `pmin::T`, `pmax::T` — mechanical power limits (MW).
- `p_nom::T` — nameplate capacity (MW).
- `penstock::Int` — penstock group index (units sharing a penstock have
  the same value).
- `gen_eff::Union{Nothing, PiecewiseLinear{T}}` — generator electrical
  efficiency curve, where `x = mechanical MW` and `y = efficiency ∈ [0, 1]`.
- `max_p_constr::OrderedDict{DateTime,T}` — time-varying capacity cap.
- `maintenance_flag::OrderedDict{DateTime,Int}` — `0` / `1`.
- `source_qmax::Dict{String, Float64}` — per-source tunnel-chain
  capacity cap (m³/s), keyed by source-reservoir name. Empty dict
  means "use overall `qmax` for every source" (single-source default).
- `startcost::T` — startup cost (EUR).
- `initial_state::Int` — `0 = off`, `1 = on` at horizon start.
- `reserves::Vector{ReserveSpec{T}}` — reserve-product participations.

Construction is keyword-only. Pass `from_res = ["upper"]` (a one-
element vector) for single-source plants. The parser populates
the vector form directly.
"""
Base.@kwdef struct Generator{T}
    name::String
    plant::String
    from_res::Vector{String}
    head_reference::Union{Nothing, String} = nothing
    to_res::String
    curves::Vector{Tuple{T, PiecewiseLinear{T}}}
    qmin::T
    qmax::T
    pmin::T
    pmax::T
    p_nom::T
    penstock::Int = 0
    gen_eff::Union{Nothing, PiecewiseLinear{T}} = nothing
    max_p_constr::OrderedDict{DateTime, T} = OrderedDict{DateTime, T}()
    maintenance_flag::OrderedDict{DateTime, Int} = OrderedDict{DateTime, Int}()
    source_qmax::Dict{String, Float64} = Dict{String, Float64}()
    startcost::T = zero(T)
    initial_state::Int = 0
    reserves::Vector{ReserveSpec{T}} = ReserveSpec{T}[]
end

"""
    Pump{T}

Pump unit. Intentionally leaner than `Generator` — captures only what
the current LP builder uses: hydraulic consumption curves, reservoir
connections, q/p limits, maintenance flags. Add `pump_eff` /
`max_p_constr` here when the LP builder grows to use them.

# Fields
See `Generator` for the shared fields. `curves` here are
`(href, PiecewiseLinear(Q, E_pump))` where `E_pump` is electrical
consumption (MW).
"""
Base.@kwdef struct Pump{T}
    name::String
    plant::String
    from_res::Vector{String}
    head_reference::Union{Nothing, String} = nothing
    to_res::String
    curves::Vector{Tuple{T, PiecewiseLinear{T}}}
    qmin::T
    qmax::T
    pmin::T
    pmax::T
    p_nom::T
    penstock::Int = 0
    maintenance_flag::OrderedDict{DateTime, Int} = OrderedDict{DateTime, Int}()
    source_qmax::Dict{String, Float64} = Dict{String, Float64}()
    startcost::T = zero(T)
    initial_state::Int = 0
end

"""
    Tunnel{T}

Passive water-transfer connection between two nodes (reservoirs or
junctions). No active control, no energy production.

# Fields
- `name::String`
- `from_node::String`, `to_node::String`
- `qmax::T` — capacity (m³/s).
- `start_height::T`, `end_height::T` — elevations (m).
- `loss_factor::T` — head-loss coefficient.
"""
Base.@kwdef struct Tunnel{T}
    name::String
    from_node::String
    to_node::String
    qmax::T
    start_height::T
    end_height::T
    loss_factor::T = zero(T)
end

"""
    Junction{T}

Topology junction point ("knutepunkt" / KP in Norwegian) with optional
regulatory flow constraints. Declared explicitly under
`model.river.<name>` in SHOP YAML when the junction carries
`max_flow` (physical river-flow cap) and/or `flow` (regulatory
minimum release — e.g. an environmental obligation at a system
boundary). Junctions without these fields stay implicit pass-through
nodes handled by the parser's virtual-tunnel resolution.

# Fields
- `name::String` — junction identifier.
- `upstream_elevation::T` — head reference at the junction. Some
  SHOP YAMLs use a sentinel value (`0.001`) for terminal junctions
  whose elevation isn't physically meaningful; treat as informational.
- `max_flow::OrderedDict{DateTime, T}` — time-keyed upper bound on
  total flow through the junction. Hard constraint in the LP.
  Sparse; the LP densifies with `Inf` default outside declared
  timestamps.
- `flow::OrderedDict{DateTime, T}` — time-keyed **regulatory minimum**
  release at the junction. Implemented as a **soft constraint with
  large penalty** in the LP (slack variable `Rj_min` with penalty
  `junc_min_penalty` ≈ 1e6 EUR per m³/s per hour). Sparse; the LP
  densifies with `0` default outside declared timestamps.
"""
Base.@kwdef struct Junction{T}
    name::String
    upstream_elevation::T = zero(T)
    max_flow::OrderedDict{DateTime, T} = OrderedDict{DateTime, T}()
    flow::OrderedDict{DateTime, T}     = OrderedDict{DateTime, T}()
end
