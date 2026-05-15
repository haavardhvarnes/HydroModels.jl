"""
Physical domain types — the user-facing description of a hydropower system.

These mirror the ProdRisk "module" concept and SHOP's level of detail.
See:
- ProdRisk documentation: https://docs.prodrisk.sintef.energy/examples/basic/hydro_module.html
- SHOP documentation:     https://docs.shop.sintef.energy/
- Skjelbred, Kong, Fosso (2019), IJEPES — for unit-level detail
"""

# ============================================================
# Reservoir types
# ============================================================

"""
    ReservoirType

Abstract supertype for reservoir kinds. Different types receive different
scheduling treatment in long-term models.
"""
abstract type ReservoirType end

"""
    RegulationReservoir <: ReservoirType

A reservoir whose end-of-stage level is a *state variable* in the long-term
scheduling problem. Contributes a dimension to the cost-to-go function and
receives Benders cut slopes.

In ProdRisk, the lower volume limit for a regulation reservoir is currently
2.0 Mm³.
"""
struct RegulationReservoir <: ReservoirType end

"""
    BufferReservoir <: ReservoirType

A small intake-pond or short-term buffer that follows a guideline curve.
Does *not* contribute to the long-term state space and does *not* receive
cuts. Operated to satisfy short-term constraints (e.g., maintaining head
to a downstream plant) only.
"""
struct BufferReservoir <: ReservoirType end

# ============================================================
# Piecewise-linear curves (forward decl — defined in utils/pwl.jl)
# ============================================================

# `PiecewiseLinear{T}` is defined in src/utils/pwl.jl.
# It carries x-y breakpoints and supports evaluation and convex-hull queries.

# ============================================================
# Penalties and soft constraints
# ============================================================

"""
    PenaltyValues{T}

Per-constraint violation penalties. Per ProdRisk convention all operational
constraints are soft; the penalty values shape solver behavior in extreme
states.
"""
Base.@kwdef struct PenaltyValues{T}
    reservoir_violation::T  = T(1_000_000)  # NOK / Mm³ typical
    discharge_violation::T  = T(100_000)
    bypass_violation::T     = T(100_000)
    spill_penalty::T        = T(0)          # often 0 — spill is allowed
end

"""
    ModuleConstraints{T}

Operational constraints for a single `HydroModule`. All time-series-valued
constraints can be `nothing` to indicate "unconstrained". All limits are
soft and may be violated subject to `penalties`.
"""
Base.@kwdef struct ModuleConstraints{T}
    max_discharge::Union{Nothing, Vector{T}} = nothing
    min_discharge::Union{Nothing, Vector{T}} = nothing
    max_bypass::Union{Nothing, T}            = nothing
    min_bypass::Union{Nothing, Vector{T}}    = nothing
    penalties::PenaltyValues{T}              = PenaltyValues{T}()
end

# ============================================================
# Turbine unit (SHOP-level detail)
# ============================================================

"""
    TurbineUnit{T}

Individual generating unit within a plant. Used by SHOP-style short-term
solvers for unit commitment. Long-term solvers typically aggregate units
into a single PQ curve at the plant level.

Fields follow Skjelbred, Kong, Fosso (2019).
"""
Base.@kwdef struct TurbineUnit{T}
    name::Symbol
    min_discharge::T
    max_discharge::T

    # Piecewise-linear hydropower production function (HPF).
    # Per Skjelbred & Kong 2019, breakpoints may be dynamically adjusted
    # per period during the SLP outer iteration.
    hpf_breakpoints::Vector{T}    # discharge values
    hpf_power::Vector{T}          # corresponding power output

    # Forbidden zones — Kong, Iliev, Skjelbred (2024)
    forbidden_zones::Vector{Tuple{T,T}} = Tuple{T,T}[]

    # Unit commitment parameters
    startup_cost::T   = zero(T)
    min_uptime::Int   = 0
    min_downtime::Int = 0
end

# ============================================================
# HydroModule — the atomic building block
# ============================================================

"""
    HydroModule{T,R<:ReservoirType}

The fundamental modeling primitive, mirroring ProdRisk's "module" concept.
A module can consist of a reservoir, a power plant (or gate, or river/
stream intake), and routing connections to downstream modules.

# Fields
- `name`: unique identifier
- `reservoir_type`: `RegulationReservoir()` or `BufferReservoir()`
- `max_vol`, `min_vol`, `initial_vol`: in Mm³ (ProdRisk convention)
- `vol_head_curve`: optional piecewise-linear volume → head relation
- `rated_discharge`: in m³/s
- `energy_factor`: kWh/m³ (base PQ relation)
- `pq_curve`: optional piecewise-linear discharge → power, at `reference_head`
- `reference_head`: head at which `pq_curve` is defined
- `units`: optional unit-level detail for SHOP-style models
- `inflow_storable_share`: fraction of inflow that can be stored (0–1)
- `inflow_series_ref`: symbolic reference to a gauging station
- `discharge_to`, `bypass_to`, `spill_to`: downstream module routing (or `nothing` for outflow)
- `constraints`: operational constraints with penalties

# Reference
ProdRisk model building blocks documentation:
https://docs.prodrisk.sintef.energy/examples/basic/hydro_module.html
"""
Base.@kwdef struct HydroModule{T,R<:ReservoirType}
    name::Symbol
    reservoir_type::R = RegulationReservoir()

    # Reservoir characteristics
    max_vol::T
    min_vol::T            = zero(T)
    initial_vol::T

    # vol_head_curve typed as Any for now; will be PiecewiseLinear{T}
    vol_head_curve = nothing

    # Plant / generation
    rated_discharge::T
    energy_factor::T
    pq_curve            = nothing
    reference_head::T   = zero(T)

    # Unit-level detail (empty for ProdRisk-class models)
    units::Vector{TurbineUnit{T}} = TurbineUnit{T}[]

    # Inflow
    inflow_storable_share::T  = one(T)
    inflow_series_ref::Symbol = :none

    # Topology
    discharge_to::Union{Nothing, Symbol} = nothing
    bypass_to::Union{Nothing, Symbol}    = nothing
    spill_to::Union{Nothing, Symbol}     = nothing

    # Constraints
    constraints::ModuleConstraints{T} = ModuleConstraints{T}()
end

# ============================================================
# Topology
# ============================================================

"""
    WaterTopology

Resolved water-flow graph derived from a collection of `HydroModule`s.
Constructed by `build_topology(modules)` in `src/problems/topology.jl`.

Holds adjacency information, a topological ordering of modules (upstream
first), and any cycle warnings.
"""
struct WaterTopology
    modules_by_name::Dict{Symbol, Int}
    discharge_edges::Vector{Tuple{Int, Int}}
    bypass_edges::Vector{Tuple{Int, Int}}
    spill_edges::Vector{Tuple{Int, Int}}
    topological_order::Vector{Int}
end

# Convenience pretty-printing
function Base.show(io::IO, ::MIME"text/plain", m::HydroModule)
    print(io, "HydroModule(:", m.name, ", ", m.reservoir_type, ")")
    print(io, "\n  vol: ", m.min_vol, "–", m.max_vol, " Mm³")
    print(io, " (init ", m.initial_vol, ")")
    print(io, "\n  discharge: ", m.rated_discharge, " m³/s @ η=", m.energy_factor)
    if !isnothing(m.discharge_to)
        print(io, "\n  ➔ ", m.discharge_to)
    end
end
