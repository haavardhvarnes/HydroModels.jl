"""
Problem type definitions — the user-facing optimization problems.

Following Gjelsvik, Mo, Haugstad (2010), there are two fundamentally
different applications of SDDP in hydropower:

1. **Single-producer scheduling** (`LongTermHydroProblem`): exogenous
   stochastic price, the producer is a price-taker. Most users.
2. **Fundamental market simulation** (`FundamentalMarketProblem`):
   endogenous price formed by market clearing across areas. Used by
   system operators and analysts.

Short-term operational scheduling is captured by `ShortTermHydroProblem`
and its stochastic variant. The link between long and short is via
`WaterValueCuts` (defined in solutions.jl).
"""

# ============================================================
# Abstract supertype
# ============================================================

"""
    AbstractOptProblem

Abstract supertype for all problem classes. Solver dispatch is on the
concrete subtype.
"""
abstract type AbstractOptProblem end

# ============================================================
# Supporting types
# ============================================================

"""
    LoadBlock{T}

A within-stage time period with characteristic load/price (e.g., peak
hours vs. off-peak). ProdRisk allows weekly stages to be subdivided
into load blocks down to hourly resolution.
"""
Base.@kwdef struct LoadBlock{T}
    name::Symbol
    duration_hours::T       # share of the stage in hours
    relative_price::T = one(T)  # multiplier on stage-average price
end

"""
    ElectricityMarket{T}

A market the producer can participate in. SHOP supports rich market
modeling (day-ahead + reserves); ProdRisk typically uses day-ahead only.

`market_type` mirrors SHOP convention:
:ENERGY, :FCR_N_UP, :FCR_N_DOWN, :FCR_D_UP, :FCR_D_DOWN,
:FRR_UP, :FRR_DOWN, :RR_UP, :RR_DOWN
"""
Base.@kwdef struct ElectricityMarket{T}
    name::Symbol
    market_type::Symbol = :ENERGY
    sale_price::Vector{T}
    buy_price::Union{Nothing, Vector{T}} = nothing
    max_sale::Union{Nothing, Vector{T}}  = nothing
    max_buy::Union{Nothing, Vector{T}}   = nothing
end

"""
    EndValueDescription{T}

End-of-horizon valuation of water in reservoirs. Options:
- `:zero` — no end value (forces full draw-down, usually wrong)
- `:fixed` — constant marginal value per Mm³
- `:cuts` — supplied set of Benders cuts (when chaining to a longer model)
"""
Base.@kwdef struct EndValueDescription{T}
    method::Symbol = :fixed
    fixed_value::T = zero(T)
    cuts = nothing
end

"""
    AbstractRiskMeasure

Risk-aversion specification for stochastic problems.
"""
abstract type AbstractRiskMeasure end

"""
    Expectation <: AbstractRiskMeasure

Risk-neutral expectation. The objective is the unweighted mean over
scenario realizations. The default for SDDP-class long-term problems.
"""
struct Expectation <: AbstractRiskMeasure end

"""
    CVaR{T}(α::T) <: AbstractRiskMeasure

Conditional Value-at-Risk at level α (e.g., 0.05 for the worst 5%).
"""
struct CVaR{T} <: AbstractRiskMeasure
    α::T
end

"""
    NestedCVaR{T}(λ::T, α::T) <: AbstractRiskMeasure

λ × Expectation + (1-λ) × CVaR_α. Common in hydro scheduling literature.
"""
struct NestedCVaR{T} <: AbstractRiskMeasure
    λ::T
    α::T
end

# ============================================================
# Long-term / medium-term scheduling (the ProdRisk problem class)
# ============================================================

"""
    LongTermHydroProblem{T, U<:UncertaintyModel{T}}

Long- or medium-term scheduling for a single price-taking producer. The
ProdRisk problem class.

# Fields
- `modules`: vector of `HydroModule`s describing the physical system
- `topology`: resolved water flow graph (built from `modules`)
- `num_stages`: typically 156 (3 years of weekly stages) for long-term
- `load_blocks`: within-stage time resolution (price periods)
- `uncertainty`: stochastic model — use `ProdRiskUncertainty` for the
  Nordic combined SDP/SDDP approach
- `stage_prices`: optional deterministic per-stage price forecast. Set
  this when the price is *not* part of the uncertainty model (e.g.
  with `StagewiseIndependent` inflow). `ProdRiskUncertainty` and
  `MarkovianUncertainty` carry their own price representations and
  should leave this as `nothing`.
- `end_value`: end-of-horizon water valuation
- `risk_measure`: risk preference (default `Expectation()`)

# Reference
Gjelsvik, Mo, Haugstad (2010), application (1).
"""
Base.@kwdef struct LongTermHydroProblem{T, U<:UncertaintyModel{T}} <: AbstractOptProblem
    modules::Vector{<:HydroModule{T}}
    topology::WaterTopology

    num_stages::Int
    load_blocks::Vector{LoadBlock{T}} = LoadBlock{T}[]

    uncertainty::U
    stage_prices::Union{Nothing, Vector{T}} = nothing
    end_value::EndValueDescription{T} = EndValueDescription{T}()
    risk_measure::AbstractRiskMeasure = Expectation()
end

# ============================================================
# Fundamental market model (the EMPS / Samnett problem class)
# ============================================================

"""
    FundamentalMarketProblem{T}

Multi-area fundamental market simulation with endogenous price formation.
Used to derive price scenarios for downstream single-producer models.

This is the Gjelsvik/Mo/Haugstad (2010) application (2). Currently a stub —
full implementation is a separate workstream.
"""
Base.@kwdef struct FundamentalMarketProblem{T} <: AbstractOptProblem
    # Placeholder — to be expanded.
    # Areas, transmission, demand, thermal generators, etc.
    placeholder::Bool = true
end

# ============================================================
# Short-term scheduling (the SHOP problem class)
# ============================================================

"""
    ShortTermHydroProblem{T}

Short-term operational scheduling for a single producer. Deterministic
with rolling horizon. Water values from a long-term model are consumed
as the end-value description.

# Fields
- `modules`: physical system (typically with `units` populated for UC)
- `topology`: water flow graph
- `horizon_stages`: number of time steps in the horizon (e.g., 168 hourly)
- `time_resolution_hours`: stage length in hours
- `price_forecast[t]`: deterministic price forecast
- `inflow_forecast[gauge_name][t]`: deterministic inflow forecast
- `water_values`: cuts from upstream long-term model (or `nothing`)
- `markets`: market models (day-ahead + reserves)
"""
Base.@kwdef struct ShortTermHydroProblem{T} <: AbstractOptProblem
    modules::Vector{<:HydroModule{T}}
    topology::WaterTopology

    horizon_stages::Int
    time_resolution_hours::T

    price_forecast::Vector{T}
    inflow_forecast::Dict{Symbol, Vector{T}}

    water_values = nothing  # ::Union{Nothing, WaterValueCuts{T}}
    markets::Vector{ElectricityMarket{T}} = ElectricityMarket{T}[]
end

# ============================================================
# SHOP-style short-term scheduling (the granular operational model)
# ============================================================

"""
    ShopShortTermProblem{T}

Short-term operational scheduling expressed in the **SHOP-style** data
model: explicit `Plant`, `Reservoir`, `Generator`, `Pump`, `Tunnel`,
`MarketSeries`, and (optionally) reserve groups and group-level Benders
cuts.

Distinct from `ShortTermHydroProblem` (which uses the coarser
`HydroModule` types and is reserved for the ProdRisk-style short-term
path). The JuMP baseline solver in `HydroModelsOpt` dispatches on
*this* type; the parser in `HydroModelsData` produces a `NamedTuple`
that maps 1:1 onto its fields.

# Convenience constructor
A no-keyword form accepts the `NamedTuple` returned by
`read_shop_yaml`:

```julia
parsed = read_shop_yaml(path)
prob = ShopShortTermProblem(parsed)
```

# Fields
- `plants::Dict{String, Plant{T}}`
- `reservoirs::Dict{String, Reservoir{T}}`
- `generators::Vector{Generator{T}}`
- `pumps::Vector{Pump{T}}`
- `tunnels::Vector{Tunnel{T}}`
- `market::MarketSeries{T}` — day-ahead price series.
- `dt_schedule::Vector{Tuple{DateTime, T}}` — `(from_time, dt_minutes)`
  breakpoints, sorted by time. Empty means a single 1-hour resolution.
- `reserve_groups::Vector{ReserveGroup{T}}` — area-level obligations.
- `reserve_prices::Dict{String, OrderedDict{DateTime, T}}` — per-product
  market sale prices (EUR/MW).
- `cut_groups::Vector{CutGroup{T}}` — group-level Benders cuts.
"""
Base.@kwdef struct ShopShortTermProblem{T} <: AbstractOptProblem
    plants::Dict{String, Plant{T}}
    reservoirs::Dict{String, Reservoir{T}}
    generators::Vector{Generator{T}}
    pumps::Vector{Pump{T}} = Pump{T}[]
    tunnels::Vector{Tunnel{T}} = Tunnel{T}[]
    junctions::Dict{String, Junction{T}} = Dict{String, Junction{T}}()
    market::MarketSeries{T} = MarketSeries{T}()
    dt_schedule::Vector{Tuple{DateTime, T}} = Tuple{DateTime, T}[]
    reserve_groups::Vector{ReserveGroup{T}} = ReserveGroup{T}[]
    reserve_prices::Dict{String, OrderedDict{DateTime, T}} =
        Dict{String, OrderedDict{DateTime, T}}()
    cut_groups::Vector{CutGroup{T}} = CutGroup{T}[]
end

"""
    ShopShortTermProblem(parsed::NamedTuple)

Convenience constructor from the `NamedTuple` returned by
`HydroModelsData.read_shop_yaml`. Field types are inferred from
`parsed.market`.
"""
function ShopShortTermProblem(parsed::NamedTuple)
    T = eltype(parsed.market)
    return ShopShortTermProblem{T}(
        plants         = parsed.plants,
        reservoirs     = parsed.reservoirs,
        generators     = parsed.generators,
        pumps          = parsed.pumps,
        tunnels        = parsed.tunnels,
        junctions      = hasproperty(parsed, :junctions) ?
                         parsed.junctions : Dict{String, Junction{T}}(),
        market         = parsed.market,
        dt_schedule    = parsed.dt_schedule,
        reserve_groups = parsed.reserve_groups,
        reserve_prices = parsed.reserve_prices,
        cut_groups     = parsed.cut_groups,
    )
end

# Helper used by the convenience constructor — pull T out of a MarketSeries.
Base.eltype(::MarketSeries{T}) where {T} = T

# ============================================================
# Stochastic short-term scheduling
# ============================================================

"""
    StochasticShortTermProblem{T, U<:UncertaintyModel{T}}

Stochastic version of short-term scheduling — Belsnes, Wolfgang, Follestad,
Aasgård (2016).

Currently a stub. To be implemented when the deterministic short-term path
is solid.
"""
Base.@kwdef struct StochasticShortTermProblem{T, U<:UncertaintyModel{T}} <: AbstractOptProblem
    modules::Vector{<:HydroModule{T}}
    topology::WaterTopology
    horizon_stages::Int
    time_resolution_hours::T
    uncertainty::U
    water_values = nothing
    markets::Vector{ElectricityMarket{T}} = ElectricityMarket{T}[]
end
