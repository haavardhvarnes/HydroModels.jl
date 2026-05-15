"""
Reserve-market types — capacity obligations on individual units and
area-level groups, covering the eight canonical Nordic products:
`FCR_N_UP`, `FCR_N_DOWN`, `FCR_D_UP`, `FCR_D_DOWN`,
`FRR_UP`, `FRR_DOWN`, `RR_UP`, `RR_DOWN`.

These are the *data containers*. Reserve constraints and penalty
wiring live in `HydroModelsOpt` (milestone A5 in the phased plan).
"""

"""
    ReserveSpec{T}

Reserve capacity specification for one product on one generator unit.

# Fields
- `product::String` — canonical product name (e.g. `"FCR_N_UP"`, `"FRR_DOWN"`).
- `pmin::T`, `pmax::T` — minimum / maximum headroom this unit must / can
  provide (MW).
- `schedule::OrderedDict{DateTime,T}` — pre-agreed delivery schedule
  (MW). May be empty when only opportunity-based participation is modelled.
- `penalty_cost::OrderedDict{DateTime,T}` — unit-level penalty for shortfall
  (EUR/MW). May be empty if shortfall is forbidden or priced at the
  group level only.
"""
Base.@kwdef struct ReserveSpec{T}
    product::String
    pmin::T
    pmax::T
    schedule::OrderedDict{DateTime, T} = OrderedDict{DateTime, T}()
    penalty_cost::OrderedDict{DateTime, T} = OrderedDict{DateTime, T}()
end

"""
    ReserveGroup{T}

Area-level reserve obligation group (e.g. `"NO5_FCR_N_UP"`) covering one
canonical product. The total `obligation` MW must be collectively
supplied by all units in the group; `penalty_cost` is charged per MW of
shortfall.

# Fields
- `name::String` — group identifier from the SHOP `reserve_group` block.
- `product::String` — canonical product name this group covers.
- `obligation::OrderedDict{DateTime,T}` — total MW per timestamp.
- `penalty_cost::OrderedDict{DateTime,T}` — EUR per MW·h of shortfall.
"""
Base.@kwdef struct ReserveGroup{T}
    name::String
    product::String
    obligation::OrderedDict{DateTime, T} = OrderedDict{DateTime, T}()
    penalty_cost::OrderedDict{DateTime, T} = OrderedDict{DateTime, T}()
end
