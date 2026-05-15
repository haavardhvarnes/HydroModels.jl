"""
Solution types — what solvers return.

The most important type here is `WaterValueCuts` — the bridge between
long-term and short-term scheduling.
"""

# ============================================================
# Generic solution wrapper
# ============================================================

"""
    Solution{T}

Generic solution returned by any solver. The exact contents of `primal`
and `dual` depend on the problem class.

- `status` follows MathOptInterface termination codes
- `objective` is the best known objective value
- `bound` is the best known dual bound (for stochastic / decomposition methods)
- `iterations` is the number of solver iterations
- `solve_time` is wall-clock seconds
"""
Base.@kwdef struct Solution{T}
    primal::Any        = nothing
    dual::Any          = nothing
    objective::T
    bound::T           = typemin(T)
    status::MOI.TerminationStatusCode = MOI.OTHER_LIMIT
    iterations::Int    = 0
    solve_time::Float64 = 0.0
    metadata::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

# ============================================================
# Benders cuts — the long-term to short-term bridge
# ============================================================

"""
    BendersCut{T}

A single Benders cut representing a tangent hyperplane to the cost-to-go
function. The cut is

    α(x) ≥ intercept + Σᵢ slopes[i] × x[i]

(for a minimization formulation, this is a lower bound; for maximization
the direction is reversed).

In ProdRisk-style models, cuts are associated with a particular `(stage,
price_node)` pair.
"""
Base.@kwdef struct BendersCut{T}
    intercept::T
    slopes::Vector{T}             # one entry per regulation reservoir
    stage::Int
    price_node::Int = 1           # 1 if no price discretization
end

"""
    WaterValueCuts{T}

Collection of Benders cuts representing the future-cost function. This is
the canonical interface between long-term and short-term scheduling in the
Nordic toolchain.

# Indexing convention
`cuts[stage][price_node]` is a `Vector{BendersCut{T}}` — the set of cuts
for that stage and price node. The number of cuts grows with SDDP
iterations.

`reservoir_names` records the *order* of reservoir state variables that
the cut slopes correspond to. The downstream consumer (SHOP-style model)
must match this ordering.

`price_node_values[stage][k]` gives the value of price node `k` at `stage`
(in NOK/MWh or equivalent).
"""
Base.@kwdef struct WaterValueCuts{T}
    cuts::Vector{Vector{Vector{BendersCut{T}}}}     # [stage][price_node][cut]
    reservoir_names::Vector{Symbol}
    price_node_values::Vector{Vector{T}}            # [stage][node]
end

# Convenience
num_cuts(wvc::WaterValueCuts, stage::Int, price_node::Int) =
    length(wvc.cuts[stage][price_node])

num_cuts(wvc::WaterValueCuts) =
    sum(length(node_cuts) for stage_cuts in wvc.cuts for node_cuts in stage_cuts)

# ============================================================
# SHOP-style cut representations (per-reservoir scalar + group)
# ============================================================

"""
    ReservoirWaterValue{T}

Per-reservoir scalar Benders cuts on end-of-horizon storage, as produced
by SDDP and consumed by a SHOP-style short-term LP.

Each cut `c` defines `V(S) ≥ ref[c] + slope[c] * S` (for a minimization
formulation); the water-value function is the upper envelope of these
cuts. `S` is end-of-horizon storage in Mm³ and slopes are in EUR/Mm³.

This is the **single-reservoir scalar** form, distinct from `WaterValueCuts`
which represents the full stage × price-node × multi-reservoir cut
collection. The two coexist: `Reservoir.water_value` holds a
`ReservoirWaterValue{T}`; the long-term solver's output is a
`WaterValueCuts{T}` which is projected onto each reservoir to populate
`Reservoir.water_value` for the short-term toolchain.
"""
struct ReservoirWaterValue{T}
    ref::Vector{T}      # intercepts (EUR), one per cut
    slope::Vector{T}    # slopes (EUR/Mm³), one per cut
end

"""
    CutGroup{T}

Group-level Benders cuts coupling multiple reservoirs, as produced by
SDDP's `cut_group` output in the SHOP YAML.

Each cut `c` defines `W_G ≤ intercept[c] + Σᵢ slopes[i,c] × S_end[i]`
where `intercept[c]` is the mean over scenarios of
`rhs_y[s] - Σᵢ π_csi × x_ci[s]`.

# Fields
- `name::String` — group identifier from the YAML `cut_group` block.
- `res_names::Vector{String}` — reservoir names in SHOP `order`.
- `res_indices::Vector{Int}` — indices into the host problem's reservoir
  vector (filled by the LP builder at construction time; may be empty
  on initial load).
- `ncuts::Int` — number of cuts (e.g. 7).
- `intercept::Vector{T}` — `[ncuts]`, mean intercept per cut.
- `slopes::Matrix{T}` — `[n_reservoirs, ncuts]`, marginal values
  (EUR/Mm³).
"""
Base.@kwdef struct CutGroup{T}
    name::String
    res_names::Vector{String}
    res_indices::Vector{Int} = Int[]
    ncuts::Int
    intercept::Vector{T}
    slopes::Matrix{T}
end
