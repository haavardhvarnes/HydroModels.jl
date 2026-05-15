"""
Uncertainty model types — different stochastic representations dispatch
different solver algorithms.

Following Gjelsvik, Mo, Haugstad (2010), the Nordic SDDP tradition treats
forward simulation and backward recursion differently. This is captured
by `ProdRiskUncertainty` and is distinct from textbook SDDP which uses
the same stochastic model in both directions.
"""

# ============================================================
# Abstract type
# ============================================================

"""
    UncertaintyModel{T}

Abstract supertype for stochastic specifications. Solver dispatch is on
the concrete type to handle algorithmic differences (e.g., scenario tree
vs. Markov chain vs. ProdRisk-style hybrid).
"""
abstract type UncertaintyModel{T} end

# ============================================================
# Standard scenario tree (Brazilian SDDP convention)
# ============================================================

"""
    ScenarioNode{T}

A single node in a scenario tree, holding a realization of the random
variables at that stage.
"""
struct ScenarioNode{T}
    stage::Int
    ω::Vector{T}
end

"""
    ScenarioTree{T} <: UncertaintyModel{T}

A full scenario tree representation. Nodes are indexed in topological
order; `parent[i]` gives the parent of node `i` (0 for the root);
`children[i]` lists the children of node `i`.

This is the textbook SDDP representation. For Nordic problems consider
`ProdRiskUncertainty` instead.
"""
struct ScenarioTree{T} <: UncertaintyModel{T}
    nodes::Vector{ScenarioNode{T}}
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    probabilities::Vector{T}
end

# ============================================================
# Stagewise-independent (SDDP.jl LinearPolicyGraph)
# ============================================================

"""
    StagewiseIndependent{T} <: UncertaintyModel{T}

Per-stage independent realizations of the random variables. Suitable
for problems where serial correlation in the noise is negligible.

- `realizations[t][k]::Vector{T}` is the k-th realization at stage t
- `probabilities[t][k]` is its probability (must sum to 1 per stage)
"""
struct StagewiseIndependent{T} <: UncertaintyModel{T}
    realizations::Vector{Vector{Vector{T}}}
    probabilities::Vector{Vector{T}}
end

# ============================================================
# Markovian (SDDP.jl MarkovianPolicyGraph)
# ============================================================

"""
    MarkovianUncertainty{T} <: UncertaintyModel{T}

A Markov chain over discrete states, with possibly different state values
per stage. `transition_matrices[t]` gives the transition probabilities
from stage t to stage t+1.

This is closer in spirit to ProdRisk's price model than to textbook SDDP,
but uses a single discretization for all variables. For the full ProdRisk
hybrid (continuous historical scenarios in the forward, discretized Markov
in the backward), use `ProdRiskUncertainty`.
"""
struct MarkovianUncertainty{T} <: UncertaintyModel{T}
    state_values::Vector{Vector{T}}        # per stage
    transition_matrices::Vector{Matrix{T}} # one per stage transition
end

# ============================================================
# ProdRisk-style hybrid (the Nordic innovation)
# ============================================================

"""
    ProdRiskUncertainty{T} <: UncertaintyModel{T}

The ProdRisk-style stochastic model, following Gjelsvik, Belsnes, Haugstad
(1999) and Gjelsvik, Mo, Haugstad (2010).

Key innovation: forward simulation and backward recursion use *different*
representations.

# Forward simulation
Uses historical time series of inflow and price directly. This preserves
the empirical correlation between inflow and price scenarios — a
correlation that statistical models tend to attenuate.

# Backward recursion
Uses a PAR(1) inflow model (residuals from a periodic autoregressive
fit) for cut computation, paired with a *discretized* price representation:
price nodes per stage with empirical transition probabilities. Cuts are
stored per (stage, price_node).

# Fields
- `inflow_par_coeffs[t, gauge]`: PAR(1) autoregressive coefficient
- `inflow_residual_std[t, gauge]`: residual std at stage t
- `historical_inflow[t, scenario, gauge]`: historical inflow series
- `historical_price[t, scenario]`: historical price series
- `price_nodes[t][k]`: value of price node k at stage t
- `price_transitions[t]`: transition probability matrix for stage t → t+1
- `price_profiles[t][k, p]`: within-stage profile for node k, period p
- `n_price_nodes`: number of price nodes per stage (typically 5–10)
- `n_min_scenarios`: minimum scenarios per price node (Nmin in ProdRisk)

# Reference
- Gjelsvik, A., Belsnes, M.M., Haugstad, A. (1999) "An algorithm for
  stochastic medium-term hydrothermal scheduling under spot price
  uncertainty", PSCC.
- Gjelsvik, A., Mo, B., Haugstad, A. (2010) "Long- and medium-term
  operations planning ... based on SDDP", in Handbook of Power Systems I,
  Springer, ch. 2.
- ProdRisk price model docs:
  https://docs.prodrisk.sintef.energy/examples/price/price_model/price_model.html
"""
struct ProdRiskUncertainty{T} <: UncertaintyModel{T}
    # Backward: PAR(1) for inflow
    inflow_par_coeffs::Matrix{T}      # [stage, gauge]
    inflow_residual_std::Matrix{T}    # [stage, gauge]

    # Forward: historical scenarios (preserves inflow-price correlation)
    historical_inflow::Array{T, 3}    # [stage, scenario, gauge]
    historical_price::Matrix{T}       # [stage, scenario]

    # Backward: discretized price model (PRISMOD-style)
    price_nodes::Vector{Vector{T}}              # [stage][node]
    price_transitions::Vector{Matrix{T}}        # [stage] is K_t × K_{t+1}
    price_profiles::Vector{Matrix{T}}           # [stage][node, period]

    # Discretization parameters
    n_price_nodes::Int
    n_min_scenarios::Int
end

# Convenience: number of stages
num_stages(u::ScenarioTree) = maximum(n.stage for n in u.nodes)
num_stages(u::StagewiseIndependent) = length(u.realizations)
num_stages(u::MarkovianUncertainty) = length(u.state_values)
num_stages(u::ProdRiskUncertainty) = size(u.historical_inflow, 1)
