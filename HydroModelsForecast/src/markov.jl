"""
Markov price-node model for stochastic price representation in
SDDP-backward recursion (PRISMOD-style).

For each stage `t`, prices are discretized into `K` "nodes" (clusters);
the chain is described by a transition matrix `P_t` whose entry
`P_t[i, j]` is the probability of moving from node `i` at stage `t` to
node `j` at stage `t+1`.

The fitting routine here is intentionally simple: quantile-based
discretization per stage plus empirical transition counts. The full
ProdRisk algorithm additionally adjusts transitions via a QP so that
the conditional mean one stage ahead exactly matches the input series;
that polish is a follow-up.

References:
- ProdRisk price model:
  <https://docs.prodrisk.sintef.energy/examples/price/price_model/>
- Gjelsvik, Belsnes, Haugstad (1999), PSCC.
"""

"""
    MarkovPriceModel{T}

Fitted Markov price-node model. Field semantics:

- `node_values[t][k]`         — value of node `k` at stage `t` (size `K`)
- `transition_matrices[t]`    — `K × K` row-stochastic matrix giving
  P(node_{t+1} | node_t). One per stage transition, so length
  `num_stages - 1`.
- `initial_probs[k]`          — marginal probability of node `k` at stage 1.
"""
struct MarkovPriceModel{T}
    node_values::Vector{Vector{T}}
    transition_matrices::Vector{Matrix{T}}
    initial_probs::Vector{T}
end

num_stages(m::MarkovPriceModel) = length(m.node_values)
num_nodes(m::MarkovPriceModel) = isempty(m.node_values) ? 0 : length(m.node_values[1])

"""
    fit(::Type{MarkovPriceModel}, prices::AbstractMatrix; K::Int)

Fit a `K`-node Markov price model from historical prices laid out as
`prices[scenario, stage]`. Each row is one historical year-long
trajectory.

The fitting steps for each stage:

1. **Discretize**: sort the per-stage prices and partition into `K`
   equal-frequency bins. The node value is the bin's mean.
2. **Assign**: each scenario at each stage is assigned to its nearest
   node.
3. **Count transitions**: between consecutive stages, count node-to-
   node transitions. Normalize rows to row-stochastic. Empty rows
   (i.e., a node that's never visited at stage `t`) get a uniform
   row.
"""
function fit(::Type{MarkovPriceModel}, prices::AbstractMatrix; K::Int)
    n_scen, n_stages = size(prices)
    n_scen >= K || throw(ArgumentError(
        "MarkovPriceModel fit needs at least K=$(K) scenarios, got $(n_scen)"))
    K >= 1 || throw(ArgumentError("K must be >= 1, got $(K)"))

    T = eltype(prices)
    node_values = Vector{Vector{T}}(undef, n_stages)
    assignments = Matrix{Int}(undef, n_scen, n_stages)

    # Per-stage discretization via equal-frequency bins on sorted prices
    for t in 1:n_stages
        col = prices[:, t]
        perm = sortperm(col)
        sorted_col = col[perm]
        bin_size = n_scen / K
        bin_means = Vector{T}(undef, K)
        for k in 1:K
            lo = Int(floor((k - 1) * bin_size)) + 1
            hi = k == K ? n_scen : Int(floor(k * bin_size))
            bin_means[k] = Statistics.mean(sorted_col[lo:hi])
        end
        node_values[t] = bin_means

        # Assign each scenario to the nearest node
        for s in 1:n_scen
            v = col[s]
            # nearest node by absolute distance
            best_k = 1
            best_d = abs(v - bin_means[1])
            for k in 2:K
                d = abs(v - bin_means[k])
                if d < best_d
                    best_d = d
                    best_k = k
                end
            end
            assignments[s, t] = best_k
        end
    end

    # Initial node-marginal probabilities (stage 1)
    counts1 = zeros(T, K)
    for s in 1:n_scen
        counts1[assignments[s, 1]] += one(T)
    end
    initial_probs = counts1 ./ T(n_scen)

    # Empirical transition matrices
    transitions = Vector{Matrix{T}}(undef, n_stages - 1)
    for t in 1:(n_stages - 1)
        M = zeros(T, K, K)
        for s in 1:n_scen
            M[assignments[s, t], assignments[s, t + 1]] += one(T)
        end
        # Row-normalize; uniform if a row has zero mass
        for i in 1:K
            row_sum = sum(M[i, :])
            if row_sum > 0
                M[i, :] ./= row_sum
            else
                M[i, :] .= one(T) / K
            end
        end
        transitions[t] = M
    end

    return MarkovPriceModel{T}(node_values, transitions, initial_probs)
end

"""
    sample(rng, model::MarkovPriceModel, n::Int)

Draw `n` Markov-chain trajectories of length `num_stages(model)`.
Returns `Vector{Vector{T}}` of length `n`, each inner vector of length
`num_stages` and containing the *node values* (not the node indices).

Starting node at stage 1 is drawn from `initial_probs`.
"""
function sample(rng::AbstractRNG, model::MarkovPriceModel{T}, n::Int) where {T}
    n_stages = num_stages(model)
    K = num_nodes(model)
    trajectories = Vector{Vector{T}}(undef, n)
    for s in 1:n
        traj = Vector{T}(undef, n_stages)
        k = _sample_categorical(rng, model.initial_probs)
        traj[1] = model.node_values[1][k]
        for t in 1:(n_stages - 1)
            k = _sample_categorical(rng, view(model.transition_matrices[t], k, :))
            traj[t + 1] = model.node_values[t + 1][k]
        end
        trajectories[s] = traj
    end
    return trajectories
end

# Internal: sample one categorical realization given a probability
# vector. Linear scan is fine for K <= ~20 (the typical range).
function _sample_categorical(rng::AbstractRNG, probs)
    u = rand(rng)
    acc = zero(eltype(probs))
    @inbounds for k in eachindex(probs)
        acc += probs[k]
        u <= acc && return k
    end
    return length(probs)   # numerical safety net
end
