"""
Scenario generation utilities for SDDP / forward simulation.

Two paths:

1. **Historical resampling** — bootstrap entire trajectories from a
   matrix of historical scenarios. Preserves the joint distribution
   across all variables (inflow, price, …) and is the typical Nordic
   choice for forward simulation.
2. **Fitted-model sampling** — draw trajectories from a fitted
   `PARProcess` or `MarkovPriceModel`. Use for backward recursion or
   when richer Monte Carlo is needed than the historical sample
   contains.

Both produce a `Vector{Vector{T}}` of trajectories, each of length
`num_stages`. Convert to Core's `StagewiseIndependent` via
[`to_stagewise_independent`](@ref) for use with the SDDP bridge.
"""

"""
    historical_bootstrap(rng, history::AbstractMatrix, n_samples::Int)

Draw `n_samples` trajectories by sampling whole rows (with
replacement) from `history`. `history[s, t]` is the value at scenario
`s` and stage `t`. Returns `Vector{Vector{T}}` of length `n_samples`.

Whole-row sampling preserves any cross-stage correlation embedded in
the historical record.
"""
function historical_bootstrap(rng::AbstractRNG, history::AbstractMatrix,
                              n_samples::Int)
    n_scen, n_stages = size(history)
    n_scen >= 1 || throw(ArgumentError("history must have at least one scenario"))
    T = eltype(history)
    out = Vector{Vector{T}}(undef, n_samples)
    for i in 1:n_samples
        s = rand(rng, 1:n_scen)
        out[i] = Vector{T}(history[s, :])
    end
    return out
end

"""
    to_stagewise_independent(trajectories::Vector{<:AbstractVector};
                              equal_probability = true)

Convert a vector of per-stage realizations into Core's
`StagewiseIndependent{T}` (the uncertainty type the SDDP bridge
dispatches on).

The argument is a *list of trajectories*, each of length `n_stages`,
where each entry is a single scalar realization at that stage. The
result has one realization per stage at the same index across all
trajectories: `realizations[t][k] = [trajectories[k][t]]`.

This is the simplest single-variable mapping. For multi-variable
uncertainty (e.g. inflow per reservoir + price), build the
realization vectors manually.
"""
function to_stagewise_independent(
        trajectories::AbstractVector{<:AbstractVector{T}};
        equal_probability::Bool = true,
    ) where {T}

    isempty(trajectories) && throw(ArgumentError("no trajectories supplied"))
    n_stages = length(trajectories[1])
    n_scen = length(trajectories)
    for tr in trajectories
        length(tr) == n_stages || throw(ArgumentError(
            "all trajectories must have the same length; got $(length(tr)) " *
            "vs expected $(n_stages)"))
    end

    realizations = [
        [T[trajectories[k][t]] for k in 1:n_scen]
        for t in 1:n_stages
    ]
    probabilities = if equal_probability
        [[one(T) / T(n_scen) for _ in 1:n_scen] for _ in 1:n_stages]
    else
        throw(ArgumentError(
            "non-equal probabilities not supported in to_stagewise_independent; " *
            "construct StagewiseIndependent manually."))
    end

    return StagewiseIndependent{T}(realizations, probabilities)
end
