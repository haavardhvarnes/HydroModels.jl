"""
Periodic Auto-Regressive of order 1 (PAR(1)) model for hydropower
inflow.

The seasonal AR(1) family of choice for Nordic SDDP backward recursion:

    X_t = μ_p + φ_p × (X_{t-1} - μ_{p-1}) + ε_t,    ε_t ~ N(0, σ_p²)

where `p = period(t)` cycles through the year (e.g., week 1..52). Each
period carries its own mean, AR(1) coefficient, and residual standard
deviation, fit from multi-year historical series.

References:
- ProdRisk inflow model:
  <https://docs.prodrisk.sintef.energy/examples/inflow/inflow_model/>
- Gjelsvik, Mo, Haugstad (2010), section on inflow modeling
"""

"""
    PARProcess{T}

Fitted PAR(1) model. Field semantics:

- `period_means[p]`     — μ_p, the long-run mean for period `p`.
- `period_ar1[p]`       — φ_p, the AR(1) coefficient for period `p`.
- `period_residual_std[p]` — σ_p, the residual standard deviation.

The first period's φ and σ are conventionally zero (no previous period
to regress against on the first step of a trajectory).
"""
struct PARProcess{T}
    period_means::Vector{T}
    period_ar1::Vector{T}
    period_residual_std::Vector{T}
end

Base.eltype(::PARProcess{T}) where {T} = T

"""
    num_periods(p::PARProcess) -> Int

Number of within-year periods (typically `52` for weekly data) the
PAR(1) model was fit to.
"""
num_periods(p::PARProcess) = length(p.period_means)

"""
    fit(::Type{PARProcess}, data::AbstractMatrix; period_first = true)

Fit a PAR(1) model to historical data laid out as a matrix:

- `period_first = true` (default): `data[year, period]`. Common when
  historical years are stacked as rows.
- `period_first = false`: `data[period, year]`. Useful when you read
  the matrix in the other orientation.

Fits each period independently via ordinary least squares against the
previous-period residual.
"""
function fit(::Type{PARProcess}, data::AbstractMatrix; period_first::Bool = true)
    X = period_first ? data : permutedims(data)
    n_years, n_periods = size(X)
    n_years >= 2 || throw(ArgumentError(
        "PARProcess fit needs at least 2 years of data, got $(n_years)"))
    n_periods >= 1 || throw(ArgumentError(
        "PARProcess fit needs at least 1 period, got $(n_periods)"))

    T = eltype(X)
    μ = vec(Statistics.mean(X; dims = 1))           # n_periods
    φ = zeros(T, n_periods)
    σ = zeros(T, n_periods)

    for p in 2:n_periods
        r_curr = X[:, p] .- μ[p]
        r_prev = X[:, p - 1] .- μ[p - 1]
        denom = sum(abs2, r_prev)
        φ[p] = denom > 1.0e-12 ? (dot(r_curr, r_prev) / denom) : zero(T)
        residuals = r_curr .- φ[p] .* r_prev
        σ[p] = n_years > 1 ? Statistics.std(residuals; corrected = true) : zero(T)
    end

    return PARProcess{T}(μ, φ, σ)
end

"""
    sample(rng, model::PARProcess, x0; n_periods = num_periods(model))

Draw one trajectory of length `n_periods` from `model`, starting with
`x0` at period 1. Returns a `Vector{T}` of length `n_periods`.

The trajectory advances one period at a time:

    out[p] = μ_p + φ_p × (out[p-1] - μ_{p-1}) + σ_p × randn()

When `n_periods > num_periods(model)` the period index cycles via
`mod1`, treating the fitted parameters as periodic.
"""
function sample(rng::AbstractRNG, model::PARProcess{T}, x0::Real;
                n_periods::Int = num_periods(model)) where {T}
    out = Vector{T}(undef, n_periods)
    out[1] = T(x0)
    np = length(model.period_means)
    for p in 2:n_periods
        idx_curr = mod1(p, np)
        idx_prev = mod1(p - 1, np)
        ε = randn(rng) * model.period_residual_std[idx_curr]
        out[p] = model.period_means[idx_curr] +
            model.period_ar1[idx_curr] *
                (out[p - 1] - model.period_means[idx_prev]) + ε
    end
    return out
end

"""
    sample(rng, model::PARProcess, x0::Real, n::Int; n_periods = num_periods(model))

Draw `n` independent trajectories. Returns `Vector{Vector{T}}` of
length `n`, each inner vector of length `n_periods`.
"""
function sample(rng::AbstractRNG, model::PARProcess{T}, x0::Real, n::Int;
                n_periods::Int = num_periods(model)) where {T}
    return [sample(rng, model, x0; n_periods = n_periods) for _ in 1:n]
end
