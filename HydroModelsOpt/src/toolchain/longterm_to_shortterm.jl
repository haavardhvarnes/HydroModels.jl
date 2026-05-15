"""
Toolchain orchestration — the Nordic SINTEF model hierarchy.

The classical chain is:
  EMPS/Samnett (long-term, market clearing)
    → ProdRisk (medium-term, single producer with cuts)
    → SHOP (short-term, operational scheduling)

For now the implemented chain is `long-term SDDP → short-term LP`:
solve a `LongTermHydroProblem` (via the SDDP.jl extension), extract
end-of-horizon water-value cuts from the trained policy, inject those
into a `ShopShortTermProblem`, and solve the short-term LP with the
cuts shaping its end-value.

# Public API

- [`HydropowerToolchain`](@ref) — bundle of long-term problem,
  short-term template, solvers, and reservoir-name mapping.
- [`run_toolchain`](@ref) — execute the chain; returns long + short
  `Solution`s and the cuts that bridged them.
- [`extract_end_value_cuts`](@ref) — best-effort cut extraction from
  an SDDP.jl trained policy (fragile across SDDP.jl versions; returns
  an empty `CutGroup` vector with a warning when extraction fails).
- [`with_end_value_cuts`](@ref) — inject a `Vector{CutGroup{T}}` into
  a `ShopShortTermProblem`, returning a new instance with the cuts
  populated.
- [`synthetic_end_value_cuts`](@ref) — construct a `CutGroup` from
  user-supplied intercepts and slopes. Useful for tests and for
  bypassing the SDDP extraction step when cuts come from another
  source (e.g. external ProdRisk).
"""

# ============================================================
# Bundle type
# ============================================================

"""
    HydropowerToolchain{T}

Bundle of problem definitions and solvers covering the long-term to
short-term hierarchy.

# Fields
- `longterm_model::LongTermHydroProblem` — the long-term scheduler.
- `shortterm_template::ShopShortTermProblem{T}` — the short-term
  template whose `cut_groups` field will be replaced when cuts arrive
  from the long-term solve.
- `longterm_solver::AbstractSolver` — typically `SDDPHydroSolver(...)`.
- `shortterm_solver::AbstractSolver` — typically
  `JuMPSolver(HiGHS.Optimizer)`.
- `reservoir_map::Dict{Symbol, String}` — maps long-term module names
  (`Symbol`) to short-term reservoir names (`String`). The default
  empty dict means "names match directly after `String(symbol)`
  conversion".
"""
Base.@kwdef struct HydropowerToolchain{T}
    longterm_model::LongTermHydroProblem
    shortterm_template::ShopShortTermProblem{T}
    longterm_solver::AbstractSolver
    shortterm_solver::AbstractSolver
    reservoir_map::Dict{Symbol, String} = Dict{Symbol, String}()
end

# ============================================================
# Cut injection
# ============================================================

"""
    with_end_value_cuts(prob, cut_groups; reservoir_map = Dict())

Return a copy of `prob::ShopShortTermProblem{T}` whose `cut_groups`
field is replaced by `cut_groups`, with each cut group's
`res_indices` resolved against `prob.reservoirs`. Long-term reservoir
names (typically the `Symbol` form like `"upper"` after
`String(:upper)`) are translated to short-term reservoir names via
`reservoir_map` when an entry exists; otherwise the long-term name is
used directly.

A cut group whose reservoirs cannot all be resolved is silently
dropped — the short-term LP only enforces cuts on reservoirs it owns.
"""
function with_end_value_cuts(
        prob::ShopShortTermProblem{T},
        cut_groups::AbstractVector{<:CutGroup};
        reservoir_map::AbstractDict = Dict{Symbol, String}(),
    ) where {T}

    short_res_names = sort!(collect(keys(prob.reservoirs)))
    short_idx = Dict(name => i for (i, name) in enumerate(short_res_names))

    resolved = CutGroup{T}[]
    for cg in cut_groups
        new_names = String[]
        new_idx = Int[]
        valid = true
        for rn in cg.res_names
            # rn is a long-term reservoir name (possibly Symbol-derived
            # like "upper"). Map to short-term name if a mapping exists.
            short_name = if rn isa Symbol
                get(reservoir_map, rn, String(rn))
            else
                key = Symbol(rn)
                get(reservoir_map, key, rn)
            end
            push!(new_names, short_name)
            i = get(short_idx, short_name, 0)
            i == 0 && (valid = false; break)
            push!(new_idx, i)
        end
        valid || continue
        push!(resolved, CutGroup{T}(
            name        = cg.name,
            res_names   = new_names,
            res_indices = new_idx,
            ncuts       = cg.ncuts,
            intercept   = T.(cg.intercept),
            slopes      = T.(cg.slopes),
        ))
    end

    return ShopShortTermProblem{T}(
        plants         = prob.plants,
        reservoirs     = prob.reservoirs,
        generators     = prob.generators,
        pumps          = prob.pumps,
        tunnels        = prob.tunnels,
        market         = prob.market,
        dt_schedule    = prob.dt_schedule,
        reserve_groups = prob.reserve_groups,
        reserve_prices = prob.reserve_prices,
        cut_groups     = resolved,
    )
end

# ============================================================
# Synthetic cut construction (test / external-source path)
# ============================================================

"""
    synthetic_end_value_cuts(reservoir_names, intercepts, slopes; name = "_synthetic")

Construct a single-cut-group `Vector{CutGroup{T}}` from user-supplied
intercepts and slopes. `intercepts` is a length-`ncuts` vector;
`slopes` is `[length(reservoir_names) × ncuts]`. Use this when cuts
come from an external source (manually-tuned values, a separate
ProdRisk run, the literature) rather than the SDDP extension.

```julia
cuts = synthetic_end_value_cuts(
    ["upper", "lower"],
    [100.0, 200.0],         # 2 cuts, intercepts in EUR
    [10.0 8.0; 5.0 4.0],    # slopes: 2 reservoirs × 2 cuts (EUR/Mm³)
)
```
"""
function synthetic_end_value_cuts(
        reservoir_names::Vector{String},
        intercepts::AbstractVector,
        slopes::AbstractMatrix;
        name::AbstractString = "_synthetic",
    )
    T = promote_type(eltype(intercepts), eltype(slopes), Float64)
    nres = length(reservoir_names)
    ncuts = length(intercepts)
    size(slopes) == (nres, ncuts) || throw(ArgumentError(
        "slopes must have shape ($(nres), $(ncuts)); got $(size(slopes))"))
    return [CutGroup{T}(
        name        = name,
        res_names   = reservoir_names,
        res_indices = Int[],   # resolved by with_end_value_cuts
        ncuts       = ncuts,
        intercept   = T.(intercepts),
        slopes      = T.(slopes),
    )]
end

# ============================================================
# SDDP cut extraction (best-effort)
# ============================================================

"""
    extract_end_value_cuts(sol::Solution, long_prob::LongTermHydroProblem)

Best-effort extraction of end-of-horizon water-value cuts from an
SDDP.jl-trained model carried in `sol.metadata[:sddp_model]`.

Returns a `Vector{CutGroup{Float64}}`. The cuts are taken from stage 1
of the policy graph — these represent the continuation value V(S_0)
at the start of the long-term horizon, which is the end-of-horizon
value description for a short-term LP whose horizon ends just before
the long-term starts.

This extraction reaches into SDDP.jl's `bellman_function` internals
and is **fragile across SDDP.jl versions**. On any failure it warns
and returns an empty vector — callers can detect this with
`isempty(cuts)` and either fall back to `synthetic_end_value_cuts`
or skip the cut injection step.

A robust version is a B3.5 follow-up.
"""
function extract_end_value_cuts(
        sol::Solution,
        long_prob::LongTermHydroProblem,
    )
    haskey(sol.metadata, :sddp_model) || begin
        @warn "extract_end_value_cuts: no :sddp_model in solution metadata; returning empty cuts."
        return CutGroup{Float64}[]
    end
    sddp_model = sol.metadata[:sddp_model]

    reg_modules = filter(m -> m.reservoir_type isa RegulationReservoir,
                         long_prob.modules)
    isempty(reg_modules) && return CutGroup{Float64}[]
    res_names = [String(m.name) for m in reg_modules]
    nres = length(res_names)

    intercepts = Float64[]
    slopes_rows = Vector{Vector{Float64}}()

    try
        node1 = sddp_model[1]
        # Best-effort access to SDDP.jl's cut storage. The field path
        # `node.bellman_function.global_theta.cuts` is stable through
        # SDDP.jl 1.13 (each entry is an SDDP.Cut with `intercept` and
        # `coefficients::Dict{Symbol, Float64}`). The whole block is
        # wrapped in try/catch because future SDDP.jl versions may
        # restructure the bellman storage; on failure callers fall
        # back to `synthetic_end_value_cuts`.
        for cut in node1.bellman_function.global_theta.cuts
            push!(intercepts, Float64(cut.intercept))
            row = Float64[Float64(get(cut.coefficients, m.name, 0.0))
                          for m in reg_modules]
            push!(slopes_rows, row)
        end
    catch err
        @warn "extract_end_value_cuts: failed to access SDDP cut storage " *
              "(SDDP.jl internal layout changed?). Returning empty cuts. " *
              "Use `synthetic_end_value_cuts` to supply cuts directly." exception = (err, catch_backtrace())
        return CutGroup{Float64}[]
    end

    ncuts = length(intercepts)
    ncuts == 0 && return CutGroup{Float64}[]

    slopes = Matrix{Float64}(undef, nres, ncuts)
    for (c, row) in enumerate(slopes_rows)
        slopes[:, c] = row
    end

    return [CutGroup{Float64}(
        name        = "_long_term_end_value",
        res_names   = res_names,
        res_indices = Int[],
        ncuts       = ncuts,
        intercept   = intercepts,
        slopes      = slopes,
    )]
end

# ============================================================
# Orchestrator
# ============================================================

"""
    run_toolchain(tc::HydropowerToolchain; cuts = nothing)

Execute the long-term → short-term chain:

1. Solve `tc.longterm_model` with `tc.longterm_solver`.
2. Extract end-of-horizon cuts via `extract_end_value_cuts` (unless
   `cuts` is supplied directly — then the long-term solve is still
   performed for the return value but its cuts are discarded).
3. Inject the cuts into `tc.shortterm_template` via
   `with_end_value_cuts`.
4. Solve the resulting short-term problem with `tc.shortterm_solver`.

Returns a `NamedTuple` `(long, short, cuts, shortterm_with_cuts)`.

When `cuts !== nothing`, the long-term solve is skipped entirely and
`long = nothing` in the result. Useful for tests and for bypassing the
fragile SDDP extraction step.
"""
function run_toolchain(tc::HydropowerToolchain; cuts = nothing)
    if cuts === nothing
        long_sol = solve(tc.longterm_model, tc.longterm_solver)
        cuts_used = extract_end_value_cuts(long_sol, tc.longterm_model)
    else
        long_sol = nothing
        cuts_used = cuts
    end

    short_with_cuts = with_end_value_cuts(
        tc.shortterm_template, cuts_used;
        reservoir_map = tc.reservoir_map,
    )
    short_sol = solve(short_with_cuts, tc.shortterm_solver)

    return (
        long              = long_sol,
        short             = short_sol,
        cuts              = cuts_used,
        shortterm_with_cuts = short_with_cuts,
    )
end
