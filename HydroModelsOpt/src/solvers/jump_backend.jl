"""
JuMP / MathOptInterface solver wrapper — the most general baseline.

Converts a `HydroModels` problem into a JuMP model and hands it to any MOI-
compatible solver (HiGHS, Gurobi, CPLEX, MadNLP, cuOpt via MOI, etc.).

This is intentionally a "naive" deterministic-equivalent formulation: for
multistage stochastic problems with many scenarios it will blow up. Use
the SDDP solver family instead for those.
"""

"""
    JuMPSolver(optimizer; options = Dict{Symbol, Any}())

Wrap a JuMP-compatible optimizer factory. `optimizer` is typically a
`MOI.OptimizerWithAttributes` or a constructor like `HiGHS.Optimizer`;
pass `nothing` for a build-only path (model constructed but not solved).
`options` are forwarded to JuMP via `set_optimizer_attribute(model,
string(k), v)` after construction.

# Examples
```julia
JuMPSolver(HiGHS.Optimizer)
JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false))
JuMPSolver(nothing)   # build but don't solve
```
"""
Base.@kwdef struct JuMPSolver{O} <: AbstractSolver
    optimizer::O
    options::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

# Convenience outer constructor: positional optimizer + keyword options.
# Base.@kwdef supports all-positional or all-keyword but not mixed, so
# add this for ergonomics: JuMPSolver(HiGHS.Optimizer; options = ...).
# Accepts any `AbstractDict` and widens the value type to `Any` so callers
# can pass `Dict(:flag => false)` without manual type annotation.
function JuMPSolver(
        optimizer;
        options::AbstractDict = Dict{Symbol, Any}(),
    )
    return JuMPSolver(
        optimizer,
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in options),
    )
end
