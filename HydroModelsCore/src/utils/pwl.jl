"""
Piecewise-linear curves — used for volume-head relations, PQ curves, and
HPF approximations.
"""

"""
    PiecewiseLinear{T}

A piecewise-linear curve defined by sorted x/y breakpoints.

# Fields
- `x::Vector{T}`: breakpoint abscissae, strictly increasing
- `y::Vector{T}`: corresponding ordinates
- `is_convex::Bool`: cached convexity flag (computed at construction)
"""
struct PiecewiseLinear{T}
    x::Vector{T}
    y::Vector{T}
    is_convex::Bool

    function PiecewiseLinear(x::Vector{T}, y::Vector{T}) where {T}
        length(x) == length(y) || throw(ArgumentError(
            "PiecewiseLinear: x and y must have equal length"))
        length(x) >= 2 || throw(ArgumentError(
            "PiecewiseLinear: need at least 2 breakpoints"))
        issorted(x) || throw(ArgumentError(
            "PiecewiseLinear: x must be sorted"))
        return new{T}(x, y, _check_convex(x, y))
    end
end

function _check_convex(x::AbstractVector, y::AbstractVector)
    length(x) <= 2 && return true
    prev_slope = (y[2] - y[1]) / (x[2] - x[1])
    @inbounds for i in 3:length(x)
        slope = (y[i] - y[i-1]) / (x[i] - x[i-1])
        slope < prev_slope - 1e-12 && return false
        prev_slope = slope
    end
    return true
end

"""
    evaluate(curve, x)

Evaluate the piecewise-linear curve at `x` using linear interpolation
between breakpoints. Values outside `[curve.x[1], curve.x[end]]` are
extrapolated using the nearest segment.
"""
function evaluate(curve::PiecewiseLinear{T}, x::Real) where {T}
    xs, ys = curve.x, curve.y
    if x <= xs[1]
        # Extrapolate using first segment
        slope = (ys[2] - ys[1]) / (xs[2] - xs[1])
        return ys[1] + slope * (x - xs[1])
    elseif x >= xs[end]
        # Extrapolate using last segment
        slope = (ys[end] - ys[end-1]) / (xs[end] - xs[end-1])
        return ys[end] + slope * (x - xs[end])
    end

    # Binary search for the segment
    idx = searchsortedlast(xs, x)
    t = (x - xs[idx]) / (xs[idx+1] - xs[idx])
    return ys[idx] + t * (ys[idx+1] - ys[idx])
end

"""
    slopes(curve)

Return the slope of each segment as a `Vector` of length `length(curve.x) - 1`.
"""
function slopes(curve::PiecewiseLinear)
    xs, ys = curve.x, curve.y
    return [(ys[i+1] - ys[i]) / (xs[i+1] - xs[i]) for i in 1:length(xs)-1]
end

Base.length(curve::PiecewiseLinear) = length(curve.x)

function Base.show(io::IO, curve::PiecewiseLinear)
    print(io, "PiecewiseLinear(", length(curve), " breakpoints, ",
          curve.is_convex ? "convex" : "non-convex", ")")
end

"""
    PWL

Alias for `PiecewiseLinear`. Provides the depr-compatible name used by
SHOP-style operational types (`Generator.curves`, `Pump.curves`,
`Reservoir.vol_breaks` / `head_breaks`, generator efficiency curves).
The underlying type is `PiecewiseLinear{T}` with its sortedness and
length checks; convexity is cached at construction.
"""
const PWL = PiecewiseLinear
