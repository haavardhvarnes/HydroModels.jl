"""
Internal helpers for the panel functions in `panels/`. Not part of the
public API.
"""

# Consistent color palette across panels — the Plotly defaults that depr
# used, retained here so a side-by-side comparison against the depr
# Plotly dashboard reads cleanly. Hex strings; Makie accepts them via
# `parse(Colorant, ...)`.
const _PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]

"""Build a stable name → hex-color mapping by index into `_PALETTE`."""
function _color_map(names)
    return Dict(n => _PALETTE[mod1(i, length(_PALETTE))] for (i, n) in enumerate(names))
end

"""
    _series_for(table, namecol, name, valcol)

Extract `(times, values)` from a long-format Tables.jl-compatible
NamedTuple, filtering on `namecol == name` and sorting by time. Empty
when no rows match.
"""
function _series_for(table::NamedTuple, namecol::Symbol, name::AbstractString,
                     valcol::Symbol)
    isempty(getproperty(table, :time)) && return DateTime[], Float64[]
    mask = getproperty(table, namecol) .== name
    times = getproperty(table, :time)[mask]
    vals = getproperty(table, valcol)[mask]
    perm = sortperm(times)
    return times[perm], vals[perm]
end

"""
    _time_grid(sol)

Return the unique sorted vector of `DateTime`s appearing in the
solution's revenue table — that's the canonical model-time grid the
LP used.
"""
function _time_grid(sol::HydroSolution)
    return sort!(unique(sol.revenue.time))
end

"""
    _aligned_series(sol, table, namecol, name, valcol)

Build a per-time vector of length `length(_time_grid(sol))`, filled
with the corresponding values from `table` and zero where the time
isn't present. Used by stacked-area panels.
"""
function _aligned_series(sol::HydroSolution, table::NamedTuple,
                         namecol::Symbol, name::AbstractString,
                         valcol::Symbol)
    times = _time_grid(sol)
    out = zeros(Float64, length(times))
    isempty(getproperty(table, :time)) && return times, out
    pos = Dict(t => i for (i, t) in enumerate(times))
    mask = getproperty(table, namecol) .== name
    for i in findall(mask)
        t = getproperty(table, :time)[i]
        idx = get(pos, t, 0)
        idx == 0 && continue
        out[idx] = getproperty(table, valcol)[i]
    end
    return times, out
end

"""
    _placeholder!(ax, msg)

Display a centered grey "no data" message on an empty axis. Used by
panels that have nothing to draw (e.g. pumps panel on a no-pump
problem).
"""
function _placeholder!(ax, msg::AbstractString)
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    Makie.text!(ax, msg;
        position = (0.5, 0.5), space = :relative,
        align = (:center, :center), color = (:gray, 0.7), fontsize = 14,
    )
    return ax
end
