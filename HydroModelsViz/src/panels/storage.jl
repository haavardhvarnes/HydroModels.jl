"""
Reservoir storage + inflow panels.
"""

"""
    plot_storage!(ax, sol; reservoirs = nothing)

Per-reservoir storage trajectory with min/max bounds drawn as a faded
band envelope behind the line. Mirrors `_fig_storage`.
"""
function plot_storage!(ax::Makie.Axis, sol::HydroSolution; reservoirs = nothing)
    R = sort!(collect(keys(sol.problem.reservoirs)))
    res_list = reservoirs === nothing ? R : filter(in(reservoirs), R)
    if isempty(res_list)
        return _placeholder!(ax, "No reservoirs")
    end

    cmap = _color_map(R)
    for r in res_list
        rs = sol.problem.reservoirs[r]
        _, svec = _series_for(sol.storage, :reservoir, r, :S)
        isempty(svec) && continue
        n = length(svec)
        col = parse(Makie.Colorant, cmap[r])
        smin_v = fill(rs.s_min, n)
        smax_v = isfinite(rs.s_max) ? fill(rs.s_max, n) : fill(maximum(svec) * 1.1, n)
        Makie.band!(ax, 1:n, smin_v, smax_v; color = (col, 0.15))
        Makie.lines!(ax, 1:n, svec; color = col, linewidth = 1.8, label = r)
    end

    ax.title = "Reservoir Storage"
    ax.ylabel = "Storage (Mm³)"
    ax.xlabel = "Time index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end

"""
    plot_inflows!(ax, sol; reservoirs = nothing)

Per-reservoir inflow time series. Mirrors `_fig_inflows`. Inflow is
read directly from `prob.reservoirs[name].inflow` (no LOCF — we draw
the raw observation points), since the LP-internal interpolated grid
isn't preserved on `HydroSolution`.
"""
function plot_inflows!(ax::Makie.Axis, sol::HydroSolution; reservoirs = nothing)
    R = sort!(collect(keys(sol.problem.reservoirs)))
    res_list = reservoirs === nothing ? R : filter(in(reservoirs), R)
    if isempty(res_list)
        return _placeholder!(ax, "No reservoirs")
    end

    cmap = _color_map(R)
    plotted = false
    for r in res_list
        rs = sol.problem.reservoirs[r]
        isempty(rs.inflow) && continue
        ts = sort!(collect(keys(rs.inflow.by_time)))
        vs = [rs.inflow.by_time[t] for t in ts]
        Makie.lines!(ax, 1:length(vs), vs;
            color = parse(Makie.Colorant, cmap[r]), linewidth = 1.5, label = r)
        plotted = true
    end
    if !plotted
        return _placeholder!(ax, "No inflow data")
    end

    ax.title = "Inflow"
    ax.ylabel = "Inflow (m³/s)"
    ax.xlabel = "Observation index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end
