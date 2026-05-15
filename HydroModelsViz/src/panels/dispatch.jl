"""
Generation / pump / spill dispatch panels.
"""

"""
    plot_dispatch!(ax, sol; gens = nothing)

Stacked generation by generator over time with the spot price overlaid
on a right-hand secondary axis (drawn as a separate `Axis` linked via
`Makie.hidespines!`). Mirrors `_fig_dispatch` from the depr Plotly
dashboard.
"""
function plot_dispatch!(ax::Makie.Axis, sol::HydroSolution; gens = nothing)
    GEN = [g.name for g in sol.problem.generators]
    gen_list = gens === nothing ? GEN : filter(in(gens), GEN)
    if isempty(gen_list)
        return _placeholder!(ax, "No generators")
    end

    times = _time_grid(sol)
    cmap = _color_map(GEN)

    # Stack from the bottom up
    cumulative = zeros(length(times))
    for g in gen_list
        _, v = _aligned_series(sol, sol.p_gen, :generator, g, :P)
        new_top = cumulative .+ v
        Makie.band!(ax, 1:length(times), cumulative, new_top;
            color = (parse(Makie.Colorant, cmap[g]), 0.7), label = g)
        cumulative = new_top
    end

    ax.title = "Generation Dispatch"
    ax.ylabel = "Power (MW)"
    ax.xlabel = "Time index"

    # Price overlay — draw on the same axis with relative scaling so the
    # picture stays single-panel. Annotate the scaling in the legend.
    if !isempty(sol.revenue.time)
        rev = sol.revenue
        prices = rev.price
        max_p = isempty(prices) ? 1.0 : maximum(prices; init = 1.0)
        max_g = isempty(cumulative) ? 1.0 : maximum(cumulative; init = 1.0)
        scale = max_g > 0 && max_p > 0 ? max_g / max_p : 1.0
        Makie.lines!(ax, 1:length(times), prices .* scale;
            color = :black, linestyle = :dot, linewidth = 1.5,
            label = "Spot × $(round(scale; sigdigits = 2))",
        )
    end

    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end

"""
    plot_pumps!(ax, sol; pumps = nothing)

Stacked pump consumption by pump over time. Mirrors `_fig_pumps`.
"""
function plot_pumps!(ax::Makie.Axis, sol::HydroSolution; pumps = nothing)
    PUMP = [p.name for p in sol.problem.pumps]
    pump_list = pumps === nothing ? PUMP : filter(in(pumps), PUMP)
    if isempty(pump_list)
        return _placeholder!(ax, "No pumps")
    end

    times = _time_grid(sol)
    cmap = _color_map(PUMP)

    cumulative = zeros(length(times))
    for p in pump_list
        _, v = _aligned_series(sol, sol.e_pump, :pump, p, :E)
        new_top = cumulative .+ v
        Makie.band!(ax, 1:length(times), cumulative, new_top;
            color = (parse(Makie.Colorant, cmap[p]), 0.7), label = p)
        cumulative = new_top
    end

    ax.title = "Pump Dispatch"
    ax.ylabel = "Power consumption (MW)"
    ax.xlabel = "Time index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end

"""
    plot_spill!(ax, sol; reservoirs = nothing)

Per-reservoir spill flow over time; reservoirs with zero spill across
the horizon are dropped. Mirrors `_fig_spill`.
"""
function plot_spill!(ax::Makie.Axis, sol::HydroSolution; reservoirs = nothing)
    R = sort!(collect(keys(sol.problem.reservoirs)))
    candidate = reservoirs === nothing ? R : filter(in(reservoirs), R)

    spill_res = String[]
    for r in candidate
        _, vals = _series_for(sol.spill, :reservoir, r, :Q_spill)
        isempty(vals) && continue
        maximum(vals; init = 0.0) > 1.0e-6 && push!(spill_res, r)
    end
    if isempty(spill_res)
        return _placeholder!(ax, "No spill")
    end

    cmap = _color_map(spill_res)
    for r in spill_res
        _, vals = _series_for(sol.spill, :reservoir, r, :Q_spill)
        Makie.lines!(ax, 1:length(vals), vals;
            color = parse(Makie.Colorant, cmap[r]), linewidth = 1.5, label = r)
    end

    ax.title = "Spill"
    ax.ylabel = "Spill (m³/s)"
    ax.xlabel = "Time index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end
