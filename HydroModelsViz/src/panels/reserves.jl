"""
Reserve-market panels: per-product allocation and per-group obligation.
"""

"""
    plot_reserves_alloc!(ax, sol; gens = nothing)

Stacked reserve capacity by product over time, summed across generators.
Mirrors `_fig_reserves_alloc`.
"""
function plot_reserves_alloc!(ax::Makie.Axis, sol::HydroSolution; gens = nothing)
    isempty(sol.r_alloc.time) && return _placeholder!(ax, "No reserve allocation")

    rows = collect(eachindex(sol.r_alloc.time))
    if gens !== nothing
        rows = filter(i -> sol.r_alloc.generator[i] in gens, rows)
    end
    isempty(rows) && return _placeholder!(ax, "No reserve allocation for this plant")

    products = sort!(unique(sol.r_alloc.product[i] for i in rows))
    times = _time_grid(sol)
    cmap = _color_map(products)
    pos = Dict(t => i for (i, t) in enumerate(times))

    cumulative = zeros(length(times))
    for prod in products
        agg = zeros(length(times))
        for i in rows
            sol.r_alloc.product[i] == prod || continue
            j = get(pos, sol.r_alloc.time[i], 0)
            j == 0 && continue
            agg[j] += sol.r_alloc.R_MW[i]
        end
        new_top = cumulative .+ agg
        Makie.band!(ax, 1:length(times), cumulative, new_top;
            color = (parse(Makie.Colorant, cmap[prod]), 0.7), label = prod)
        cumulative = new_top
    end

    ax.title = "Reserve Capacity Allocated"
    ax.ylabel = "Reserve (MW)"
    ax.xlabel = "Time index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end

"""
    plot_reserves_groups!(ax, sol)

Per reserve-group: obligation line + delivered line; shortfall (slack)
drawn as a filled red band when non-zero. Mirrors
`_fig_reserves_groups`.
"""
function plot_reserves_groups!(ax::Makie.Axis, sol::HydroSolution)
    rgs = sol.problem.reserve_groups
    isempty(rgs) && return _placeholder!(ax, "No reserve group obligations defined")

    times = _time_grid(sol)
    pos = Dict(t => i for (i, t) in enumerate(times))

    for (g_idx, rg) in enumerate(rgs)
        col = parse(Makie.Colorant, _PALETTE[mod1(g_idx, length(_PALETTE))])

        # Obligation (from problem; LOCF onto the model grid)
        oblig = zeros(length(times))
        if !isempty(rg.obligation)
            obs_times = sort!(collect(keys(rg.obligation)))
            obs_vals = [rg.obligation[t] for t in obs_times]
            for (j, t) in enumerate(times)
                idx = searchsortedlast(obs_times, t)
                idx == 0 && continue
                oblig[j] = obs_vals[idx]
            end
        end
        Makie.lines!(ax, 1:length(times), oblig;
            color = col, linewidth = 2.0,
            label = "$(rg.name) obligation")

        # Delivered (sum from r_alloc filtered to this product)
        delivered = zeros(length(times))
        for i in eachindex(sol.r_alloc.time)
            sol.r_alloc.product[i] == rg.product || continue
            j = get(pos, sol.r_alloc.time[i], 0)
            j == 0 && continue
            delivered[j] += sol.r_alloc.R_MW[i]
        end
        Makie.lines!(ax, 1:length(times), delivered;
            color = col, linewidth = 1.5, linestyle = :dash,
            label = "$(rg.name) delivered")

        # Slack (only if any non-zero)
        rgname_key = "$(rg.name)/$(rg.product)"
        slack = zeros(length(times))
        for i in eachindex(sol.r_slack.time)
            sol.r_slack.group[i] == rgname_key || continue
            j = get(pos, sol.r_slack.time[i], 0)
            j == 0 && continue
            slack[j] = sol.r_slack.slack_MW[i]
        end
        if any(>(1.0e-6), slack)
            Makie.band!(ax, 1:length(times), zeros(length(times)), slack;
                color = (Makie.RGBAf(0.94, 0.33, 0.23, 0.4), 0.4),
                label = "$(rg.name) shortfall")
        end
    end

    ax.title = "Reserve Group Obligations"
    ax.ylabel = "Reserve (MW)"
    ax.xlabel = "Time index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end
