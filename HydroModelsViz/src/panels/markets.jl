"""
Market panels: prices + revenue.
"""

"""
    plot_prices!(ax, sol)

Spot price plus each reserve-product price overlaid on the same axis.
Mirrors `_fig_prices`. Reserve prices are LOCF-interpolated to the
model time grid.
"""
function plot_prices!(ax::Makie.Axis, sol::HydroSolution)
    if isempty(sol.revenue.time)
        return _placeholder!(ax, "No market data")
    end
    times = sol.revenue.time
    prices = sol.revenue.price

    Makie.lines!(ax, 1:length(times), prices;
        color = parse(Makie.Colorant, "#1F77B4"), linewidth = 2.0, label = "Spot")

    # Reserve product prices (from prob.reserve_prices, LOCF onto sol time grid)
    products = sort!(collect(keys(sol.problem.reserve_prices)))
    cmap = _color_map(products)
    pos = Dict(t => i for (i, t) in enumerate(times))
    for prod in products
        price_ts = sol.problem.reserve_prices[prod]
        isempty(price_ts) && continue
        obs_t = sort!(collect(keys(price_ts)))
        obs_v = [price_ts[t] for t in obs_t]
        vec = zeros(length(times))
        for (j, t) in enumerate(times)
            idx = searchsortedlast(obs_t, t)
            idx == 0 && continue
            vec[j] = obs_v[idx]
        end
        all(iszero, vec) && continue
        Makie.lines!(ax, 1:length(times), vec;
            color = parse(Makie.Colorant, cmap[prod]),
            linewidth = 1.5, linestyle = :dot, label = prod)
    end

    ax.title = "Market Prices"
    ax.ylabel = "Price (EUR/MWh)"
    ax.xlabel = "Time index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end

"""
    plot_revenue!(ax, sol)

Hourly revenue as bars (kEUR) with cumulative revenue overlaid as a
line. Both share the axis; the cumulative line is scaled to fit the
bar range for single-panel rendering. Mirrors `_fig_revenue`.
"""
function plot_revenue!(ax::Makie.Axis, sol::HydroSolution)
    if isempty(sol.revenue.time)
        return _placeholder!(ax, "No revenue data")
    end

    rev = sol.revenue.revenue
    hourly_kEUR = rev ./ 1.0e3
    cum_MEUR = cumsum(rev) ./ 1.0e6

    Makie.barplot!(ax, 1:length(rev), hourly_kEUR;
        color = parse(Makie.Colorant, "#636EFA"),
        label = "Hourly (kEUR)")

    # Cumulative on same axis, scaled into the bar range
    max_bar = isempty(hourly_kEUR) ? 1.0 : maximum(abs, hourly_kEUR; init = 1.0)
    max_cum = isempty(cum_MEUR) ? 1.0 : maximum(abs, cum_MEUR; init = 1.0)
    scale = max_cum > 0 ? max_bar / max_cum : 1.0
    Makie.lines!(ax, 1:length(rev), cum_MEUR .* scale;
        color = parse(Makie.Colorant, "#EF553B"), linewidth = 2.0,
        label = "Cumulative × $(round(scale; sigdigits = 2)) (MEUR)")

    ax.title = "Revenue"
    ax.ylabel = "Hourly revenue (kEUR)"
    ax.xlabel = "Time index"
    Makie.axislegend(ax; position = :rt, framevisible = false)
    return ax
end
