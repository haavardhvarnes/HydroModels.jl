"""
Cascade topology panel — reservoirs as nodes, water-flow connections
as directed edges.

depr's `_fig_topology` (~190 lines) computes elevation-aware connected-
component layouts. The Makie port uses a simpler topological-order
layout that is adequate for cascades up to ~50 reservoirs (the
production NO5 scale). Edges:

- Generator/pump arcs `(from_res, to_res)` → solid arrows.
- Tunnel arcs with both endpoints in reservoirs → grey arrows.
"""
function plot_topology!(ax::Makie.Axis, sol::HydroSolution)
    R = sort!(collect(keys(sol.problem.reservoirs)))
    isempty(R) && return _placeholder!(ax, "No reservoirs")

    # Build directed adjacency from generators / pumps / tunnels
    ridx = Dict(r => i for (i, r) in enumerate(R))
    nr = length(R)
    parents = [Set{Int}() for _ in 1:nr]
    children = [Set{Int}() for _ in 1:nr]
    gen_edges = Tuple{Int, Int}[]
    pump_edges = Tuple{Int, Int}[]
    tun_edges = Tuple{Int, Int}[]

    # B1.6: `from_res` is `Vector{String}` — one graph edge per source.
    for g in sol.problem.generators
        t = get(ridx, g.to_res, 0)
        t > 0 || continue
        for src in g.from_res
            f = get(ridx, src, 0)
            f > 0 && f != t || continue
            push!(children[f], t)
            push!(parents[t], f)
            push!(gen_edges, (f, t))
        end
    end
    for p in sol.problem.pumps
        t = get(ridx, p.to_res, 0)
        t > 0 || continue
        for src in p.from_res
            f = get(ridx, src, 0)
            f > 0 && f != t || continue
            push!(children[f], t)
            push!(parents[t], f)
            push!(pump_edges, (f, t))
        end
    end
    for tun in sol.problem.tunnels
        f = get(ridx, tun.from_node, 0)
        t = get(ridx, tun.to_node, 0)
        f > 0 && t > 0 && f != t || continue
        push!(children[f], t)
        push!(parents[t], f)
        push!(tun_edges, (f, t))
    end

    # Layered layout: each node's y-level = longest path from a source
    levels = zeros(Int, nr)
    function _level(i, depth = 0)
        depth > nr && return 0   # cycle guard
        isempty(parents[i]) && return 0
        return 1 + maximum(_level(p, depth + 1) for p in parents[i])
    end
    for i in 1:nr
        levels[i] = _level(i)
    end
    # Group by level, distribute x within each level
    max_level = maximum(levels; init = 0)
    by_level = [Int[] for _ in 0:max_level]
    for i in 1:nr
        push!(by_level[levels[i] + 1], i)
    end
    xs = zeros(nr)
    ys = zeros(nr)
    for (lvl, group) in enumerate(by_level)
        n = length(group)
        for (k, i) in enumerate(group)
            xs[i] = n == 1 ? 0.5 : (k - 1) / (n - 1)
            ys[i] = max_level == 0 ? 0.0 : 1.0 - (lvl - 1) / max(max_level, 1)
        end
    end

    # Draw edges
    function _draw_edges(edges, color, label)
        isempty(edges) && return
        seg_x = Float64[]
        seg_y = Float64[]
        for (f, t) in edges
            push!(seg_x, xs[f]); push!(seg_x, xs[t]); push!(seg_x, NaN)
            push!(seg_y, ys[f]); push!(seg_y, ys[t]); push!(seg_y, NaN)
        end
        Makie.lines!(ax, seg_x, seg_y;
            color = (parse(Makie.Colorant, color), 0.7),
            linewidth = 1.5, label = label)
    end
    _draw_edges(tun_edges, "#888888", "tunnel")
    _draw_edges(gen_edges, "#1F77B4", "generator")
    _draw_edges(pump_edges, "#EF553B", "pump")

    # Draw nodes
    Makie.scatter!(ax, xs, ys;
        color = parse(Makie.Colorant, "#FECB52"),
        markersize = 14, strokecolor = :black, strokewidth = 0.5)
    for i in 1:nr
        Makie.text!(ax, R[i]; position = (xs[i], ys[i] + 0.02),
            align = (:center, :bottom), fontsize = 10)
    end

    ax.title = "Cascade Topology"
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    if !isempty(gen_edges) || !isempty(pump_edges) || !isempty(tun_edges)
        Makie.axislegend(ax; position = :rb, framevisible = false)
    end
    return ax
end
