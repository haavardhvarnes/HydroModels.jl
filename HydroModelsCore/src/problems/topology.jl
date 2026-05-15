"""
Topology resolution — turn a vector of `HydroModule`s into a `WaterTopology`.
"""

"""
    build_topology(modules::Vector{<:HydroModule})

Resolve symbolic module references (`discharge_to`, `bypass_to`, `spill_to`)
into integer indices and compute a topological ordering (upstream first).

Throws `ArgumentError` if:
- A reference points to a non-existent module
- The graph contains a cycle

Returns a `WaterTopology` ready to be embedded in a problem.
"""
function build_topology(modules::Vector{<:HydroModule})
    n = length(modules)
    by_name = Dict{Symbol, Int}()
    for (i, m) in enumerate(modules)
        haskey(by_name, m.name) && throw(ArgumentError(
            "build_topology: duplicate module name :$(m.name)"))
        by_name[m.name] = i
    end

    discharge = Tuple{Int,Int}[]
    bypass    = Tuple{Int,Int}[]
    spill     = Tuple{Int,Int}[]

    for (i, m) in enumerate(modules)
        _add_edge!(discharge, by_name, i, m.discharge_to)
        _add_edge!(bypass,    by_name, i, m.bypass_to)
        _add_edge!(spill,     by_name, i, m.spill_to)
    end

    # Combined adjacency for cycle detection and topological sort
    adj = [Int[] for _ in 1:n]
    for (u, v) in vcat(discharge, bypass, spill)
        push!(adj[u], v)
    end

    order = _topological_sort(adj)
    isnothing(order) && throw(ArgumentError(
        "build_topology: water flow graph contains a cycle"))

    return WaterTopology(by_name, discharge, bypass, spill, order)
end

function _add_edge!(edges, by_name, from::Int, to::Union{Nothing, Symbol})
    isnothing(to) && return
    haskey(by_name, to) || throw(ArgumentError(
        "build_topology: reference to unknown module :$to"))
    push!(edges, (from, by_name[to]))
end

# Kahn's algorithm; returns `nothing` on cycle.
function _topological_sort(adj::Vector{Vector{Int}})
    n = length(adj)
    indeg = zeros(Int, n)
    for u in 1:n
        for v in adj[u]
            indeg[v] += 1
        end
    end
    queue = [u for u in 1:n if indeg[u] == 0]
    order = Int[]
    while !isempty(queue)
        u = popfirst!(queue)
        push!(order, u)
        for v in adj[u]
            indeg[v] -= 1
            indeg[v] == 0 && push!(queue, v)
        end
    end
    return length(order) == n ? order : nothing
end
