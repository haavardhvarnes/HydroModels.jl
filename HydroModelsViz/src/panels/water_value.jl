"""
Water-value panel — SDDP cut envelope per reservoir + optimal final-
storage marker.

depr's `_fig_water_value` produces a Plotly subplot grid (one cell per
reservoir-with-cuts). For the Makie port we render a single axis when
called with one reservoir, or all per-reservoir cuts overlaid with
unique colours when called for the full set. The grid layout belongs
to the dashboard composition, not this panel.
"""
function plot_water_value!(ax::Makie.Axis, sol::HydroSolution;
                            reservoirs = nothing)
    # Reservoirs that actually carry per-reservoir cut data.
    candidate = String[]
    for (rname, rr) in sol.problem.reservoirs
        rr.water_value === nothing && continue
        isempty(rr.water_value.ref) && continue
        push!(candidate, rname)
    end
    if reservoirs !== nothing
        candidate = filter(in(reservoirs), candidate)
    end
    sort!(candidate)
    isempty(candidate) && return _placeholder!(ax, "No water value cuts")

    cmap = _color_map(candidate)

    # Optimal final storage from sol.storage (last time per reservoir)
    last_t = isempty(sol.storage.time) ? nothing : maximum(sol.storage.time)
    s_opt = Dict{String, Float64}()
    if last_t !== nothing
        for i in eachindex(sol.storage.time)
            sol.storage.time[i] == last_t || continue
            s_opt[sol.storage.reservoir[i]] = sol.storage.S[i]
        end
    end

    for rname in candidate
        rr = sol.problem.reservoirs[rname]
        smin = rr.s_min
        smax = isfinite(rr.s_max) ? rr.s_max : smin + 1.0
        svec = collect(range(smin, smax; length = 200))

        ncuts = length(rr.water_value.ref)
        envelope = Float64[]
        for s in svec
            v = minimum(rr.water_value.ref[c] + rr.water_value.slope[c] * s
                        for c in 1:ncuts)
            push!(envelope, v)
        end

        col = parse(Makie.Colorant, cmap[rname])
        Makie.lines!(ax, svec, envelope; color = col, linewidth = 2.5,
            label = "$(rname) envelope")

        # A subset of individual cuts behind the envelope (light grey)
        step = max(1, div(ncuts, 30))
        for c in 1:step:ncuts
            y = [rr.water_value.ref[c] + rr.water_value.slope[c] * s for s in svec]
            Makie.lines!(ax, svec, y; color = (col, 0.15), linewidth = 0.5)
        end

        if haskey(s_opt, rname)
            s_t = s_opt[rname]
            v_t = minimum(rr.water_value.ref[c] + rr.water_value.slope[c] * s_t
                          for c in 1:ncuts)
            Makie.scatter!(ax, [s_t], [v_t];
                color = parse(Makie.Colorant, "#EF553B"),
                markersize = 14, label = "$(rname) S_T")
        end
    end

    ax.title = "Water Value Functions (SDDP Envelope)"
    ax.ylabel = "Future cost (EUR)"
    ax.xlabel = "End-of-horizon storage (Mm³)"
    Makie.axislegend(ax; position = :lt, framevisible = false)
    return ax
end
