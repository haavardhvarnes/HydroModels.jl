"""
Dashboard composition: lay out the 11 panel functions into a single
`Figure`.
"""

"""
    plot_dashboard(sol; size = (1600, 1800))

Build a 4×3 `Figure` containing the 11 panel functions plus a KPI
header (objective, termination, total revenue). Returns the `Figure`.
The backend must be loaded by the caller (`using CairoMakie`,
`using GLMakie`, or `using WGLMakie`).

# Panel layout

```
┌──────────────────────────────────────────────────────────────┐
│  KPI:  termination = OPTIMAL    objective = 0.0105 MEUR  …   │
├────────────────┬────────────────┬────────────────┬───────────┤
│  Dispatch      │  Storage       │  Pumps         │  Spill    │
├────────────────┼────────────────┼────────────────┼───────────┤
│  Prices        │  Revenue       │  Reserve alloc │  Rsv grps │
├────────────────┼────────────────┼────────────────┴───────────┤
│  Water value   │  Inflows       │  Topology                  │
└────────────────┴────────────────┴────────────────────────────┘
```
"""
function plot_dashboard(sol::HydroSolution; size = (1600, 1800))
    fig = Makie.Figure(size = size)

    # KPI header
    objective_MEUR = sol.objective / 1.0e6
    total_rev_MEUR = isempty(sol.revenue.revenue) ? 0.0 :
        sum(sol.revenue.revenue) / 1.0e6
    n_res = length(sol.problem.reservoirs)
    n_gen = length(sol.problem.generators)
    n_pump = length(sol.problem.pumps)
    kpi = Makie.Label(fig[1, 1:4],
        "Termination: $(sol.termination)   |   " *
        "Objective: $(round(objective_MEUR; digits = 4)) MEUR   |   " *
        "Energy revenue: $(round(total_rev_MEUR; digits = 4)) MEUR   |   " *
        "$(n_res) reservoirs · $(n_gen) generators · $(n_pump) pumps";
        fontsize = 14, halign = :left,
    )

    # Row 1 — Dispatch / Storage / Pumps / Spill
    plot_dispatch!(Makie.Axis(fig[2, 1]), sol)
    plot_storage!(Makie.Axis(fig[2, 2]), sol)
    plot_pumps!(Makie.Axis(fig[2, 3]), sol)
    plot_spill!(Makie.Axis(fig[2, 4]), sol)

    # Row 2 — Prices / Revenue / Reserve alloc / Reserve groups
    plot_prices!(Makie.Axis(fig[3, 1]), sol)
    plot_revenue!(Makie.Axis(fig[3, 2]), sol)
    plot_reserves_alloc!(Makie.Axis(fig[3, 3]), sol)
    plot_reserves_groups!(Makie.Axis(fig[3, 4]), sol)

    # Row 3 — Water value / Inflows / Topology (last spans 2 columns)
    plot_water_value!(Makie.Axis(fig[4, 1]), sol)
    plot_inflows!(Makie.Axis(fig[4, 2]), sol)
    plot_topology!(Makie.Axis(fig[4, 3:4]), sol)

    return fig
end
