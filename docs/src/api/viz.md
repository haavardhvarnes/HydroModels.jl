# `HydroModelsViz` — API reference

The visualisation layer ships a Makie-based 11-panel dashboard plus
per-panel `plot_*!(ax, sol; …)` helpers for custom layouts.

The package depends only on `Makie` (the recipe API). Loading
`CairoMakie`, `GLMakie`, or `WGLMakie` activates the matching
extension which provides backend-specific helpers (e.g.
`save_dashboard`, `dashboard_html`); direct `Makie.save(path, fig)`
always works.

## Dashboard

```@docs
plot_dashboard
```

## Per-panel builders

Each takes a `Makie.Axis` (or `Makie.Axis3`) as first argument and
mutates it. Returning the axis is idiomatic Makie.

```@docs
plot_dispatch!
plot_storage!
plot_pumps!
plot_spill!
plot_prices!
plot_revenue!
plot_reserves_alloc!
plot_reserves_groups!
plot_water_value!
plot_inflows!
plot_topology!
```

## Picking a backend

| Backend | Best for | Output |
|---|---|---|
| `CairoMakie` | papers, reports | static SVG / PDF / PNG |
| `GLMakie` | native interactive exploration on the laptop | OS window |
| `WGLMakie` | browser-embedded dashboards | self-contained HTML |

```julia
using HydroModelsViz, CairoMakie

fig = plot_dashboard(sol)
save("dashboard.png", fig)
```

```julia
using HydroModelsViz, WGLMakie

fig = plot_dashboard(sol)
# the WGLMakie extension provides `dashboard_html(fig, path)` for a
# self-contained HTML export
```
