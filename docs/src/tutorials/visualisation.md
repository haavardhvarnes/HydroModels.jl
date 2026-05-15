# Visualisation — Makie dashboards

`HydroModelsViz` ships an 11-panel Makie dashboard built around the
short-term `HydroSolution` returned by `solve`. The shape mirrors
the `HydroModels_depr` Plotly dashboard but is rewritten in Makie
recipes.

## Setup

```julia
using HydroModelsOpt
using HydroModelsData
using HydroModelsViz
using HiGHS

# Pick a Makie backend
using CairoMakie         # static SVG / PDF / PNG, headless-friendly
# using GLMakie          # native interactive
# using WGLMakie         # browser-embedded
```

## Solve something

```julia
yaml_path = joinpath(dirname(pathof(HydroModelsData)),
                     "..", "test", "data", "minimal.yaml")
parsed = read_shop_yaml(yaml_path)
prob   = ShopShortTermProblem(parsed)
sol    = solve(prob, JuMPSolver(HiGHS.Optimizer))
```

## Build the dashboard

```julia
fig = plot_dashboard(sol)
save("dashboard.png", fig)
```

The dashboard is a `4×3` `Figure` with eleven panels plus a KPI
header:

| Row | Panels |
|---|---|
| 1 (KPI header) | objective, total revenue, total generation, total spill, group-slack flag |
| 2 | dispatch · storage · pumps |
| 3 | spill · prices · revenue |
| 4 | reserve allocation · reserve groups · water value |
| 5 | inflows · topology DAG |

## Per-panel builders

If you want a different layout — say, just the dispatch + storage
side-by-side — call the individual `plot_*!(ax, sol; …)` functions
into your own `Axis`:

```julia
fig = Figure(size = (1100, 400))
ax_disp = Axis(fig[1, 1]; title = "Dispatch")
ax_stor = Axis(fig[1, 2]; title = "Storage")
plot_dispatch!(ax_disp, sol)
plot_storage!(ax_stor, sol)
fig
```

The eleven panel builders are:

- `plot_dispatch!`
- `plot_storage!`
- `plot_pumps!`
- `plot_spill!`
- `plot_prices!`
- `plot_revenue!`
- `plot_reserves_alloc!`
- `plot_reserves_groups!`
- `plot_water_value!`
- `plot_inflows!`
- `plot_topology!`

All mutate their `Axis` argument and return it (Makie idiom).

## Topology DAG

The topology panel draws the cascade as a directed graph with edges
for tunnels, generators, pumps, and spill routing. Multi-source
generators emit one edge per source reservoir. After B1.6 the topology
layout reflects all the per-source connections, not just the highest-
capacity source.

## Backend choice

| Backend | Best for | Output |
|---|---|---|
| `CairoMakie` | papers, reports, CI screenshots | SVG / PDF / PNG |
| `GLMakie` | native interactive exploration | OS window |
| `WGLMakie` | browser-embedded dashboards | self-contained HTML |

Each backend activates an extension on `HydroModelsViz` that adds
backend-specific export helpers. The basic `Makie.save(path, fig)`
call always works regardless of backend.

## Headless rendering

`CairoMakie` is the right choice for headless renders (CI, Docker,
remote SSH). For `WGLMakie` browser exports, the
`HydroModelsVizWGLMakieExt` extension exposes `dashboard_html(fig,
path)` to write a self-contained HTML file.

## See also

- [Viz API reference](../api/viz.md) for the per-function signatures.
- The `HydroModels_depr/src/plotting.jl` Plotly dashboard is the
  functional spec the Makie port mirrors.
