"""
    HydroModelsViz

Makie-based visualization layer for the HydroModels meta-package. The
public API consists of:

- `plot_dashboard(sol)` — build a 4×3 `Figure` with 11 panels
  (dispatch, storage, pumps, spill, prices, revenue, reserve
  allocation, reserve groups, water value, inflows, cascade topology)
  plus a KPI header.
- Individual `plot_*!(ax, sol; …)` functions — render a single panel
  into a user-provided `Makie.Axis`. Call these directly when you want
  a custom layout.

The package depends only on `Makie` (the recipe API). Loading
`CairoMakie`, `GLMakie`, or `WGLMakie` activates the matching extension
which provides backend-specific helpers (`save_dashboard`,
`dashboard_html`, …). Direct `Makie.save(path, fig)` always works.

Reference: the depr `HydroModels_depr/src/plotting.jl` PlotlyJS
dashboard is the functional spec. The Makie port mirrors the same 11
panels with the same data → glyph mapping; layout and styling are
Makie-idiomatic and intentionally simpler than the elaborate Plotly
hover/tooltip set-up.
"""
module HydroModelsViz

using Dates
using Printf

using HydroModelsCore
using HydroModelsOpt
using Makie

# Backend-presence markers — extensions flip these `Ref`s in their
# `__init__`. Mutation avoids the Julia 1.12 method-overwrite ban that
# applies to extensions redefining a base-package method body.
const _CAIROMAKIE_LOADED = Ref(false)
const _GLMAKIE_LOADED = Ref(false)
const _WGLMAKIE_LOADED = Ref(false)

_has_cairomakie() = _CAIROMAKIE_LOADED[]
_has_glmakie() = _GLMAKIE_LOADED[]
_has_wglmakie() = _WGLMAKIE_LOADED[]

include("helpers.jl")
include("panels/dispatch.jl")
include("panels/storage.jl")
include("panels/reserves.jl")
include("panels/markets.jl")
include("panels/water_value.jl")
include("panels/topology.jl")
include("dashboard.jl")

export plot_dispatch!, plot_storage!, plot_pumps!, plot_spill!
export plot_prices!, plot_revenue!
export plot_reserves_alloc!, plot_reserves_groups!
export plot_water_value!, plot_inflows!, plot_topology!
export plot_dashboard

end # module HydroModelsViz
