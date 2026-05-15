# Guidance for Claude Code in HydroModelsViz

Extends [/CLAUDE.md](../CLAUDE.md). This file narrows scope for work
inside this subpackage.

## Status

**Active** — Makie-based 11-panel dashboard landed in milestone A4
of the phased plan. The depr Plotly dashboard (1125 lines of
PlotlyJS) was the functional spec; the Makie port mirrors the same
data → glyph mapping with a Makie-idiomatic layout.

## Scope

This package owns:

- 11 panel functions, each rendering one chart into a user-provided
  `Makie.Axis`:
  - `plot_dispatch!`, `plot_storage!`, `plot_pumps!`, `plot_spill!`
  - `plot_prices!`, `plot_revenue!`
  - `plot_reserves_alloc!`, `plot_reserves_groups!`
  - `plot_water_value!`, `plot_inflows!`, `plot_topology!`
- `plot_dashboard(sol; size = (1600, 1800))` — composes all 11
  panels into a 4×3 `Figure` with a KPI header.
- Three weakdep + extension stubs (CairoMakie / GLMakie / WGLMakie)
  that set a `Ref{Bool}` marker on activation. Backend-specific
  helpers (`save_dashboard`, `dashboard_html`) are *not* implemented
  yet — users call `Makie.save("path.png", fig)` directly. The
  marker pattern uses `Ref` mutation rather than method overrides to
  avoid Julia 1.12's method-overwrite ban for extensions.

## Out of scope

- Domain types → `HydroModelsCore`.
- Solver outputs / cut data structures → `HydroModelsOpt`
  (the cut **types** are in Core; the **solver pipeline** is in Opt).
- I/O → `HydroModelsData`.
- Backend-specific helper functions like `save_dashboard(sol, path)`
  or `dashboard_html(sol, path)`. These will be added to the
  extensions once the WGLMakie / Bonito HTML-export API stabilises.

## Plotting stack

- **`Makie`** is the recipe API — regular dep.
- **`CairoMakie`** (static SVG / PDF), **`GLMakie`** (native
  interactive), **`WGLMakie`** (browser-embedded) are weakdeps with
  matching `ext/HydroModelsViz<Backend>Ext.jl` files.
- **No** `PlotlyJS`. **No** `Plots.jl`.

The panel functions use only the `Makie` API (`lines!`, `band!`,
`scatter!`, `barplot!`, `text!`, `Axis`, `Figure`). They don't depend
on any backend. The backend is loaded by the caller; what differs is
the rendering output (raster / vector / interactive).

## Public-API design choices

- **Functions, not `@recipe` macros**. The plan note mentioned
  `@recipe`, but Makie's recipe macros are best for type-driven
  polymorphic plotting (`plot(my_data)` dispatching by data type).
  Our panels are multi-trace stacked / overlaid plots with optional
  filters (`reservoirs = …`, `gens = …`), which is what plain
  `plot_*!(ax, sol; kwargs)` functions handle most naturally. This
  is also how Makie's own gallery examples build dashboards.
- **Each panel renders into a caller-provided `Axis`**. Lets users
  compose custom layouts (e.g. a 1×3 row of three panels for a
  specific report). `plot_dashboard` is just one such layout.
- **Solution carries the originating problem**. `HydroSolution{T}`
  has a `problem::ShopShortTermProblem{T}` field added in A4 so
  panels can reach reservoir bounds, water-value cuts, reserve
  obligations, and topology without re-parsing the YAML.

## Style narrowings

- Use `parse(Makie.Colorant, "#RRGGBB")` for hex colors. Direct
  string colors don't compose with the `(c, α)` alpha-tuple form
  Makie expects.
- Time axes use 1-based integer indices (`1:length(times)`) rather
  than `DateTime` ticks for now — Makie's DateTime axis support
  varies by version and the indices are unambiguous for the LP grid
  (which is uniformly spaced within each `dt_schedule` segment).
  Custom tick labels will arrive when the API is stable.
- Empty panels render a centered grey "no data" placeholder via the
  `_placeholder!` helper rather than raising.

## Testing

```
julia --project=. -e 'using Pkg; Pkg.test("HydroModelsViz")'
```

Tests load CairoMakie (the static-output backend), parse and solve
`HydroModelsData/test/data/minimal.yaml`, render every panel into a
fresh `Axis`, build the full dashboard, and save it to a temporary
PNG to verify Figure serialization. All 28 tests pass on the minimal
fixture.

## Pitfalls

1. **Extension marker pattern**. The `_CAIROMAKIE_LOADED`,
   `_GLMAKIE_LOADED`, `_WGLMAKIE_LOADED` `Ref{Bool}`s are flipped in
   each extension's `__init__`. Don't redefine the `_has_<backend>`
   functions in the extensions — Julia 1.12 forbids extension method
   overwrites and you'd get `Method overwriting is not permitted
   during Module precompilation`.
2. **`plot_topology!` uses a simplified layered layout**. depr's
   `_fig_topology` (~190 lines) computes elevation-aware connected-
   component positions. The current Makie port uses longest-path-
   from-source levels and horizontal distribution within level —
   adequate for cascades up to ~50 reservoirs. Replace with a proper
   force-directed or elevation-aware layout when richer cascade
   visualisation is needed.
3. **Multi-panel `plot_water_value!`**. The depr Plotly version
   makes a subplot grid (one cell per reservoir-with-cuts). The
   Makie port overlays all reservoirs on one axis. For a multi-cell
   layout, call `plot_water_value!` with a single `reservoirs = […]`
   filter and tile axes yourself.