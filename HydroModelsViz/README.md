# HydroModelsViz

Visualization layer of the **HydroModels** meta-package. **Placeholder —
no content yet.**

Planned to be built on **Makie**:

- `CairoMakie` for static SVG/PDF (papers, reports) — via weakdep + extension
- `GLMakie` for native interactive — via weakdep + extension
- `WGLMakie` for browser-embedded dashboards — via weakdep + extension

PlotlyJS and Plots.jl are **not** used. The 11-panel system dashboard
from `HydroModels_depr` is the functional spec; the implementation
will be rewritten using Makie recipes when the viz layer takes shape.
