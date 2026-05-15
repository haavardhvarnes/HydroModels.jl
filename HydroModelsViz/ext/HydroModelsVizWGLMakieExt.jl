"""
WGLMakie extension — activates when both `HydroModelsViz` and
`WGLMakie` are loaded. Use for browser-embedded interactive dashboards.
The Plotly self-contained-HTML behaviour from `HydroModels_depr` is
equivalent to WGLMakie + `Bonito.export_static`; the actual one-line
HTML export helper will be added here in a follow-up once the
WGLMakie / Bonito API stabilises across versions.

```julia
using WGLMakie
fig = plot_dashboard(sol)
# manual HTML export with current WGLMakie:
#   using Bonito; export_static("dashboard.html", App(fig))
```
"""
module HydroModelsVizWGLMakieExt

using HydroModelsViz
using WGLMakie

function __init__()
    HydroModelsViz._WGLMAKIE_LOADED[] = true
end

end # module HydroModelsVizWGLMakieExt
