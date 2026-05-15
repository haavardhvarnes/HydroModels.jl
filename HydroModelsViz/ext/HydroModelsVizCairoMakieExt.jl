"""
CairoMakie extension — activates when both `HydroModelsViz` and
`CairoMakie` are loaded. The extension currently only signals the
backend is available; users save to file via the standard Makie API:

```julia
using CairoMakie
fig = plot_dashboard(sol)
save("dashboard.png", fig)   # or .svg / .pdf
```

A future `save_dashboard(sol, path; kwargs...)` one-liner may be added
here once the API stabilises.
"""
module HydroModelsVizCairoMakieExt

using HydroModelsViz
using CairoMakie

# Flip the base-module backend marker when the extension loads. Tests
# use `HydroModelsViz._has_cairomakie() === true` to confirm activation.
function __init__()
    HydroModelsViz._CAIROMAKIE_LOADED[] = true
end

end # module HydroModelsVizCairoMakieExt
