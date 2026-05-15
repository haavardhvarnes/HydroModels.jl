"""
GLMakie extension — activates when both `HydroModelsViz` and `GLMakie`
are loaded. Use for native interactive windows:

```julia
using GLMakie
display(plot_dashboard(sol))
```
"""
module HydroModelsVizGLMakieExt

using HydroModelsViz
using GLMakie

function __init__()
    HydroModelsViz._GLMAKIE_LOADED[] = true
end

end # module HydroModelsVizGLMakieExt
