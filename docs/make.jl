using Documenter
using HydroModels
using HydroModelsCore
using HydroModelsOpt

makedocs(
    sitename = "HydroModels.jl",
    modules = [HydroModels, HydroModelsCore, HydroModelsOpt],
    pages = [
        "Home" => "index.md",
        "Types" => "types.md",
        "Solvers" => "solvers.md",
        "References" => "references.md",
    ],
)
