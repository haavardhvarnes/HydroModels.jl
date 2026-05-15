#!/usr/bin/env julia
#
# Development setup script for the HydroModels monorepo.
# Run from the repo root:   julia --project=. dev.jl
#

using Pkg

# Order matters: Core must be developed before anything that depends on it.
packages = [
    "HydroModelsCore",
    "HydroModelsOpt",
    "HydroModelsData",
    "HydroModelsForecast",
    "HydroModelsViz",
    "HydroModels",
]

println("Setting up HydroModels development environment...")

Pkg.develop([PackageSpec(path = p) for p in packages])
Pkg.instantiate()

println("Done. `using HydroModels` to load the umbrella API.")
