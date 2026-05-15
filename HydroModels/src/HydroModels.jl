"""
    HydroModels

Umbrella package for the HydroModels meta-package — re-exports the
public API of every functional subpackage so that `using HydroModels`
brings the full optimization workflow into scope.

Re-exported subpackages:

- `HydroModelsCore` — physical types, problem formulations, uncertainty
  model containers, topology builders, backend abstraction, generic
  utilities.
- `HydroModelsOpt` — SDDP, SLP/SHOP, GPU Lagrangian, and JuMP-backed
  solvers, plus uncertainty fitting and the long-term → short-term
  toolchain.

Stub subpackages (`HydroModelsData`, `HydroModelsForecast`,
`HydroModelsViz`) are not re-exported until they contain content.

See `CLAUDE.md` at the repo root for architectural principles and the
list of canonical references.
"""
module HydroModels

using Reexport

@reexport using HydroModelsCore
@reexport using HydroModelsOpt

end # module HydroModels
