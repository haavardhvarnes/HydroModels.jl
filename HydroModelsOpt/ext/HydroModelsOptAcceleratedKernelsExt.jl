"""
AcceleratedKernels.jl extension — gives backend-agnostic high-level
parallel primitives (foreachindex, sum, reduce, sort, scan, ...).

When loaded, HydroModelsOpt's internal aggregation routines prefer AK
functions over hand-rolled kernels.
"""
module HydroModelsOptAcceleratedKernelsExt

using HydroModelsOpt
using AcceleratedKernels

# TODO: override sum/reduce paths in solvers/lagrangian/* to use AK.

end # module
