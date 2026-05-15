"""
HiGHS solver extension — provides a convenience JuMPSolver factory.
"""
module HydroModelsOptHiGHSExt

using HydroModelsOpt
using HiGHS

HydroModelsOpt.default_lp_solver() = HydroModelsOpt.JuMPSolver(HiGHS.Optimizer)

end # module
