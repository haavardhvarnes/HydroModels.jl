"""
Markovian SDDP — corresponds to SDDP.jl's MarkovianPolicyGraph.

When the `SDDP` extension is loaded, this delegates to SDDP.jl. Otherwise
implements a minimal native version.
"""

# TODO: solve(::LongTermHydroProblem{T, MarkovianUncertainty{T}}, ::SDDPHydroSolver)
