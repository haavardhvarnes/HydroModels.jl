"""
Combined SDP/SDDP algorithm — the Nordic ProdRisk approach.

Reference: Gjelsvik, A., Belsnes, M.M., Haugstad, A. (1999) "An algorithm
for stochastic medium-term hydrothermal scheduling under spot price
uncertainty", PSCC. See also Gjelsvik, Mo, Haugstad (2010), Springer
Handbook of Power Systems I, ch. 2.

# Key algorithmic ideas
1. The forward simulation uses historical inflow + price time series
   directly. This preserves the empirical inflow-price correlation.
2. The backward recursion uses a PAR(1) inflow model plus a *discretized*
   price representation (price nodes per stage with empirical transition
   probabilities).
3. Cuts are stored per (stage, price_node) pair. The cut applicable in
   the forward simulation is selected based on the realized price.

This is structurally different from textbook SDDP and from SDDP.jl's
MarkovianPolicyGraph; the asymmetry between forward and backward is the
defining feature.
"""

# TODO: solve(::LongTermHydroProblem{T, ProdRiskUncertainty{T}}, ::SDDPHydroSolver)
#
# Algorithm sketch:
#   1. Initialize cuts (empty per (stage, price_node))
#   2. For iter in 1:max_iterations:
#      a. Forward: for each sampled forward scenario,
#         - walk forward through historical (inflow, price) series
#         - at each stage, solve LP with current cuts for the price node
#           closest to (or interpolated between) the realized price
#         - record reservoir trajectories and incurred costs
#      b. Backward: for each stage from T down to 1,
#         - for each price node at the stage,
#           for each visited reservoir state in forward,
#             solve the LP using sampled next-stage realizations
#             (PAR(1) inflow + Markov price transitions), generate cut
#      c. Check convergence: compare forward expected cost to
#         backward bound
#   3. Final simulation with full cut set
