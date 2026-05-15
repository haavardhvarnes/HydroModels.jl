"""
High-level constructors for common problem patterns. These are the user-
facing builders that handle topology resolution and default-population.
"""

# TODO: convenience constructor that takes raw modules and infers the
# topology + sensible defaults. e.g.,
#
#   prob = long_term_problem(modules; num_stages=156,
#                            uncertainty=ProdRiskUncertainty(...))
#
# For now, users construct LongTermHydroProblem directly with a pre-built
# topology.
