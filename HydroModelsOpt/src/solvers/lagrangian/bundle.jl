"""
Bundle method extension to the vanilla subgradient approach.

Vanilla subgradient is slow and oscillatory. Bundle methods maintain a
piecewise-linear lower approximation of the dual function (a "bundle" of
cuts) and choose step direction via a stabilized master problem.
"""

# TODO: bundle_update!(λ, cuts, stab_center, ...)
