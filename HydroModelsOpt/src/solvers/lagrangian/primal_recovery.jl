"""
Primal recovery — the Lagrangian dual gives a lower bound. To get an
implementable primal solution we need a recovery heuristic.

Options:
- Round-and-repair: take the Lagrangian-relaxed primal, project onto
  feasibility, repair violations.
- Restoration LP: re-solve a small feasibility LP fixing certain decisions.
- Subgradient-based primal averaging (Larsson-Patriksson).
"""

# TODO: recover_primal(prob, λ, x_relaxed; method=:round_and_repair)
