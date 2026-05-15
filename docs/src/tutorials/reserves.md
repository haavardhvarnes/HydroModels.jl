# Reserves — eight Nordic products with soft-slack penalties

The LP baseline supports all eight canonical Nordic reserve products:

| Product | Direction | Symmetric? |
|---|---|---|
| `FCR_N_UP` | up | yes (counts both ways) |
| `FCR_N_DOWN` | down | yes |
| `FCR_D_UP` | up | no |
| `FCR_D_DOWN` | down | no |
| `FRR_UP` | up | no |
| `FRR_DOWN` | down | no |
| `RR_UP` | up | no |
| `RR_DOWN` | down | no |

FCR_N is symmetric — its allocation counts against both the up and
down joint-headroom budgets. The others are uni-directional.

This tutorial uses the shipped `minimal_with_reserves.yaml` fixture,
which adds reserve specs to the basic 2-reservoir cascade.

## Setup

```julia
using HydroModelsOpt
using HydroModelsData
using HiGHS

yaml_path = joinpath(dirname(pathof(HydroModelsData)),
                     "..", "test", "data", "minimal_with_reserves.yaml")
parsed = read_shop_yaml(yaml_path)
prob   = ShopShortTermProblem(parsed)
sol    = solve(prob, JuMPSolver(HiGHS.Optimizer))
```

## Inspect reserve allocations

The per-unit allocation table:

```julia
using DataFrames
df_alloc = DataFrame(sol.r_alloc)
# columns: (time, unit, product, R)
```

Group-level slack — the soft-penalty variable that absorbs
infeasible obligations:

```julia
df_slack = DataFrame(sol.r_slack)
# columns: (time, group, product, R_slack)
```

If `sum(df_slack.R_slack) ≈ 0` the group obligations are met without
slack; otherwise the LP accepted slack at some timesteps.

## FOS §8 joint headroom

For every generator at every timestep the LP enforces:

```
Pg + sum(up_reserves)   <= Pmax    (upper joint-headroom bound)
Pg - sum(down_reserves) >= 0       (lower joint-headroom bound)
```

`FCR_N_UP` and `FCR_N_DOWN` count on *both* sides because they're
symmetric. The remaining products count only on the corresponding
side.

## Sparse-time obligations (an inherited quirk)

Group obligations are **sparse-assigned** to the model time grid:
only the explicit timestamps in the YAML's
`group.<name>.obligation:` block carry a non-zero value. All other
timesteps default to zero, meaning the group has no obligation
there.

This is in contrast to unit reserve **schedules**, which *do* use
LOCF (last-observation-carried-forward) extrapolation. The asymmetry
is inherited from the depr Plotly dashboard and preserved here for
parity.

**Practical takeaway**: if you want a group obligation to bind
across the horizon, populate `obligation` at every timestep
explicitly. The shipped `minimal_with_reserves.yaml` does this.

## Per-unit `pmax = 0` means the unit doesn't offer that product

If a `ReserveSpec` on a generator has `pmax = 0`, the LP's reserve
variable for that `(unit, product)` pair has no upper bound, no
objective coefficient, and no constraint contributions. The LP
solver's presolver eliminates it. This is the canonical way to say
"this unit cannot deliver FCR_D, period."

## Penalty semantics

Group obligations have a slack variable `Rslack[g, j] >= 0`; the
objective subtracts `penalty[g, j] · dt[j] · Rslack[g, j]`.

- `penalty = 0` makes the slack free (the LP can ignore the
  obligation).
- A high penalty (e.g. 1000 EUR/MW·h) turns the obligation into an
  effective hard constraint, with the slack remaining only as a
  safety vent for genuinely infeasible periods.

Set the penalty in the YAML's `group.<name>.penalty` block.

## See also

- [LP baseline tutorial](lp_baseline.md) for the basic dispatch
  pipeline.
- [Visualisation tutorial](visualisation.md) — the dashboard includes
  dedicated `plot_reserves_alloc!` and `plot_reserves_groups!` panels.
