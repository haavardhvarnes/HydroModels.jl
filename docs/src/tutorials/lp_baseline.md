# LP baseline — short-term LP from a SHOP YAML

This tutorial walks through the most common entry point: parse a SHOP
YAML, build a short-term LP, solve with HiGHS, and inspect the
resulting tables.

## Setup

```julia
using HydroModelsOpt    # @reexports HydroModelsCore
using HydroModelsData
using HiGHS
```

`HiGHS` is loaded as a separate package; the
`HydroModelsOptHiGHSExt` extension fires and installs HiGHS as the
default LP solver.

## Parse the YAML

The shipped fixture `minimal.yaml` is a two-reservoir cascade with
one generator per plant. It lives inside `HydroModelsData/test/data/`.

```julia
yaml_path = joinpath(dirname(pathof(HydroModelsData)),
                     "..", "test", "data", "minimal.yaml")
parsed = read_shop_yaml(yaml_path)
```

`parsed` is a `NamedTuple` with the parsed plants, reservoirs,
generators, pumps, tunnels, junctions, reserve groups, market series
and timestep schedule. See [Data API reference](../api/data.md) for
the full field list.

## Build the problem

`ShopShortTermProblem` is the SHOP-style short-term problem
formulation in `HydroModelsCore`:

```julia
prob = ShopShortTermProblem(parsed)
```

This is a pure data step — no solver has been touched yet.

## Solve

```julia
sol = solve(prob, JuMPSolver(HiGHS.Optimizer))
```

The first call compiles the JuMP model build (a few seconds); the
second call onward is fast.

`sol` is a [`HydroSolution{Float64}`](@ref HydroSolution) with:

- `sol.termination::String` — JuMP termination status (`"OPTIMAL"`,
  `"INFEASIBLE"`, etc.).
- `sol.objective::Float64` — objective value (EUR for the LP
  baseline).
- A handful of Tables.jl tables: `storage`, `spill`, `q_gen`,
  `p_gen`, `q_pump`, `e_pump`, `q_tunnel`, `revenue`, `r_alloc`,
  `r_slack`, `water_value`, `junctions`.

## Inspect outputs

Tables are `NamedTuple` of `Vector` columns. They satisfy the
Tables.jl interface so you can convert them to your preferred
container:

```julia
using DataFrames
DataFrame(sol.storage)
# 24×5 DataFrame
#  Row │ time                 reservoir  S          S_violation_up  S_violation_dn
#  ────┼──────────────────────────────────────────────────────────────────────────
#    1 │ 2025-01-01T00:00:00  upper      100.0      0.0             0.0
#    …
```

Or read the columns directly:

```julia
sol.storage.S[1:3]
# 3-element Vector{Float64}:
#  100.0
#   98.5
#   97.0
```

## Total revenue

```julia
sum(sol.revenue.revenue)
```

This matches `sol.objective` up to the sign of the reserve-slack
and storage-violation penalties.

## When does the LP become a MILP?

If any plant owns both a generator and a pump (a reversible pump-
turbine), the problem is automatically a MILP — binary mode
variables forbid simultaneous generation and pumping. Pick a
MIP-capable optimizer up front with:

```julia
solver = is_milp(prob) ? JuMPSolver(HiGHS.Optimizer) : default_lp_solver()
sol = solve(prob, solver)
```

See [MILP unit commitment](milp_unit_commitment.md) for a worked
pumped-storage example.

## What's modelled

The shipped LP includes:

- Storage / head convex combinations on each reservoir.
- Generator PQ curves at the reference head, with slack variables
  for head deviation.
- Pump hydraulic curves and electrical efficiency.
- Passive tunnel routing with capacity limits.
- Mass balance per reservoir per timestep, with variable-duration
  timesteps from `dt_schedule`.
- **Spill routing** — surplus spill flows to the inferred downstream
  reservoir or junction, not into the void (see B1.5 in
  [Status](../status.md)).
- **Multi-source plants** — generators with vector-valued
  `from_res` split their discharge across upstream reservoirs by
  capacity (B1.6).
- **Junctions** — `model.river.*` blocks with `max_flow` (hard cap)
  and `flow` (soft minimum, with a high-penalty slack).
- **End-of-horizon water value** — per-reservoir scalar cuts or
  group-level Benders cuts, attached via
  [`with_end_value_cuts`](@ref with_end_value_cuts).
- **Reserve products** — all eight Nordic products with FOS §8
  joint headroom; group obligations with soft-slack penalties.

See [Reserves tutorial](reserves.md) for the reserve-aware path and
[Toolchain tutorial](toolchain_long_to_short.md) for the long → short
cut bridge.

## Next steps

- [MILP unit commitment](milp_unit_commitment.md) — reversible
  pump-turbine plants.
- [Reserves](reserves.md) — adding reserve obligations with soft
  slack penalties.
- [Visualisation](visualisation.md) — build an 11-panel dashboard
  from this very `sol`.
