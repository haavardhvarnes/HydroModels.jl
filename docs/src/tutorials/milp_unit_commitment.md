# MILP unit commitment — reversible pump-turbine plants

When a plant owns both a generator and a pump (a reversible pump-
turbine), the LP gets a binary `mode[plant, t]` variable that forbids
simultaneous generation and pumping. The result is a MILP rather
than an LP.

This tutorial uses the shipped `pumped_storage.yaml` fixture, which
has a single plant with both a generator and a pump and a 4× price
ratio between cheap and peak hours — enough that the optimal
schedule arbitrages by pumping at the cheap hour and generating at
the peak.

## Setup

```julia
using HydroModelsOpt
using HydroModelsData
using HiGHS

yaml_path = joinpath(dirname(pathof(HydroModelsData)),
                     "..", "test", "data", "pumped_storage.yaml")
parsed = read_shop_yaml(yaml_path)
prob   = ShopShortTermProblem(parsed)
```

## Detect MILP up front

```julia
is_milp(prob)
# true
```

`is_milp(prob)` returns `true` iff any plant in `prob` owns both a
generator and a pump. Use it to pick a MIP-capable optimizer.

## Solve

```julia
solver = is_milp(prob) ? JuMPSolver(HiGHS.Optimizer) : default_lp_solver()
sol = solve(prob, solver)
```

The LP build automatically tightens the HiGHS tolerances for the MILP
path:

- `mip_rel_gap = 0.001`
- `mip_feasibility_tolerance = 1e-4`

These overrides are necessary because water-value cut slopes routinely
hit O(1e8) EUR/Mm³ while flow variables are O(1e-3) — the default
1e-7 primal-feasibility tolerance is too tight on this dynamic range.

## Inspect the arbitrage

The optimal schedule for `pumped_storage.yaml` pumps at the first
two cheap-priced timesteps and generates at the peak. The storage
trajectory of the upper reservoir rises during pumping and falls
during generation:

```julia
using DataFrames
df = DataFrame(sol.storage)
df_upper = filter(:reservoir => ==( "upper"), df)
df_upper.S
# 24-element Vector{Float64}:
#  50.0
#  60.0     ← pumping
#  70.0     ← pumping
#  …
```

The pump dispatch table reflects the same schedule:

```julia
DataFrame(sol.e_pump)
# row containing positive E entries at the cheap hours, zero elsewhere
```

## Forbidden states

The binary `mode[plant, t]` variable forces `Pg[u, t] <= Pmax_u · mode[plant, t]`
for every generator `u` of the plant, and `Ppump[v, t] <= Pmax_v · (1 - mode[plant, t])`
for every pump `v` of the plant. So either generation or pumping is
active in any timestep, never both.

If you want to relax this (e.g. for a generator-only plant that
shares an intake with a pump in another plant), restructure the YAML
so that the generator and pump live in *different* plants — the
binary lock applies per plant.

## What's not yet wired

The current MILP path covers binary mode locking only. The following
extensions are planned and tracked in
[Status](../status.md) under B1.5–B1.7:

- **Startup costs** — `Generator.startcost` / `Pump.startcost` are
  parsed but not yet incorporated as objective penalties. Adding
  these requires binary commitment variables `u[unit, t]` and
  startup variables `su[unit, t] >= u[t] - u[t-1]`.
- **Minimum on/off times** — the Core schema does not yet carry
  `min_uptime` / `min_downtime` fields. Adding them is a small
  schema change with downstream parser and LP implications.

Both are tracked under milestone B1.5a / B1.5b.
