# Guidance for Claude Code in HydroModelsData

Extends [/CLAUDE.md](../CLAUDE.md). This file narrows scope for work
inside this subpackage.

## Status

**Active** — SHOP / Harmonie YAML reader landed in milestone A2 of the
phased plan. EMPS v10 reader and HDF5 results I/O remain on the
roadmap; they have not been written yet.

## Scope

This package owns:

- **SHOP / Harmonie YAML reader** — [`read_shop_yaml`](src/parsing/shop_yaml.jl).
  Parses the YAML used by SINTEF SHOP and Eidsiva Harmonie into
  `HydroModelsCore` types. Production-tested on Norwegian NO3 / NO5
  cascades.
- **Tables.jl interface** on `HydroModelsCore.InflowSeries`,
  `HydroModelsCore.MarketSeries`, and on the long-format builders
  [`inflow_table`](src/tables.jl) / [`market_table`](src/tables.jl) /
  [`reserve_obligation_table`](src/tables.jl).
- *(planned)* **EMPS v10 reader** — written from scratch when needed;
  do **not** lift code from SINTEF's GPLv3
  [ReSDDP](https://gitlab.sintef.no/energy/res100/resddp)
  `reademps.jl`.
- *(planned)* **HDF5 results I/O** — for bulk numerical output from the
  optimization layer. Add `HDF5` to `Project.toml` then; not added yet.

## Out of scope

- Inflow / price model fitting → `HydroModelsForecast` (still a stub).
- Anything that solves an optimization problem → `HydroModelsOpt`.
- Plotting → `HydroModelsViz` (still a stub).

## Output convention

Solution / forcing tables produced by this package declare the
**Tables.jl** interface. We do **not** return `DataFrames.DataFrame`
objects. Users who want `DataFrame`s convert at the boundary:
`DataFrame(inflow_table(reservoirs))`. If internal `groupby` / `join`
ever becomes necessary, add `DataFrames` as a **weakdep + extension** —
never as a regular dep.

## Dependencies allowed

Regular deps:

- `HydroModelsCore` — produces all Core types directly.
- `YAML` — file parsing.
- `OrderedCollections` — insertion-ordered time series.
- `Tables` — interface for `InflowSeries`, `MarketSeries`, and the
  long-format builders.
- Stdlibs: `Dates`.

Forbidden in Data:

- `JuMP`, any solver dep — `HydroModelsOpt` territory.
- `DataFrames` as a regular dep (weakdep + extension only if needed).
- Any plotting library.
- `HDF5` until it is actually used by the I/O code paths.

## Test-data convention

**Public fixture (checked in)**: `test/data/minimal.yaml` is a small
synthetic two-reservoir cascade that exercises the main parser
pathways (plants, reservoirs with `vol_head` / `hrl` / `lrl` / `inflow`,
generator with PQ curve, three tunnels of which one survives emission,
market series, time-resolution schedule). It is safe to publish.

**Local real-data fixtures (not committed)**: the three NO5 cases under
`HydroModels_depr/data/real/` are proprietary. Tests that want to
exercise them read the directory from the `HYDROMODELS_TEST_DATA_DIR`
environment variable. The test file's "Real-data smoke test" testset
skips gracefully when the variable is unset or the directory is empty.
The root `.gitignore` blocks `data/real/`, `**/Cases_NO[1-9]*.yaml`,
and `**/Cases_PRE-*`, `**/Cases_POST-*` patterns so accidental copies
don't leak.

## Pitfalls

1. **Empty / degenerate curves**. `HydroModelsCore.PiecewiseLinear`
   requires `length(x) >= 2` and sorted abscissae. Depr's bare `PWL`
   did not validate. The parser uses `_try_piecewise` to silently drop
   curves that fail the check rather than crash on malformed input;
   add a warning if the dropped curves matter to your case.
2. **`_ensure_origin` semantics**. Prepends `(0, 0)` only when the
   first abscissa is strictly positive. Curves whose first point is
   already `(0, ·)` pass through unchanged.
3. **Penstock tunnels collapse to self-loops**. After plant→reservoir
   resolution, a tunnel `reservoir → plant` whose target is then
   resolved back to the same reservoir is suppressed (`a == b`). Only
   reservoir-to-reservoir bypass tunnels survive emission. This is
   intentional — the LP builder accounts for penstock flow inside the
   generator dispatch, not via a separate Tunnel object.
4. **Capacity-weighted upstream inference**. `_bfs_nearest` is called
   with a capacity dict on the reverse graph so a large headwater is
   preferred over a small forebay when both are reachable through KP
   waypoints. Don't change this without understanding why depr added
   it (the Eidsiva NO5 cascade has 213 Mm³ headwaters and 0.001 Mm³
   forebays sharing KP nodes).
5. **Typed junctions — resolved in B1.7.** Junction points
   ("knutepunkt"/KP in Norwegian) appear in real SHOP YAMLs in two
   ways: (a) declared under `model.river.<name>` with extra fields
   `upstream_elevation`, `max_flow`, `flow` (BergheimKP and
   VangenSeaLevelKP in the NO5 case — both terminal of their
   respective cascades), and (b) referenced only from `connections:`
   with no explicit declaration (~22 implicit KP nodes in NO5:
   GolKP, NesKP, LoggaKP, …). Case (a) is now parsed into
   `Junction{T}` and stored on `parsed.junctions`; the parser filter
   excludes the `b_/f_/w_` prefixed river-segment names. Case (b)
   stays implicit — the existing virtual-tunnel logic handles
   them via `_resolve_sources` / `_resolve_targets` recursion.
   The LP enforces `Qjunc ≤ max_flow` (hard) and `Qjunc + Rj_min ≥
   flow` (soft, large penalty) for every typed junction. The
   `spill_to_reservoir` BFS-fallback inference (B1.5) was widened
   in B1.7 to terminate at `reservoir_names ∪ junction_names` and
   to traverse the **full** connection graph instead of the
   tunnel-resolved subgraph — terminal reservoirs (Limarka,
   Eikrebekken, Strandefjorden, Vassbygdvatn in NO5) now resolve to
   the typed Junction at the end of their chain. See
   `benchmarks/junction.txt` for the operational impact on NO5.
6. **Spill inference uses `_bfs_nearest` to chase through KP nodes**
   (B1.5+). The tier order is: (a) most common `g.to_res` among
   generators originating from the reservoir, (b) most common
   `t.to_node` among virtual tunnels, (c) BFS through `fwd_graph` to
   the nearest reservoir. Tier (c) is what handles cascades whose
   river chain `R → river → KP → river → R'` has KP nodes not
   declared in `model.river:` — the virtual-tunnel emission code at
   lines 627–690 wouldn't otherwise pick those up. See
   `_infer_spill_destinations`.
7. **Multi-source plants — resolved in B1.6.** `Generator.from_res`
   and `Pump.from_res` are now `Vector{String}` and the parser
   populates the full set of upstream intake reservoirs reachable
   through tunnel chains (Hemsil2 → [Eikrebekken, Logga, Ruståni];
   Aurland2HøyeFall → 5 sources, etc.). The LP introduces per-
   `(generator, source)` flow-split variables `Qg_from[u, s, j]` so
   each intake reservoir is debited directly when the plant
   generates; the LP picks the split subject to per-source tunnel-
   chain capacity caps stored in `Generator.source_qmax`.
   `head_reference` defaults to `from_res[1]` (the highest-capacity
   source, also the entry the legacy `incoming_by_plant` singular
   map returns). Phantom virtual reservoir-to-reservoir tunnels —
   the parser previously emitted them for non-dominant sources of
   multi-source plants to fake mass conservation through the
   dominant intake — are now suppressed in the tunnel-emission loop.
   See `benchmarks/multi_source.txt` for the NO5 PreSpot before /
   after diagnostic.
8. **Virtual-river tunnels are reservoir-to-reservoir only.** The
   river-resolution code at `shop_yaml.jl:668–690` emits a `Tunnel`
   only when both source and target resolve to known reservoirs.
   River chains terminating at a plant or at an unresolved KP are
   silently dropped. Plant-as-river-terminal is handled by the
   separate plant-keyed `incoming_by_plant` / `incoming_by_plant_all`
   / `outgoing_by_plant` inference; truly dead-end chains (Limarka →
   BergheimKP with no downstream reservoir declared in the YAML)
   become terminal spillways in the LP. Phantom tunnels from
   non-dominant intakes of multi-source plants (B1.6) are explicitly
   skipped — see Pitfall 7.
9. **No explicit `spill` / `flood_descr` block parsing.** Real SHOP
   YAMLs occasionally carry an explicit per-reservoir spillway
   destination; the parser does not read it. `spill_to_reservoir` is
   always inferred from the connection graph (see Pitfall 6). Adding
   the explicit override is a small parser change — the field would
   bypass the three-tier inference and use the YAML value directly.

## Testing

```
julia --project=. -e 'using Pkg; Pkg.test("HydroModelsData")'
```

The "Real-data smoke test" testset is `@test_skip`'d when
`HYDROMODELS_TEST_DATA_DIR` is unset. Set it to a directory of SHOP
YAMLs to opt in:

```
HYDROMODELS_TEST_DATA_DIR=/abs/path/to/depr/data/real \
  julia --project=. -e 'using Pkg; Pkg.test("HydroModelsData")'
```
