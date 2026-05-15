# Status

The roadmap is split into three mega-phases (A / B / C), each with
3–5 milestones. Below is the current state. Detail planning notes
live in `/Users/havard/.claude/plans/` and are out of repo scope.

## Mega-phase A — feature parity with the deprecated scaffold ✅

| Milestone | Description | Status |
|---|---|---|
| A1 | Expanded domain types in `HydroModelsCore` (`Plant`, `Reservoir`, `Generator`, `Pump`, `Tunnel`, `PWL`, reserves, time-series) | **landed** |
| A2 | Data layer activation: SHOP YAML reader (`read_shop_yaml`) | **landed** |
| A3 | LP baseline solver — JuMP + HiGHS, `HydroSolution` with Tables.jl outputs | **landed** |
| A4 | Viz layer with Makie — 11-panel dashboard, weakdeps for CairoMakie / GLMakie / WGLMakie | **landed** |
| A5 | Reserves with soft slack penalties (8 Nordic products) | **landed** |

Release point: `v0.1.0` — "depr-parity LP for SHOP-style short-term
scheduling." That is what is registered on the private registry today.

## Mega-phase B — MILP and stochastic

| Milestone | Description | Status |
|---|---|---|
| B1 | MILP unit commitment — binary mode for reversible pump-turbines | **landed** |
| B1.5 | Spill routing into the topological downstream | **landed** |
| B1.6 | Multi-source plants (vector-valued `Generator.from_res` / `Pump.from_res`) | **landed** |
| B1.7 | Typed `Junction{T}` (KP) with `max_flow` (hard) + `flow` (soft) constraints | **landed** |
| B1.5a | Startup costs (`Generator.startcost` / `Pump.startcost`) | planned |
| B1.5b | Minimum on/off times (new Core schema fields) | planned |
| B2 | SDDP via `SDDP.jl` extension — `StagewiseIndependent` dispatch | **landed** |
| B2.5 | SDDP — `ScenarioTree` + `MarkovianUncertainty` dispatch | planned |
| B3 | Long-term → short-term toolchain — `HydropowerToolchain`, `run_toolchain` | **landed** |
| B3.5 | Robust `WaterValueCuts` JSON round-trip via SDDP.write_cuts_to_file | planned |

Release point after B: `v0.2.0` — "MILP + standard stochastic
scheduling via SDDP.jl."

## Mega-phase C — forecasting and research differentiation

| Milestone | Description | Status |
|---|---|---|
| C1 | Forecasting layer activation — `PARProcess`, `MarkovPriceModel`, scenario generation | **landed** |
| C2 | `ProdRiskUncertainty` SDDP path (via SDDP.MarkovianGraph, **symmetric**) | **landed** |
| C2.5 | Native Nordic SDP/SDDP — **asymmetric** forward/backward sampling, PAR(1)-in-the-backward (Gjelsvik 1999) | planned |
| C3 | GPU Lagrangian — scenario decomposition, LP path | **landed** |
| C3.5 | GPU Lagrangian — DP path via KA Bellman kernels, Float32-on-Metal demotion | **landed** |
| C3.6 | Sophisticated primal recovery + `BufferReservoir` subproblem form | planned |
| C3.7 | Bundle-method master problem (replaces vanilla subgradient) | planned |

Release point after C: `v0.3.0` — "Nordic SDP/SDDP + GPU Lagrangian;
research-grade differentiator."

## Out-of-roadmap

- **Documentation deployment** — the docs build locally but are not
  yet hosted on GitHub Pages. Deferred until CI lands.
- **CI** — deliberately deferred. Local tests are green on macOS
  with Julia 1.11 and 1.12.
- **Public benchmark suite** — NO5 PreSpot cases are proprietary
  and only available locally via `HYDROMODELS_TEST_DATA_DIR`. A
  sanitised public benchmark suite is out of scope for now.
- **Multi-destination plants** (`Generator.to_res::Vector{String}`)
  — no real case has surfaced one yet.
- **Multi-source efficiency curves** — current model uses a single
  PQ curve per generator referenced to a single `head_reference`
  reservoir.
- **EMPS v10 file reader** — planned but not yet implemented. The
  ReSDDP-inspired format is documented in `CLAUDE.md`.

## Verification matrix

| Path | Test | Status |
|---|---|---|
| LP baseline on `minimal.yaml` | `HydroModelsOpt/test/test_lp_baseline.jl` | green |
| MILP on `pumped_storage.yaml` | `HydroModelsOpt/test/test_milp.jl` | green |
| Reserves on `minimal_with_reserves.yaml` | `HydroModelsOpt/test/test_reserves.jl` | green |
| Spill routing on `cascade_spill.yaml` | `HydroModelsOpt/test/test_spill_routing.jl` | green |
| Multi-source on `multi_source.yaml` | `HydroModelsOpt/test/test_multi_source.jl` | green |
| Junction min-flow on `junction.yaml` | `HydroModelsOpt/test/test_junction.jl` | green |
| SDDP StagewiseIndependent | `HydroModelsOpt/test/test_sddp.jl` | green |
| SDDP ProdRiskUncertainty | `HydroModelsOpt/test/test_sddp_prodrisk.jl` | green |
| Long → short toolchain | `HydroModelsOpt/test/test_toolchain.jl` | green |
| Lagrangian LP + DP | `HydroModelsOpt/test/test_lagrangian.jl` | green |

## Operational impact logs

Benchmark / parity logs from real-data smoke tests live in
[`benchmarks/`](https://github.com/haavardhvarnes/HydroModels.jl/tree/main/benchmarks):

| Log | Captures |
|---|---|
| `benchmarks/spill_routing.txt` | NO5 PreSpot before/after for B1.5 |
| `benchmarks/multi_source.txt` | NO5 PreSpot before/after for B1.6 |
| `benchmarks/junction.txt` | NO5 PreSpot before/after for B1.7 |
| `benchmarks/c3_5_metal.csv` | Lagrangian DP CPU vs Metal sweep |
