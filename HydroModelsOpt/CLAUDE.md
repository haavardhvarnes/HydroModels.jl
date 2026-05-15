# Guidance for Claude Code in HydroModelsOpt

Extends [/CLAUDE.md](../CLAUDE.md). This file narrows scope for work
inside this subpackage.

## Scope

This package owns:

- Solver algorithms — SDDP family (`SDDPHydroSolver` and its
  variants), SLP/SHOP family (`SLPHydroSolver`,
  `UnitCommitmentMode`, `UnitLoadDispatchMode`), Lagrangian
  (`LagrangianHydroSolver`), and the JuMP baseline (`JuMPSolver`).
- The `AbstractSolver` / `AbstractGPUSolver` hierarchy and the
  solver-side strategy types (`ParallelismMode`,
  `DecompositionStrategy`, `StepSizeRule` + concrete subtypes,
  `compute_step`).
- KA kernels for GPU-friendly solver inner loops
  (`kernels/subgradient.jl`, `kernels/bellman.jl`,
  `kernels/scenario_batch.jl`).
- Uncertainty-model **fitting** (`uncertainty/par_model.jl`,
  `uncertainty/markov_price.jl`,
  `uncertainty/scenario_generation.jl`). These currently live here
  but will migrate to `HydroModelsForecast` when that layer comes
  online.
- Long-term → short-term toolchain
  (`toolchain/longterm_to_shortterm.jl`, `HydropowerToolchain`,
  `run_toolchain`).
- The `solve(::AbstractOptProblem, ::AbstractSolver)` verb and all
  concrete `(Problem, Solver)` dispatches.

Out of scope — redirect to the right sibling:

- Domain types (modules, reservoirs, problem formulations,
  uncertainty containers, cut representations) → `HydroModelsCore`.
- File I/O → `HydroModelsData`.
- Plotting → `HydroModelsViz`.

## Dependencies allowed

Regular deps:

- `HydroModelsCore` (re-exported via `Reexport.@reexport using
  HydroModelsCore`, so a single `using HydroModelsOpt` is enough
  for downstream users).
- `Reexport`, `JuMP`, `MathOptInterface`.
- `KernelAbstractions`, `GPUArrays`.
- Stdlibs: `LinearAlgebra`, `SparseArrays`, `Random`, `Statistics`.

Weakdeps (one extension per, all in `ext/`):

- GPU vendors: `CUDA`, `AMDGPU`, `Metal`, `oneAPI`. Each extends
  `HydroModelsCore.default_backend` (the Core function, not an
  Opt-local one).
- Solver wrappers: `HiGHS`, `MadNLP`, `SDDP`.
- `AcceleratedKernels` — extends internal aggregation paths.

## Extension wiring

- GPU vendor extensions call **`HydroModelsCore.set_default_backend!`**
  from their `__init__` to install `GPUBackend(<Vendor>Backend())` as
  the value returned by `default_backend()`. This is a `Ref`-based
  hook, not a method override — Julia 1.12 bans precompile-time method
  overwriting, so the extension cannot redefine `default_backend()`
  itself. The Ref lives in Core; Opt owns the wiring.
- Solver extensions extend Opt-owned factories (e.g.
  `HydroModelsOpt.default_lp_solver`, declared as `function f end`
  with no methods, and `HydroModelsOpt._backend_eltype` for vendor
  eltype quirks) and Opt-owned solver dispatch methods.
- Extension filenames must be `HydroModelsOpt<Vendor>Ext.jl` and
  must match the `[extensions]` block of `Project.toml` exactly,
  otherwise Pkg's extension system silently fails to load them.

## Pitfalls

1. **All ProdRisk subproblems are soft + penalty.** Do not add hard
   constraints inside subproblems. Doing so makes them infeasible in
   edge cases and destroys the cut interpretation.
2. **ProdRisk forward and backward use different uncertainty.** The
   forward pass samples from historical scenarios; the backward
   recursion uses the fitted PAR(1) + Markov node model. Do not
   conflate them.
3. **Delegate standard SDDP to SDDP.jl.** For scenario-tree and
   stagewise-independent SDDP, route through `HydroModelsOptSDDPExt`.
   Only implement custom variants for the Nordic combined SDP/SDDP
   (Gjelsvik 1999) and the GPU Lagrangian.
4. **Premature MILP.** ProdRisk-class long-term problems are pure
   LPs with piecewise-linear approximations. Integer variables enter
   at SHOP-scale (unit commitment). Do not add binary variables to
   the long-term problem unless explicitly modeling unit-level detail.
5. **Reservoir type mismatch.** `RegulationReservoir` contributes a
   state variable and gets cuts; `BufferReservoir` follows guidelines
   and does *not* get cuts. Mixing them in cut computations is a bug.

## Performance

- Allocations in inner solver loops are forbidden. Use `view`s and
  preallocated buffers. Profile with `BenchmarkTools.@benchmark` and
  `--track-allocation=user`.
- Sparse matrix operations dominate SDDP runtime. On GPU, route
  through cuSPARSE / rocSPARSE via the backend, not through generic
  `SparseMatrixCSC`.
- For GPU kernels, measure occupancy and bandwidth — wall-clock time
  alone is misleading on cold launches.

## Testing

Solver-loadability smoke tests plus the LP-baseline end-to-end suite
(`test/test_lp_baseline.jl`) which loads HiGHS, parses
`HydroModelsData/test/data/minimal.yaml`, builds a
`ShopShortTermProblem`, solves with HiGHS, and asserts `OPTIMAL` +
bounded objective + Tables.jl shape on every output table.

Algorithmic tests for SDDP / SLP / Lagrangian arrive with their
respective milestones. Use the 2-reservoir cascade from
`examples/01_two_reservoir_cascade.jl` and the SDDP.jl example library
as canonical small instances. Run with

```
julia --project=. -e 'using Pkg; Pkg.test("HydroModelsOpt")'
```

from the repo root.

## JuMP baseline (milestone A3 — landed)

The JuMP LP/MILP baseline is the first concrete solver. Dispatch path:

```
parsed = read_shop_yaml(path)           # HydroModelsData
prob   = ShopShortTermProblem(parsed)   # HydroModelsCore type
sol    = solve(prob, JuMPSolver(HiGHS.Optimizer))
```

Returns a `HydroSolution{Float64}` with 11 Tables.jl-compatible output
tables: `storage`, `spill`, `q_gen`, `p_gen`, `q_pump`, `e_pump`,
`q_tunnel`, `revenue`, `r_alloc`, `r_slack`, `water_value`. All tables
are `NamedTuple`s of columns; no DataFrames hard dep — convert at the
boundary with `DataFrame(sol.storage)` if needed.

Internal architecture:

- `src/solvers/jump_baseline.jl` — `_ShortTermLPModel` flat layout
  plus `_build_short_term_lp` (port of `HydroModels_depr/src/build_model.jl`).
  Includes reservoir storage/head convex combinations, generator PQ
  curves with head-clamping slacks, pump hydraulic curves, passive
  tunnel flows, mass balance with variable timestep duration,
  end-of-horizon water-value cuts (per-reservoir scalar form or
  group-level Benders), reserve-product joint headroom (FOS §8), and
  binary mode variables for reversible pump-turbine plants.
- `src/solvers/jump_solve.jl` — `HydroSolution` struct and `solve`
  method plus per-output-table builders.

The internal `_ShortTermLPModel` is **not** exported.

Notable Julia 1.12 conventions:

- `default_lp_solver` is declared as `function default_lp_solver end`
  (no methods). `HydroModelsOptHiGHSExt` adds the zero-arg method
  when `HiGHS` is loaded. This is the canonical way to avoid method-
  overwrite warnings on 1.12+ when an extension provides the
  implementation of a parent-package function.
- HiGHS solver attributes are set by `_build_short_term_lp`:
  `mip_rel_gap = 0.001`, `primal_feasibility_tolerance = 1e-4`,
  `mip_feasibility_tolerance = 1e-4`. Water-value cut slopes can hit
  ~5e8 EUR/Mm³ while flow vars are O(1e-3); the default 1e-7 primal
  tolerance is too tight on this dynamic range.

## Reserve products (milestone A5 — landed)

A5 verified the reserve code path end-to-end via
`test/test_reserves.jl` and a dedicated fixture at
`HydroModelsData/test/data/minimal_with_reserves.yaml`. Key semantics
to keep in mind when extending:

- **Eight canonical products**: `FCR_N_{UP,DOWN}`, `FCR_D_{UP,DOWN}`,
  `FRR_{UP,DOWN}`, `RR_{UP,DOWN}`. `FCR_N` is symmetric (it counts
  against both up and down joint-headroom budgets); the others are
  uni-directional.
- **Soft + penalty everywhere**. Group obligations have a slack
  variable `Rslack[g, j] >= 0`; the objective subtracts
  `penalty[g, j] * dt[j] * Rslack[g, j]`. Setting `penalty = 0`
  makes the slack free (LP can ignore the obligation); setting a
  high penalty (e.g. 1000 EUR/MW·h) effectively turns the obligation
  into a hard constraint, with the slack remaining as a safety vent
  for genuinely infeasible periods.
- **Sparse-time obligations (depr semantics, retained)**. Group
  obligations and penalties are *sparse-assigned* to the model time
  grid — only the explicit timestamps carry a non-zero value, and
  all other timesteps default to zero. This is **not LOCF**, unlike
  unit reserve schedules which *do* use LOCF. The asymmetry is
  intentional in depr; preserved here for parity. To bind a group
  obligation across the horizon, populate the obligation at every
  timestep explicitly (see `minimal_with_reserves.yaml`).
- **FOS §8 joint headroom**. For each generator at each timestep:
  `Pg + sum(up_reserves) <= Pmax`, `Pg - sum(down_reserves) >= 0`.
  `FCR_N_UP` and `FCR_N_DOWN` count on *both* sides because they're
  symmetric products.
- **Per-unit `pmax = 0` means the unit doesn't offer that product**.
  The LP's reserve variable for that `(unit, product)` pair has no
  upper bound, no objective coefficient, and no constraint
  contributions, so the LP solver's presolver eliminates it. No
  correctness impact.

## MILP unit commitment (milestone B1 — landed)

B1 verified the reversible-pump-turbine binary-mode path via
`test/test_milp.jl` and the fixture
`HydroModelsData/test/data/pumped_storage.yaml`. Key points:

- **`is_milp(prob)` predicate** — exported. Returns `true` iff any
  plant in `prob` owns both a generator and a pump. Use it to pick a
  MIP-capable optimizer before solving:

  ```julia
  solver = is_milp(prob) ? JuMPSolver(HiGHS.Optimizer) : default_lp_solver()
  ```

- **Binary `mode[plant, t]` variable**. Added unconditionally in
  `_build_short_term_lp` whenever the reversible-plant detection
  fires. Constraints: `Pg[u, t] <= Pmax_u * mode[plant, t]` for
  generators in that plant and `Ppump[v, t] <= Pmax_v * (1 - mode)`
  for pumps. Forbids simultaneous generation and pumping in the same
  timestep.
- **Tight HiGHS tolerances**: `mip_rel_gap = 0.001`,
  `mip_feasibility_tolerance = 1e-4`. Set by `_build_short_term_lp`.
- **Arbitrage proof point**: the `pumped_storage.yaml` fixture has a
  price ratio of 4× between cheap and peak hours; the LP correctly
  pumps at the cheap hour and generates at the peaks. Storage
  trajectories in `sol.storage` rise during pumping periods and
  decline during generation. A dedicated test asserts this peak-of-
  upper-storage falls in the first two timesteps.

### Deferred (B1.5 follow-up)

The plan also mentions startup costs and minimum on/off times under
B1. These require:

- **Startup costs** — `Generator.startcost` and `Pump.startcost` are
  already parsed; the LP needs binary commitment variables
  `u[unit, t]`, startup variables `su[unit, t] >= u[t] - u[t-1]`
  (with `u[1] = initial_state`), and an objective term subtracting
  `startcost * sum(su)`. For a NO5-scale case (33 generators × 144
  timesteps) this adds ~5 000 binaries and tightens the MILP.
- **Min on/off times** — would require new fields on `Generator` /
  `Pump` (`min_uptime`, `min_downtime`) which the Core types don't
  carry today. Adding these is a small Core schema change that has
  downstream parser and LP implications.

Both extensions fit the same B1 theme but each merits its own focused
milestone (B1.5a, B1.5b) rather than bundling all three into one big
change. The current B1 landed the reversible-plant path cleanly; the
rest will follow when the user prioritises them.

## Spill routing (milestone B1.5 — landed)

Before B1.5, the LP treated `Qspill[i, j]` as a per-reservoir sink:
the mass-balance constraint subtracted spill from reservoir `i`, but
no corresponding `+` term appeared anywhere downstream. Water that
overtopped any upstream dam silently disappeared. This biased
diagnostics on real SHOP cases (NO5 PreSpot reported 3903 m³/s·tsteps
of spill spread across five reservoirs, of which 21 % was rooted at
intermediate nodes whose downstream had spare capacity).

B1.5 fixes this. `Reservoir{T}` (in `HydroModelsCore/src/types/shop.jl`)
gains an optional `spill_to_reservoir::Union{Nothing, String} = nothing`
field. The SHOP parser at
[`HydroModelsData/src/parsing/shop_yaml.jl`](../HydroModelsData/src/parsing/shop_yaml.jl)
infers it after parsing generators and tunnels — the helper
`_infer_spill_destinations(generators, tunnels, reservoir_names)`
picks the most common `g.to_res` first, falling back to the most
common `t.to_node` among river-type tunnels when no generator
originates from the reservoir. Ties on destination name break
lexicographically for determinism. `_break_spill_cycles!` rounds out
the inference defensively — physically-realisable cascades are
acyclic but a malformed input could produce a loop.

In the LP ([`HydroModelsOpt/src/solvers/jump_baseline.jl`](src/solvers/jump_baseline.jl)),
`_ShortTermLPModel` gained two precomputed fields:

- `spill_to_idx::Vector{Int}` — `0` for terminal, else the
  destination reservoir index in `R`.
- `spill_in::Vector{Vector{Int}}` — reverse adjacency:
  `spill_in[d]` lists upstream reservoirs whose spill flows into
  reservoir `d`.

The mass-balance constraint adds a single term:

```julia
+ sum(Qspill[u, j] * conv[j] for u in lp.spill_in[i]; init = 0.0)
```

at both the inner-time and `S_end` constraints. Terminal reservoirs
(empty `spill_in[i]`) retain the previous sink behaviour exactly —
the `init = 0.0` in the JuMP sum handles the empty-source case.

The anti-degeneracy penalty
`- 1e-3 * dt[j] * Qspill[i, j]` stays. A unit of water spilled and
then re-spilled downstream is double-penalised, giving the LP a small
physical-realism preference for turbines / tunnels over multi-hop
spill chains. The 1e-3 weight is small enough not to bias the
revenue objective.

### What the fix does (and doesn't) change

Routing **does not change the objective on the existing fixtures**
(`minimal.yaml`, `pumped_storage.yaml`, `minimal_with_reserves.yaml`,
`pumped_storage.yaml`, `cascade_spill.yaml` for the all-routing path),
because those fixtures' spilled water lands at terminal nodes where
the LP can't generate from it anyway. The NO5 PreSpot smoke (logged
in `benchmarks/spill_routing.txt`) is the same story at production
scale: same objective (613 MEUR), same total spill, but the post-B1.5
"Top spillers" list collapses from five reservoirs to the two true
terminal nodes (Limarka and Eikrebekken).

The fix is a **correctness improvement**, not a performance one. The
diagnostic from `scripts/run_pipeline.jl` is now interpretable as
"where is water actually leaving the system?" rather than the older
"where in the cascade is local capacity binding?" — which makes it a
useful guide to where adding a downstream tunnel in the SHOP topology
would actually capture more water.

### Tests

- [`test_spill_routing.jl`](test/test_spill_routing.jl) — exercises
  the `cascade_spill.yaml` fixture (forced spill at `upper`,
  asserting routed water reaches `lower`'s mass balance) and a
  regression block confirming `minimal.yaml` and
  `pumped_storage.yaml` give bit-identical objectives between the
  routed default and a manually-stripped (`spill_to_reservoir =
  nothing` everywhere) solve.
- The fixture
  [`HydroModelsData/test/data/cascade_spill.yaml`](../HydroModelsData/test/data/cascade_spill.yaml)
  is sized so the upstream reservoir is unavoidably bottlenecked
  (50 m³/s inflow, 5 m³/s turbine + 5 m³/s bypass), making the
  routed-vs-sunk difference visible in the trajectory.

## SDDP.jl bridge (milestone B2 — landed for `StagewiseIndependent`)

B2 wires `LongTermHydroProblem{T, StagewiseIndependent{T}}` through
SDDP.jl's `LinearPolicyGraph` machinery. The extension lives at
[`ext/HydroModelsOptSDDPExt.jl`](ext/HydroModelsOptSDDPExt.jl) and
fires when both `HydroModelsOpt` and `SDDP` are loaded. The
canonical-instance test is at `test/test_sddp.jl`.

### Dispatch

```julia
using HydroModelsOpt, HiGHS, SDDP

prob = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
    modules = [...],
    topology = build_topology([...]),
    num_stages = 12,
    uncertainty = StagewiseIndependent{Float64}(realizations, probabilities),
    stage_prices = [...],          # deterministic per-stage price forecast
)
solver = SDDPHydroSolver(;
    subproblem_solver = JuMPSolver(HiGHS.Optimizer),
    max_iterations    = 200,
    num_forward_scenarios = 100,
)
sol = solve(prob, solver)   # returns Solution{Float64}
```

`sol.objective` is the in-sample mean of the simulated total cost,
`sol.bound` is `SDDP.calculate_bound(sddp_model)`, and the trained
`sddp_model` lives in `sol.metadata[:sddp_model]` for cut export or
warm-starting.

### Unit convention

The bridge treats LP coefficients as unitless. Revenue per stage is

```
stage_prices[t] * sum(energy_factor[i] * turbine[i] for i)
```

with `turbine` in m³/s; storage in Mm³ but treated as direct units of
flow × stage. For Nordic weekly schedules in kWh/m³ × EUR/MWh, the
caller is responsible for unit scaling (pre-multiply `energy_factor`
by 0.168 or similar). The bridge does not enforce a particular unit
system.

### Required Core extension

A new field was added to `LongTermHydroProblem`:

```julia
stage_prices::Union{Nothing, Vector{T}} = nothing
```

`StagewiseIndependent` only carries inflow realizations, so the
deterministic price forecast goes here. `ProdRiskUncertainty` and
`MarkovianUncertainty` carry their own price representations and
should leave `stage_prices = nothing`.

### Other variants — partial coverage

- `LongTermHydroProblem{T, ScenarioTree{T}}` → `SDDP.PolicyGraph`
  (stubbed; B2.5 follow-up)
- `LongTermHydroProblem{T, MarkovianUncertainty{T}}` →
  `SDDP.MarkovianPolicyGraph` (stubbed; B2.5 follow-up)
- `LongTermHydroProblem{T, ProdRiskUncertainty{T}}` →
  `SDDP.MarkovianPolicyGraph` (landed in C2 — see section below)
- `WaterValueCuts{T}` extraction from `sddp_model.nodes[t].bellman_function`
  (currently only the `SDDP.PolicyGraph` is stored in
  `sol.metadata[:sddp_model]`; users can call
  `SDDP.write_cuts_to_file(sddp_model, "cuts.json")` directly)

### Solver-struct cleanup

While wiring B2, three solver structs had a spurious type parameter
that prevented kwarg construction. Fixed in passing:

- `SDDPHydroSolver{R, S}` — was `{T, R, S}` with `T` used only for
  `convergence_tolerance::T = T(1e-3)`. Now `convergence_tolerance::Float64 = 1.0e-3`.
- `LagrangianHydroSolver{B}` — was `{T, B}` with `T` used only for
  `tol::T = T(1e-4)`. Now `tol::Float64 = 1.0e-4`.
- `SLPHydroSolver{S}` — was `{T, S}` with `T` used only for
  `head_tolerance::T = T(0.01)`. Now `head_tolerance::Float64 = 0.01`.

The `T` was never inferrable from any required argument, so the
kwarg constructors threw `UndefVarError: T not defined in
HydroModelsOpt` whenever the user omitted both type parameters and
the tolerance. SDDP and Lagrangian / SLP runs only make sense in
`Float64` anyway (SDDP.jl is `Float64`-only internally), so the
concrete `Float64` is the right choice.

## ProdRiskUncertainty SDDP path (milestone C2 — landed via SDDP.jl)

C2 adds `solve(::LongTermHydroProblem{T, ProdRiskUncertainty{T}},
::SDDPHydroSolver)`. The bridge lives alongside the
`StagewiseIndependent` one in
[`ext/HydroModelsOptSDDPExt.jl`](ext/HydroModelsOptSDDPExt.jl); tests
at `test/test_sddp_prodrisk.jl`.

### Mapping

- **Price-Markov outer structure** —
  `SDDP.MarkovianGraph(transitions)` builds one policy-graph node per
  `(stage, price_node)`. `transitions[1]` is a `(1, K_1)` initial
  distribution (uniform), so SDDP's deterministic root advances into
  the K_1 stage-1 price nodes; `transitions[t]` for `t > 1` is the
  Markov matrix `u.price_transitions[t - 1]`. Stage-1 price node value
  feeds the stage objective.
- **Inflow uncertainty** — at each stage `t`, the realizations are
  the `n_scenarios` historical years embedded in
  `u.historical_inflow[t, :, :]`. Each historical year contributes
  one equiprobable realization. The bridge assumes one gauge per
  module (so `size(historical_inflow, 3) == nmod`).
- **Subproblem skeleton** — state per `RegulationReservoir`, controls
  per module (turbine + spill + inflow placeholder), mass balance
  with upstream routing — identical to the StagewiseIndependent path.
  Stage objective uses the Markov node's price value.

### Deviation from the plan ("natively, not via SDDP.jl")

The plan called for a *native* combined SDP/SDDP per
Gjelsvik / Belsnes / Haugstad 1999, with the **asymmetric
forward/backward sampling** that is the Nordic differentiator:

- Forward: historical scenarios (preserves inflow-price correlation)
- Backward: fitted PAR(1) + Markov

The C2 MVP uses SDDP.jl's **symmetric** Markovian machinery for both
forward and backward (same historical-inflow distribution either
way). This delivers a working `ProdRiskUncertainty` dispatch today
and a clean research playground, at the cost of losing the ProdRisk
asymmetry. A native algorithm with PAR(1)-in-the-backward is **C2.5**
on the roadmap — it would integrate the `HydroModelsForecast.PARProcess`
sampling produced in C1.

The deviation is signalled in the returned solution via
`sol.metadata[:forward_backward_asymmetric] === false` so callers can
detect this path.

### Validation

`_validate_prodrisk` checks:
- `size(historical_inflow)` agrees with `num_stages × _ × num_modules`
- `price_nodes` length matches `num_stages`
- `price_transitions` length matches `num_stages − 1`
- Each `price_transitions[t]` has shape `(K_t, K_{t+1})` and is
  row-stochastic to `1e-6`

If `stage_prices` is set, the bridge warns and ignores it — prices
come from `u.price_nodes` on the Markov node path.

## Long-term → short-term toolchain (milestone B3 — landed)

B3 implements the Nordic-style `long-term SDDP → short-term LP` chain.
Public API lives in
[`src/toolchain/longterm_to_shortterm.jl`](src/toolchain/longterm_to_shortterm.jl).
The canonical-instance test is at `test/test_toolchain.jl`, and a
full-workflow example at `examples/02_toolchain_long_to_short.jl`.

### Dispatch

```julia
tc = HydropowerToolchain{Float64}(
    longterm_model     = LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(...),
    shortterm_template = ShopShortTermProblem(parsed_yaml),
    longterm_solver    = SDDPHydroSolver(; subproblem_solver = JuMPSolver(HiGHS.Optimizer)),
    shortterm_solver   = JuMPSolver(HiGHS.Optimizer),
    reservoir_map      = Dict(:upper => "upper", :lower => "lower"),
)
res = run_toolchain(tc)
# res.long  — Solution{T} from the long-term SDDP solve
# res.short — HydroSolution{T} from the short-term LP with cuts injected
# res.cuts  — Vector{CutGroup{T}} extracted from SDDP and fed in
# res.shortterm_with_cuts — the configured short-term problem
```

### Bypass path (user-supplied cuts)

When cuts come from an external source — a previous ProdRisk run, the
literature, or a synthetic test fixture — supply them directly to
`run_toolchain` and the long-term solve is skipped:

```julia
cuts = synthetic_end_value_cuts(
    ["upper", "lower"],
    [50_000.0, 30_000.0, 70_000.0],
    [1.0e6 0.5e6 1.5e6; 0.5e6 0.25e6 0.75e6],
)
res = run_toolchain(tc; cuts = cuts)
@assert res.long === nothing   # skipped
```

### Cut extraction from SDDP.jl

[`extract_end_value_cuts(sol, long_prob)`](src/toolchain/longterm_to_shortterm.jl)
reaches into SDDP.jl's `nodes[1].bellman_function.global_theta.cut_oracle.cuts`
to pull the stage-1 continuation-value cuts (which serve as the
end-of-horizon water value for a short-term horizon ending just
before the long-term starts). The path through SDDP.jl internals is
**fragile across SDDP.jl versions**; on failure the function warns
and returns an empty cut vector. Callers should detect this with
`isempty(cuts)` and either fall back to `synthetic_end_value_cuts` or
skip the cut injection step.

A robust round-trip — emitting `WaterValueCuts{T}` (Core's richer
multi-stage/multi-price-node form) from `SDDP.write_cuts_to_file`
JSON, and re-reading them on a fresh process — is a B3.5 follow-up.

### Reservoir-name mapping

Long-term `HydroModule.name` is a `Symbol`; short-term
`Reservoir.name` is a `String`. The default empty `reservoir_map`
means "names match after `String(symbol)`" — e.g. `:upper` →
`"upper"`. When the two models use different conventions (e.g.
long-term `:Aurland_upper` vs short-term `"Aurland_HRV_upper"`),
populate `reservoir_map::Dict{Symbol, String}` with the explicit
translations. Cuts whose reservoirs cannot all be resolved against
`prob.reservoirs` are silently dropped (the short-term LP only
enforces cuts on reservoirs it owns).

### Determinism guarantee

The short-term LP is purely deterministic; given the same cuts, the
dispatch is bit-identical across reruns (modulo solver-specific
numerical noise). `test_toolchain.jl` asserts this with `==` on the
storage and dispatch tables across two consecutive `run_toolchain`
calls with synthetic cuts.

## Lagrangian scenario-decomposition solver (milestone C3 — research scaffold)

C3 lands a scenario-decomposed Lagrangian dispatch for
`LongTermHydroProblem{T, StagewiseIndependent{T}}` in
[`src/solvers/lagrangian/scenario_decomposition.jl`](src/solvers/lagrangian/scenario_decomposition.jl).
Tests at `test/test_lagrangian.jl`.

### What it does

1. Samples `solver.n_scenarios` whole-path inflow scenarios from
   `prob.uncertainty.realizations` (cyclic Monte Carlo through the
   per-stage realization list).
2. Dualises **per-reservoir end-of-horizon storage** against a target
   equal to the initial volume. The Lagrangian per scenario is

   ```
   max  Σ_t price[t] · Σ_i η_i · turbine_i,t  −  Σ_i λ_i · S_end_i
   ```

   so the multipliers `λ_i ≥ 0` push final storage upward (toward the
   target) by penalising depletion. (This is *not* a non-anticipativity
   coupling — that would dualise the per-stage state across scenarios.
   The current coupling has a clean interpretation and produces a
   working subgradient method; non-anticipativity is C3.5.)
3. Solves the per-scenario LP via the user-supplied `JuMPSolver`.
4. Aggregates `S_end` across scenarios through a backend-portable
   KernelAbstractions kernel (`_accumulate_columnwise!` → mean). On
   `CPUBackend()` this runs on the host; with a vendor extension
   loaded it dispatches to that backend automatically.
5. Updates multipliers with `compute_step(solver.step_size, …)` and
   loops to `solver.max_iter` or until `maximum(abs, g) < solver.tol`.

### Dispatch

```julia
solver = LagrangianHydroSolver(;
    subproblem_solver = JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)),
    step_size = DiminishingStep(1.0, 0.9),
    max_iter = 100,
    n_scenarios = 16,
)
sol = solve(prob, solver)
# sol.primal :: NamedTuple of (scenario_objectives, scenario_S_end)
# sol.dual   :: NamedTuple of (multipliers, targets)
# sol.metadata[:variant] == :LagrangianStagewiseIndependent
```

The solver requires `prob.stage_prices` to be set (the
StagewiseIndependent uncertainty carries inflows only) and
`solver.subproblem_solver` to be a `JuMPSolver` with a non-`nothing`
optimizer. Both invariants raise `ArgumentError` on the validation
path; non-StagewiseIndependent uncertainty types throw
`ErrorException`.

### Solver-struct change

`LagrangianHydroSolver` gained two fields in C3:

- `subproblem_solver::S = nothing` — wraps the per-scenario LP
  optimizer (currently must be a `JuMPSolver`).
- `n_scenarios::Int = 16` — Monte Carlo path count for the outer
  scenario decomposition.

The struct keeps `{B<:ComputeBackend, S}` parametrisation; `S`
defaults to `Nothing` and the validation step rejects that early.

### Backend-portable aggregation

The mean-of-`S_end` computation goes through KernelAbstractions:

```julia
@kernel function _accumulate_columnwise!(out, X)
    j = @index(Global)
    s = zero(eltype(out))
    @inbounds for i in 1:size(X, 1)
        s += X[i, j]
    end
    @inbounds out[j] = s
end
```

Driven by `_mean_with_backend(backend, X)` which allocates `out` via
`HydroModelsCore.allocate(backend, ...)` and dispatches the kernel
through `ka_backend(backend)`. The same source compiles on
`CPUBackend()`, `GPUBackend(CUDABackend())`, etc.

### Status: research scaffold, not production solver

What's **landed**:
- Per-scenario LP build + solve end-to-end through HiGHS
- KA aggregation that the GPU extensions will exercise once their
  CI hardware is available
- A working subgradient method with three step-size rules
  (`ConstantStep`, `DiminishingStep`, `PolyakStep`)
- Validation and stub dispatches for the other uncertainty types

What's **deliberately deferred to C3.5+**:
- **Bundle method** (`use_bundle::Bool = false` placeholder in the
  struct). Vanilla subgradient is slow and oscillatory; a bundle
  method gives proper proximal updates.
- **Sophisticated primal recovery** (`primal_recovery::Symbol`
  placeholder). Currently the "primal" returned is just the
  per-scenario S_end matrix; recovering a single feasible primal
  schedule from the relaxed dual is its own algorithmic problem.
- **Non-anticipativity decomposition** — the production
  scenario-decomposed Lagrangian dualises per-stage state across
  scenarios, not end-of-horizon storage. That formulation needs cuts
  per scenario subset and is bigger than C3.
- **GPU benchmarks** — the kernel is portable but exercised only on
  `CPUBackend()` in CI. Wiring `HydroModelsOptCUDAExt` to forward a
  `GPUBackend(CUDABackend())` through the same path is straightforward
  but currently has no test hardware.
- **Dispatches for `ScenarioTree`, `MarkovianUncertainty`,
  `ProdRiskUncertainty`** — informative `ErrorException` stubs only.
- **Proper dual bound**. Vanilla subgradient does not produce a
  monotone upper bound; the returned `Solution.bound` is `NaN` and
  `Solution.status` is `MOI.OTHER_LIMIT`. A bundle method with a
  master problem would fix this.

The C3 milestone establishes the **dispatch path and KA portability
contract**, not a competitive Lagrangian implementation. Treat it as
the research playground that downstream work plugs into.

## Lagrangian GPU-native DP path (milestone C3.5 — landed)

C3.5 adds a GPU-native dispatch alongside C3's LP path. Where C3 runs
the per-scenario LP on CPU via HiGHS (~95 % of wall time), C3.5
dualises **both** the end-of-horizon storage targets (`λ`) and the
cross-reservoir mass-balance equalities (`μ`), reducing each
`(scenario, reservoir)` subproblem to a 1-D Bellman DP that runs as a
KA kernel on any backend. On Apple Silicon with Metal loaded, the
default backend resolves to `GPUBackend(MetalBackend())` and the
default `dispatch = :auto` routes through the DP path.

### Dispatch

```julia
solver = LagrangianHydroSolver(;
    # backend defaults to Metal on Apple Silicon when `using Metal`
    step_size = DiminishingStep(1.0, 0.9),
    max_iter = 50,
    n_scenarios = 64,
    K_grid = 64,                       # storage grid points per reservoir
    dual_flow_coupling = true,         # set false to keep C3's λ-only dualisation
    dispatch = :auto,                  # :auto / :lp / :dp
)
sol = solve(prob, solver)
# sol.metadata[:variant]   == :LagrangianDP
# sol.metadata[:dispatch]  == :dp
# sol.dual.λ               :: Vector{Float64}
# sol.dual.μ               :: Array{Float64, 3} or nothing
# sol.primal.S_traj        :: Array{Float64, 3}  (T+1, nmod, n_scen)
# sol.primal.q_traj        ::                      (T,   nmod, n_scen)
# sol.primal.σ_traj        ::                      (T,   nmod, n_scen)
```

`dispatch = :auto` resolves to `:dp` on any `GPUBackend`, `:lp` on
`CPUBackend()`. Explicit `:dp` forces the GPU-native path on CPU
(for cross-validation); explicit `:lp` forces the C3 path on a GPU
backend (for isolating "DP vs LP" effects from the spatial-coupling
dualisation).

### Kernels (in `src/kernels/`)

- `bellman_dp_kernel!` — backward sweep at one stage; ndrange
  `(K, nmod, n_scen)`. Launched once per stage by the host-side loop
  because KA doesn't provide intra-kernel barriers across `ndrange`.
- `bellman_dp_terminal!` — initialises `V_{T+1}(S) = −λ_i · S`.
- `forward_simulate_kernel!` (in `kernels/scenario_batch.jl`) — rolls
  out the optimal policy from `S0`; ndrange `(nmod, n_scen)`.
- `compute_subgradient_flow_kernel!` (in `kernels/subgradient.jl`) —
  computes mass-balance residual `g_μ_{s,i,t}`; ndrange
  `(n_stages, nmod, n_scen)`.
- `subgradient_update_kernel!` (reused from C3) — projected
  subgradient step on `λ` and `μ`.

The action axis collapses: with a fixed target next-state `S_{t+1}`,
mass balance pins `q + σ = S_t + inflow − S_{t+1}`, and the
bang-bang split (`q` up to `q_max` first, then `σ`) is closed-form.
The DP loops only over next-state grid indices — `O(K²)` work per
stage per `(scenario, reservoir)`. **Compute-bound on the inner
reward calculation, not bandwidth-bound.**

### Float32 on Metal

Apple GPUs have no native Float64. The existing `_backend_eltype`
hook in `HydroModelsOptMetalExt` demotes `Float64 → Float32` for
device buffers; the host-side outer subgradient loop stays in
Float64 and promotes back at the kernel boundary. Discretisation
error from the storage grid (`O(1/K)`) dominates the precision loss
in practice.

### Caveats (documented up front)

1. **Sublinear convergence.** Vanilla subgradient is `O(1/√k)`;
   a bundle method (`solver.use_bundle = true`) is the natural
   follow-up. C3.5 lands correctness, not convergence speed.
2. **Mass balance is violated until `μ` converges.** The returned
   primal trajectory is the DP forward simulation along the optimal
   policy — not a feasible primal until the dual converges.
   Sophisticated primal recovery (re-solving one scenario LP at fixed
   `(λ, μ)`) is a C3.6 follow-up.
3. **RegulationReservoir only.** `_validate_dp_inputs` rejects
   problems containing `BufferReservoir` modules; pass-through nodes
   without storage need a different subproblem form. Defer to C3.6.

### When does Metal actually beat CPU?

The C3.5 milestone is the *correctness + portability* contract.
Whether `GPUBackend(MetalBackend())` is faster than `CPUBackend()`
depends on instance size:

- For the 2-reservoir, 4-stage cascade (the test fixture), CPU is
  faster — Metal kernel launch overhead dominates the tiny workload.
- The crossover is expected around
  `n_scen · n_reg · n_stages > 50_000` based on the per-thread work
  estimate (K² reward evaluations per stage). At `n_scen = 512,
  n_reg = 10, n_stages = 52, K = 64`, total ops ≈ 1.6 × 10⁹ — well
  within Metal's strength zone.

The benchmark script at
[`test/bench_lagrangian_metal.jl`](test/bench_lagrangian_metal.jl)
sweeps `(n_scen, n_reg, n_stages, K_grid)` and writes
`benchmarks/c3_5_metal.csv`. Run with
`HYDROMODELS_RUN_BENCH=1 julia --project=. HydroModelsOpt/test/bench_lagrangian_metal.jl`.

### Backend extension hook change (Julia 1.12)

`default_backend()` now reads from a `Ref{Union{Nothing, ComputeBackend}}`
in `HydroModelsCore` (`_DEFAULT_BACKEND`). GPU vendor extensions
(CUDA, AMDGPU, Metal, oneAPI) install themselves via
`HydroModelsCore.set_default_backend!(GPUBackend(...))` from their
`__init__`. This replaces the previous pattern where each extension
defined a method on `default_backend()` itself — Julia 1.12 bans
that during precompile.

A side-effect worth knowing: when `Metal` is loaded (e.g. in the
test target), `default_backend()` returns
`GPUBackend(MetalBackend())`, and `LagrangianHydroSolver(; ...)` with
no explicit `backend = ...` picks Metal automatically. Tests that
specifically want the LP path on CPU now pass
`backend = CPUBackend()` and `dispatch = :lp` explicitly. The
existing test_lagrangian.jl C3 testsets were updated in C3.5 to make
this explicit.

## Pipeline scripts

- `scripts/run_pipeline.jl <yaml> [<market>]` — parse, solve, print
  objective + key totals + table sizes. Use for inspecting a single
  case from the command line.
- `scripts/parity_check.jl [<market>]` — sweep every `.yaml` in
  `HYDROMODELS_TEST_DATA_DIR` (default `../HydroModels_depr/data/real`)
  and print a one-line summary per case (objective, generation, spill,
  reserve allocation, group slack, elapsed). Used for manual
  objective / generation / reserve comparison against depr.
