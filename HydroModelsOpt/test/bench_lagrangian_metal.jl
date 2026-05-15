"""
Metal-vs-CPU benchmark for the C3.5 Lagrangian DP dispatch.

Sweeps `(n_scen, n_reg, n_stages, K_grid)` and measures wall-clock per
outer iteration on `CPUBackend()` (running the LP path via HiGHS) and
on `GPUBackend(MetalBackend())` (running the DP path via KA kernels).

Run this script **manually**, not via `Pkg.test` — it allocates large
device arrays and would slow the default test loop down. Gated on
`Metal.functional() == true` and the `HYDROMODELS_RUN_BENCH` env var:

```bash
HYDROMODELS_RUN_BENCH=1 julia --project=. HydroModelsOpt/test/bench_lagrangian_metal.jl
```

Results are appended to `benchmarks/c3_5_metal.csv`; the same case is
re-run if you invoke the script again.

The instance generator builds an N-reservoir linear cascade with
identical per-reservoir parameters; the only thing that scales across
instances is `(n_scen, n_reg, n_stages, K_grid)`.
"""

if get(ENV, "HYDROMODELS_RUN_BENCH", "0") != "1"
    @info "bench_lagrangian_metal.jl: HYDROMODELS_RUN_BENCH != 1, exiting. " *
          "Set HYDROMODELS_RUN_BENCH=1 to run."
    exit(0)
end

using HydroModelsCore
using HydroModelsOpt
using HiGHS
using BenchmarkTools
using Statistics
using Printf
using Dates

using Metal

# ============================================================
# Instance generator: linear cascade with uniform parameters
# ============================================================

function bench_cascade(n_reg::Int, n_stages::Int, n_scen::Int)
    modules = HydroModule{Float64, RegulationReservoir}[]
    for i in 1:n_reg
        m = HydroModule{Float64, RegulationReservoir}(
            name = Symbol("res$i"),
            max_vol = 100.0, initial_vol = 50.0,
            rated_discharge = 10.0,
            energy_factor = 1.0 - 0.05 * (i - 1),
            discharge_to = i < n_reg ? Symbol("res$(i+1)") : nothing,
        )
        push!(modules, m)
    end
    topology = build_topology(modules)

    # Synthetic K=4 realizations per stage, all reservoirs share the same
    # inflow vector at each realization. Cyclic sampling in
    # `_enumerate_or_sample_scenarios` produces the `n_scen` paths.
    K_real = 4
    realizations = [
        [Float64[2.0 + 1.0 * k + 0.3 * i for i in 1:n_reg] for k in 1:K_real]
        for _ in 1:n_stages
    ]
    probabilities = [fill(1.0 / K_real, K_real) for _ in 1:n_stages]
    uncertainty = StagewiseIndependent{Float64}(realizations, probabilities)
    stage_prices = [25.0 + 5.0 * sin(2π * t / max(n_stages, 1)) for t in 1:n_stages]

    return LongTermHydroProblem{Float64, StagewiseIndependent{Float64}}(
        modules = modules, topology = topology,
        num_stages = n_stages, uncertainty = uncertainty,
        stage_prices = stage_prices,
    )
end

# ============================================================
# Benchmark runner — measures average per-iteration wall-clock
# ============================================================

function bench_one(prob, backend, dispatch::Symbol;
                   K_grid::Int, n_scen::Int, max_iter::Int)
    subproblem_solver = dispatch === :lp ?
        JuMPSolver(HiGHS.Optimizer; options = Dict(:output_flag => false)) :
        nothing
    solver = LagrangianHydroSolver(;
        backend = backend,
        subproblem_solver = subproblem_solver,
        step_size = DiminishingStep(1.0, 0.9),
        max_iter = max_iter,
        n_scenarios = n_scen,
        dispatch = dispatch,
        K_grid = K_grid,
        dual_flow_coupling = false,
    )
    # Warm-up to avoid first-call compile time.
    solve(prob, solver)
    # Time three solve()s and take the median; per-iteration time =
    # total / max_iter.
    times = Float64[]
    for _ in 1:3
        t0 = time_ns()
        solve(prob, solver)
        push!(times, (time_ns() - t0) / 1e9)
    end
    return median(times) / max_iter
end

# ============================================================
# Sweep
# ============================================================

function main()
    Metal.functional() || error(
        "Metal is not functional on this host — cannot run the GPU side of " *
        "this benchmark. Run on Apple Silicon hardware.")

    sweep = [
        # (n_scen, n_reg, n_stages, K_grid)
        (16,  2,  4,  32),     # tiny — CPU should dominate
        (32,  2,  12, 32),
        (64,  4,  24, 64),
        (128, 6,  52, 64),
        (256, 8,  52, 64),
        (512, 10, 52, 64),     # large — Metal should start winning
    ]

    println("\n" * "="^80)
    @printf("%-8s %-6s %-8s %-6s | %-12s %-12s %-10s\n",
            "n_scen", "n_reg", "n_stages", "K", "CPU LP (s)", "Metal DP (s)", "speedup")
    println("="^80)

    bench_path = joinpath(@__DIR__, "..", "..", "benchmarks", "c3_5_metal.csv")
    mkpath(dirname(bench_path))
    new_file = !isfile(bench_path)
    open(bench_path, "a") do io
        new_file && println(io,
            "timestamp,n_scen,n_reg,n_stages,K_grid,cpu_lp_sec,metal_dp_sec,speedup")

        for (n_scen, n_reg, n_stages, K_grid) in sweep
            prob = bench_cascade(n_reg, n_stages, n_scen)

            t_cpu = bench_one(prob, CPUBackend(), :lp;
                              K_grid = K_grid, n_scen = n_scen, max_iter = 3)
            t_metal = bench_one(prob, GPUBackend(MetalBackend()), :dp;
                                K_grid = K_grid, n_scen = n_scen, max_iter = 3)
            speedup = t_cpu / t_metal

            @printf("%-8d %-6d %-8d %-6d | %-12.4f %-12.4f %-10.2f\n",
                    n_scen, n_reg, n_stages, K_grid,
                    t_cpu, t_metal, speedup)
            println(io,
                "$(now()),$(n_scen),$(n_reg),$(n_stages),$(K_grid)," *
                "$(t_cpu),$(t_metal),$(speedup)")
        end
    end
    println("="^80)
    println("Results appended to $(bench_path)")
end

main()
