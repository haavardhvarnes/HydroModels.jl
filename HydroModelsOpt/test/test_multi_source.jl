using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HiGHS
using Tables
using Test

# ============================================================
# B1.6 — multi-source generators (vector-valued `Generator.from_res`)
# ============================================================
#
# Fixture: `multi_source.yaml` — three upstream reservoirs (upper_a,
# upper_b, upper_c) feed `MergePlant` via separate penstock tunnels
# that merge at an `IntakeKP` junction node. The plant discharges to
# `lower`. Per-source tunnel caps are 12, 8, 5 m³/s (sum = 25 = plant
# q_max). A peak-price hour forces dispatch at q_max so every per-
# source cap binds.

const MULTI_SOURCE_PATH = joinpath(@__DIR__, "..", "..",
    "HydroModelsData", "test", "data", "multi_source.yaml")

@testset "Multi-source — parser populates the source vector" begin
    parsed = read_shop_yaml(MULTI_SOURCE_PATH)
    @test length(parsed.generators) == 1
    g = parsed.generators[1]
    @test g.plant == "MergePlant"
    @test g.to_res == "lower"
    # Sorted by capacity desc: upper_a (s_max=100) > upper_b (60) > upper_c (30).
    @test g.from_res == ["upper_a", "upper_b", "upper_c"]
    @test g.head_reference === nothing   # defaults to from_res[1]
    @test length(g.source_qmax) == 3
    @test g.source_qmax["upper_a"] ≈ 12.0
    @test g.source_qmax["upper_b"] ≈ 8.0
    @test g.source_qmax["upper_c"] ≈ 5.0
end

@testset "Multi-source — phantom virtual tunnels suppressed" begin
    # The parser previously emitted Logga→Eikrebekken-style phantom
    # tunnels remapping non-dominant sources to the dominant one. With
    # B1.6, multi-source plants suppress those because per-source
    # Qg_from variables handle the attribution directly. The fixture
    # has no other inter-reservoir paths, so `parsed.tunnels` should
    # be empty.
    parsed = read_shop_yaml(MULTI_SOURCE_PATH)
    @test isempty(parsed.tunnels)
end

@testset "Multi-source — LP per-source flow split end-to-end" begin
    parsed = read_shop_yaml(MULTI_SOURCE_PATH)
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    @test sol.termination == "OPTIMAL"
    @test isfinite(sol.objective)
    @test sol.objective > 0

    sc = Tables.columns(sol.storage)
    final_by_res = Dict{String, Float64}()
    initial_by_res = Dict("upper_a" => 100.0, "upper_b" => 60.0,
                          "upper_c" => 30.0, "lower" => 200.0)
    for i in eachindex(sc.S)
        final_by_res[String(sc.reservoir[i])] = sc.S[i]
    end
    # Every upstream reservoir lost water — the LP saturates each
    # per-source cap during the peak-price window. (Without B1.6 only
    # the dominant source would deplete.)
    for r in ("upper_a", "upper_b", "upper_c")
        @test final_by_res[r] < initial_by_res[r] + 1.0
    end
    # Lower accumulates the dispatched water.
    @test final_by_res["lower"] > initial_by_res["lower"]
end

@testset "Multi-source — Q_gen never exceeds plant q_max and saturates at peak" begin
    parsed = read_shop_yaml(MULTI_SOURCE_PATH)
    prob = ShopShortTermProblem(parsed)
    sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
        options = Dict(:output_flag => false)))

    qc = Tables.columns(sol.q_gen)
    @test all(isfinite, qc.Q)
    @test maximum(qc.Q) <= 25.0 + 1.0e-6
    # The peak-price hour can saturate only if every per-source cap is
    # exercised in parallel (sum of caps = q_max). This is the test
    # that fails without B1.6's per-source flow split.
    @test maximum(qc.Q) >= 24.99
end
