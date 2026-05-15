using HydroModelsOpt
using Test

@testset "HydroModelsOpt" begin
    @testset "Loadability" begin
        # Solver implementations are stubs; this asserts the module
        # graph loads, the core verb resolves, and the strategy
        # singletons construct.
        @test isdefined(HydroModelsOpt, :solve)
        @test isdefined(HydroModelsOpt, :AbstractSolver)
        @test isdefined(HydroModelsOpt, :SDDPHydroSolver)
        @test isdefined(HydroModelsOpt, :SLPHydroSolver)
        @test isdefined(HydroModelsOpt, :LagrangianHydroSolver)
        @test SynchronousParallel() isa ParallelismMode
        @test ScenarioDecomposition() isa DecompositionStrategy
        @test ConstantStep(1.0) isa StepSizeRule
    end

    @testset "Reexports HydroModelsCore" begin
        @test isdefined(HydroModelsOpt, :HydroModule)
        @test isdefined(HydroModelsOpt, :LongTermHydroProblem)
        @test isdefined(HydroModelsOpt, :ShopShortTermProblem)
        @test isdefined(HydroModelsOpt, :CPUBackend)
    end

    @testset "JuMP LP baseline (HiGHS)" begin
        include("test_lp_baseline.jl")
    end

    @testset "Reserves (FOS §8 joint headroom, group obligation)" begin
        include("test_reserves.jl")
    end

    @testset "MILP (reversible pump-turbine binary mode)" begin
        include("test_milp.jl")
    end

    @testset "SDDP.jl bridge (StagewiseIndependent on cascade)" begin
        include("test_sddp.jl")
    end

    @testset "SDDP.jl bridge (ProdRiskUncertainty via Markovian graph)" begin
        include("test_sddp_prodrisk.jl")
    end

    @testset "Long-term → short-term toolchain" begin
        include("test_toolchain.jl")
    end

    @testset "Lagrangian (scenario-decomposed)" begin
        include("test_lagrangian.jl")
    end

    @testset "Spill routing (B1.5)" begin
        include("test_spill_routing.jl")
    end

    @testset "Multi-source generators (B1.6)" begin
        include("test_multi_source.jl")
    end

    @testset "Typed junctions (B1.7)" begin
        include("test_junction.jl")
    end
end
