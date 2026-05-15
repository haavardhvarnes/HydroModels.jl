using HydroModels
using Test

@testset "HydroModels umbrella" begin
    @testset "Re-exports from HydroModelsCore" begin
        for name in (
            :HydroModule, :TurbineUnit, :ReservoirType,
            :RegulationReservoir, :BufferReservoir,
            :ModuleConstraints, :PenaltyValues, :WaterTopology,
            :UncertaintyModel, :ScenarioTree, :ScenarioNode,
            :StagewiseIndependent, :MarkovianUncertainty, :ProdRiskUncertainty,
            :AbstractOptProblem, :LongTermHydroProblem,
            :FundamentalMarketProblem, :ShortTermHydroProblem,
            :StochasticShortTermProblem, :ElectricityMarket, :LoadBlock,
            :EndValueDescription,
            :AbstractRiskMeasure, :Expectation, :CVaR, :NestedCVaR,
            :Solution, :WaterValueCuts, :BendersCut,
            :ReservoirWaterValue, :CutGroup,
            :Plant, :Reservoir, :Generator, :Pump, :Tunnel,
            :InflowSeries, :MarketSeries,
            :ReserveSpec, :ReserveGroup,
            :ComputeBackend, :CPUBackend, :GPUBackend, :default_backend,
            :PiecewiseLinear, :PWL, :evaluate, :slopes, :build_topology,
            :allocate, :zeros_like, :ones_like, :is_gpu, :ka_backend,
        )
            @test isdefined(HydroModels, name)
        end
    end

    @testset "Re-exports from HydroModelsOpt" begin
        for name in (
            :AbstractSolver, :AbstractGPUSolver,
            :JuMPSolver, :SDDPHydroSolver,
            :SLPHydroSolver, :UnitCommitmentMode, :UnitLoadDispatchMode,
            :LagrangianHydroSolver,
            :ParallelismMode, :SynchronousParallel,
            :AsynchronousParallel, :TotallyAsynchronous,
            :DecompositionStrategy, :ScenarioDecomposition, :SpatialDecomposition,
            :StepSizeRule, :ConstantStep, :DiminishingStep, :PolyakStep,
            :compute_step,
            :HydropowerToolchain, :run_toolchain,
            :solve,
        )
            @test isdefined(HydroModels, name)
        end
    end
end
