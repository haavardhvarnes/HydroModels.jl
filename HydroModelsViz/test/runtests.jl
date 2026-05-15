using HydroModelsCore
using HydroModelsOpt
using HydroModelsData
using HydroModelsViz
using HiGHS
using CairoMakie
using Makie
using Test

const FIXTURE = joinpath(
    dirname(@__DIR__), "..",
    "HydroModelsData", "test", "data", "minimal.yaml",
)

@testset "HydroModelsViz" begin
    @testset "Module loads, exports present" begin
        for sym in (:plot_dispatch!, :plot_storage!, :plot_pumps!, :plot_spill!,
                    :plot_prices!, :plot_revenue!,
                    :plot_reserves_alloc!, :plot_reserves_groups!,
                    :plot_water_value!, :plot_inflows!, :plot_topology!,
                    :plot_dashboard)
            @test isdefined(HydroModelsViz, sym)
        end
    end

    @testset "CairoMakie extension activated" begin
        @test HydroModelsViz._has_cairomakie() === true
    end

    # Solve minimal.yaml end-to-end and render every panel into a fresh
    # Axis. The base Makie API doesn't require CairoMakie for in-memory
    # plotting; CairoMakie is needed only to save a PNG, which we do at
    # the end of the test as a smoke-render.
    @testset "Panels render on a solved minimal.yaml" begin
        parsed = read_shop_yaml(FIXTURE; market_name = "NO3")
        prob = ShopShortTermProblem(parsed)
        sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)))
        @test sol.termination == "OPTIMAL"

        # Each panel into its own one-cell Figure to exercise the panel
        # function in isolation.
        for panelfn in (plot_dispatch!, plot_storage!, plot_pumps!,
                        plot_spill!, plot_prices!, plot_revenue!,
                        plot_reserves_alloc!, plot_reserves_groups!,
                        plot_water_value!, plot_inflows!, plot_topology!)
            fig = Figure(size = (600, 400))
            ax = Axis(fig[1, 1])
            @test panelfn(ax, sol) === ax
        end
    end

    @testset "Dashboard composition returns a Figure" begin
        parsed = read_shop_yaml(FIXTURE; market_name = "NO3")
        prob = ShopShortTermProblem(parsed)
        sol = solve(prob, JuMPSolver(HiGHS.Optimizer;
            options = Dict(:output_flag => false)))

        fig = plot_dashboard(sol; size = (1200, 1400))
        @test fig isa Figure

        # Save via standard Makie API (CairoMakie loaded) to verify
        # the figure is fully formed and serializable.
        tmppath = tempname() * ".png"
        try
            Makie.save(tmppath, fig)
            @test isfile(tmppath)
            @test filesize(tmppath) > 0
        finally
            rm(tmppath; force = true)
        end
    end
end
