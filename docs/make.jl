using Documenter

using HydroModelsCore
using HydroModelsData
using HydroModelsOpt
using HydroModelsForecast
using HydroModelsViz
using HydroModels

DocMeta.setdocmeta!(HydroModelsCore, :DocTestSetup,
                    :(using HydroModelsCore); recursive = true)
DocMeta.setdocmeta!(HydroModelsData, :DocTestSetup,
                    :(using HydroModelsData); recursive = true)
DocMeta.setdocmeta!(HydroModelsOpt, :DocTestSetup,
                    :(using HydroModelsOpt); recursive = true)
DocMeta.setdocmeta!(HydroModelsForecast, :DocTestSetup,
                    :(using HydroModelsForecast); recursive = true)
DocMeta.setdocmeta!(HydroModelsViz, :DocTestSetup,
                    :(using HydroModelsViz); recursive = true)

makedocs(
    sitename = "HydroModels.jl",
    authors  = "Håvard Hvarnes <haavardhvarnes@gmail.com>",
    modules  = [
        HydroModels,
        HydroModelsCore,
        HydroModelsData,
        HydroModelsOpt,
        HydroModelsForecast,
        HydroModelsViz,
    ],
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://haavardhvarnes.github.io/HydroModels.jl",
        edit_link  = "main",
        sidebar_sitename = false,
        inventory_version = "0.1.0",
    ),
    pages    = [
        "Home"             => "index.md",
        "Installation"     => "installation.md",
        "Architecture"     => "architecture.md",
        "Tutorials"        => [
            "tutorials/lp_baseline.md",
            "tutorials/milp_unit_commitment.md",
            "tutorials/reserves.md",
            "tutorials/sddp_stagewise_independent.md",
            "tutorials/toolchain_long_to_short.md",
            "tutorials/lagrangian_gpu.md",
            "tutorials/visualisation.md",
        ],
        "API reference"    => [
            "api/core.md",
            "api/data.md",
            "api/opt.md",
            "api/forecast.md",
            "api/viz.md",
        ],
        "Status"           => "status.md",
        "References"       => "references.md",
    ],
    warnonly  = [:missing_docs, :cross_references, :linkcheck, :docs_block],
    checkdocs = :none,
)
