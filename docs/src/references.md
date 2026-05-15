# Canonical references

The architecture is anchored in a small set of canonical papers.
Non-trivial design decisions should be checked against these.

## Long / medium-term — SDDP family

- **Pereira, M. V. F. & Pinto, L. M. V. G.** (1991) "Multi-stage
  stochastic optimization applied to energy planning",
  *Mathematical Programming* 52: 359–375. The original SDDP
  method.
- **Gjelsvik, A., Mo, B., Haugstad, A.** (2010) "Long- and medium-
  term operations planning and stochastic modelling in hydro-
  dominated power systems based on SDDP", in *Handbook of Power
  Systems I*, Springer, ch. 2. The canonical Nordic SDDP review and
  *the* reference for ProdRisk algorithmic choices.
- **Gjelsvik, A., Belsnes, M. M., Haugstad, A.** (1999) "An algorithm
  for stochastic medium-term hydrothermal scheduling under spot
  price uncertainty", *Proc. PSCC* (Trondheim). The combined SDP/SDDP
  algorithm — what ProdRisk actually implements.
- **Helseth, A. & Braaten, H.** (2015) "Efficient parallelization of
  the SDDP algorithm applied to hydropower scheduling", *Energies*
  8(12): 14287–14297. Source for `ParallelismMode` types
  (synchronous / asynchronous / totally asynchronous).
- **Helseth, A., Mo, B., Hågenvik, H. O., Schäffer, L. E.** (2022)
  "Hydropower scheduling with state-dependent discharge
  constraints: an SDDP approach", *J. Water Resour. Plann. Manage.*
  148(11).

## Short-term — SHOP family

- **Skjelbred, H. I., Kong, J., Fosso, O. B.** (2019) "Dynamic
  incorporation of nonlinearity into MILP formulation for short-term
  hydro scheduling", *Int. J. Electr. Power & Energy Syst.* 116:
  105530. Source for `SLPHydroSolver` design.
- **Belsnes, M. M., Wolfgang, O., Follestad, T., Aasgård, E. K.**
  (2016) "Applying successive linear programming for stochastic
  short-term hydropower optimization", *Electr. Power Syst. Res.*
  130: 167–180.
- **Kong, J., Skjelbred, H. I., Fosso, O. B.** (2020) "An overview
  on formulations and optimization methods for the unit-based
  short-term hydro scheduling problem", *Electr. Power Syst. Res.*
  178: 106027.
- **Skjelbred, H. I.** (2019) Comprehensive modern SHOP formulation.
  PhD thesis, NTNU Open.

## GPU optimization

- **Lu, H. et al.** — cuPDLP.jl and cuPDLP+ / cuPDLPx papers.
  First-order GPU LP solver line of work.
- **MadIPM.jl** — interior-point on GPU via cuDSS.
- **NVIDIA cuOpt** — production GPU solver (LP / MIP / VRP).
- **Helseth, A. & Braaten, H.** (2015) — parallelization choices for
  hydropower SDDP (also relevant under SDDP family above).

## SINTEF documentation (operational ground truth)

- ProdRisk: <https://docs.prodrisk.sintef.energy>
- SHOP: <https://docs.shop.sintef.energy>
- MadSuite: <https://madsuite.org>
- ReSDDP (open-source SDDP for hydropower, GPLv3 — referenced for
  design only, no code dependency):
  <https://gitlab.sintef.no/energy/res100/resddp>

## Julia ecosystem dependencies (a partial bibliography)

- **JuMP.jl** — Dunning, I., Huchette, J., Lubin, M. (2017) "JuMP:
  A modeling language for mathematical optimization", *SIAM Review*
  59(2): 295–320.
- **SDDP.jl** — Dowson, O., Kapelevich, L. (2021) "SDDP.jl: a Julia
  package for stochastic dual dynamic programming", *INFORMS J.
  Computing* 33(1): 27–33.
- **KernelAbstractions.jl** — Churavy, V. et al., active project.
  <https://github.com/JuliaGPU/KernelAbstractions.jl>.
- **Makie.jl** — Danisch, S., Krumbiegel, J. (2021) "Makie.jl: Flexible
  high-performance data visualization for Julia", *J. Open Source
  Software* 6(65): 3349.

## See also

- The root [`CLAUDE.md`](https://github.com/haavardhvarnes/HydroModels.jl/blob/main/CLAUDE.md)
  is the agent-facing summary of these references and the
  architectural principles they motivate.
- The per-subpackage `CLAUDE.md` files (`HydroModelsCore/CLAUDE.md`,
  `HydroModelsOpt/CLAUDE.md`, …) narrow the scope further.
