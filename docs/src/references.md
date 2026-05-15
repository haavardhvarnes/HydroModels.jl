# Canonical references

The architecture is anchored in a small set of papers; deep changes should
be checked against these.

## Long/medium-term (SDDP family)
- Pereira, M.V.F., Pinto, L.M.V.G. (1991) "Multi-stage stochastic optimization
  applied to energy planning", *Mathematical Programming* 52: 359-375.
- Gjelsvik, A., Mo, B., Haugstad, A. (2010) "Long- and medium-term operations
  planning and stochastic modelling in hydro-dominated power systems based
  on SDDP", in *Handbook of Power Systems I*, Springer, ch. 2.
- Gjelsvik, A., Belsnes, M.M., Haugstad, A. (1999) "An algorithm for
  stochastic medium-term hydrothermal scheduling under spot price
  uncertainty", *Proc. PSCC* (Trondheim).
- Helseth, A., Braaten, H. (2015) "Efficient parallelization of the SDDP
  algorithm applied to hydropower scheduling", *Energies* 8(12): 14287-14297.
- Helseth, A., Mo, B., Hågenvik, H.O., Schäffer, L.E. (2022) "Hydropower
  scheduling with state-dependent discharge constraints: an SDDP approach",
  *J. Water Resour. Plann. Manage.* 148(11).

## Short-term (SHOP family)
- Skjelbred, H.I., Kong, J., Fosso, O.B. (2019) "Dynamic incorporation of
  nonlinearity into MILP formulation for short-term hydro scheduling",
  *Int. J. Electr. Power & Energy Syst.* 116: 105530.
- Belsnes, M.M., Wolfgang, O., Follestad, T., Aasgård, E.K. (2016)
  "Applying successive linear programming for stochastic short-term
  hydropower optimization", *Electr. Power Syst. Res.* 130: 167-180.
- Kong, J., Skjelbred, H.I., Fosso, O.B. (2020) "An overview on
  formulations and optimization methods for the unit-based short-term
  hydro scheduling problem", *Electr. Power Syst. Res.* 178: 106027.
- Skjelbred, H.I. (2020) PhD thesis, NTNU.

## SINTEF documentation (operational ground truth)
- ProdRisk: <https://docs.prodrisk.sintef.energy>
- SHOP: <https://docs.shop.sintef.energy>
- MadSuite: <https://madsuite.org>

## GPU optimization
- Lu, H. et al. — cuPDLP.jl and cuPDLP+/cuPDLPx papers
- MadNLP/MadIPM project — <https://github.com/MadNLP>
- NVIDIA cuOpt — <https://github.com/NVIDIA/cuopt>
