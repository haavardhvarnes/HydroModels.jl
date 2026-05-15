# HydroModelsData

I/O layer of the **HydroModels** meta-package. **Placeholder — no
content yet.**

Planned scope when populated:

- SHOP / Harmonie YAML reader (carried over from `HydroModels_depr`)
- EMPS v10 reader (ReSDDP-inspired, written from scratch — not lifted
  from the GPLv3 ReSDDP source)
- HDF5 results I/O

Outputs will declare the **Tables.jl** interface rather than returning
`DataFrame`s. Tables.jl is the lightweight ecosystem interface;
DataFrames is just one consumer. Users convert at the boundary
(`DataFrame(sol.storage)`).
