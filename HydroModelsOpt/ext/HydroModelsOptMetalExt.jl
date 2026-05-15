"""
Apple Metal extension.

When `Metal` is loaded:

1. `__init__` calls `HydroModelsCore.set_default_backend!` to install
   `GPUBackend(MetalBackend())` as the value returned by
   `HydroModelsCore.default_backend()`, provided Metal is functional on
   the host. This is the Julia 1.12-correct hook — `set_default_backend!`
   mutates a `Ref` rather than overwriting a precompiled method.
2. `HydroModelsOpt._backend_eltype(::GPUBackend{<:MetalBackend}, Float64)`
   demotes to `Float32`. Apple Silicon GPUs do not support native
   double-precision arithmetic, so kernels launched on a `MetalBackend`
   must operate in `Float32`. The host-side outer loop in
   `HydroModelsOpt`'s Lagrangian solver stays in `Float64` and promotes
   the kernel result back at the boundary.
"""
module HydroModelsOptMetalExt

using HydroModelsCore
using HydroModelsOpt
using Metal
using KernelAbstractions

function __init__()
    if Metal.functional()
        HydroModelsCore.set_default_backend!(
            HydroModelsCore.GPUBackend(MetalBackend()),
        )
    end
end

# Apple GPUs are single-precision only. Demote Float64 → Float32 when
# allocating device buffers for kernel inputs / outputs.
HydroModelsOpt._backend_eltype(
        ::HydroModelsCore.GPUBackend{<:MetalBackend},
        ::Type{Float64},
    ) = Float32

end # module
