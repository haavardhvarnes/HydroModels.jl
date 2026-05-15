"""
CUDA extension — wires up `GPUBackend(CUDABackend())` as the default
backend when CUDA is functional. Uses `set_default_backend!` from
`__init__` (Julia 1.12 bans precompile-time method overwriting).
"""
module HydroModelsOptCUDAExt

using HydroModelsCore
using HydroModelsOpt
using CUDA
using KernelAbstractions

function __init__()
    if CUDA.functional()
        HydroModelsCore.set_default_backend!(
            HydroModelsCore.GPUBackend(CUDABackend()),
        )
    end
end

end # module
