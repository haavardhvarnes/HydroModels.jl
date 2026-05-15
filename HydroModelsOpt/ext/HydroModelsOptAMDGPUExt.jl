"""
AMDGPU extension — installs `GPUBackend(ROCBackend())` as the default
backend when AMDGPU is functional. Uses `set_default_backend!` from
`__init__` (Julia 1.12 bans precompile-time method overwriting).
"""
module HydroModelsOptAMDGPUExt

using HydroModelsCore
using HydroModelsOpt
using AMDGPU
using KernelAbstractions

function __init__()
    if AMDGPU.functional()
        HydroModelsCore.set_default_backend!(
            HydroModelsCore.GPUBackend(ROCBackend()),
        )
    end
end

end # module
