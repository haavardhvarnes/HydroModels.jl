"""
Intel oneAPI extension — installs `GPUBackend(oneAPIBackend())` as the
default backend when oneAPI is functional. Uses `set_default_backend!`
from `__init__` (Julia 1.12 bans precompile-time method overwriting).
"""
module HydroModelsOptoneAPIExt

using HydroModelsCore
using HydroModelsOpt
using oneAPI
using KernelAbstractions

function __init__()
    if oneAPI.functional()
        HydroModelsCore.set_default_backend!(
            HydroModelsCore.GPUBackend(oneAPIBackend()),
        )
    end
end

end # module
