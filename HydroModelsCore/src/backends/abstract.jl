"""
Backend abstraction — vendor-neutral compute substrate.

This module exists so that no other code in `HydroModels` needs to know
whether it's running on NVIDIA, AMD, Apple, Intel, or just multi-threaded
CPU. All vendor-specific code lives in the four GPU package extensions
at `HydroModelsOpt/ext/HydroModelsOpt<Vendor>Ext.jl`.

The principle: write kernels through KernelAbstractions, dispatch on
`ComputeBackend`, and let the package extensions wire up the concrete
backend types.
"""

# ============================================================
# Backend types
# ============================================================

"""
    ComputeBackend

Abstract type representing a compute substrate. Concrete subtypes are
`CPUBackend` and `GPUBackend{B<:KernelAbstractions.Backend}`. The latter
is parameterized by the underlying KA backend (`CUDABackend`,
`ROCBackend`, `MetalBackend`, `oneAPIBackend`, etc.) when the relevant
extension is loaded.
"""
abstract type ComputeBackend end

"""
    CPUBackend()

Compute on the host CPU. Falls back to multi-threaded execution via
KernelAbstractions' CPU backend when kernels are invoked.
"""
struct CPUBackend <: ComputeBackend end

"""
    GPUBackend(ka_backend)

GPU compute via a specific KernelAbstractions backend. Construct via
the appropriate package extension, e.g.

```julia
using CUDA
backend = GPUBackend(CUDABackend())
```
"""
struct GPUBackend{B<:KernelAbstractions.Backend} <: ComputeBackend
    backend::B
end

# ============================================================
# Backend selection
# ============================================================

"""
    default_backend()

Return a sensible default backend for the current environment.

Falls back to `CPUBackend()`. Vendor extensions (CUDA, AMDGPU, Metal,
oneAPI in `HydroModelsOpt/ext/`) set this to a `GPUBackend` from their
`__init__` when the vendor is functional on the host. Selection is via
`set_default_backend!(::ComputeBackend)` rather than method overrides —
Julia 1.12 bans method overwriting during precompilation, so the
extension hook is a settable `Ref` instead of a redefined method.
"""
default_backend() = something(_DEFAULT_BACKEND[], CPUBackend())

"""
    set_default_backend!(backend::ComputeBackend)

Install `backend` as the value returned by future calls to
`default_backend()`. Vendor extensions invoke this from `__init__` when
their GPU is functional.
"""
function set_default_backend!(backend::ComputeBackend)
    _DEFAULT_BACKEND[] = backend
    return backend
end

const _DEFAULT_BACKEND = Ref{Union{Nothing, ComputeBackend}}(nothing)

# ============================================================
# Allocation helpers
# ============================================================

"""
    allocate(backend, T, dims...)

Allocate an uninitialized array of element type `T` and given dimensions
on the specified backend.

Concrete backends implement specialized methods; the CPU fallback returns
a plain `Array`.
"""
allocate(::CPUBackend, ::Type{T}, dims...) where {T} = Array{T}(undef, dims...)

# Generic GPU dispatch goes through KernelAbstractions.allocate
function allocate(b::GPUBackend, ::Type{T}, dims...) where {T}
    return KernelAbstractions.allocate(b.backend, T, dims...)
end

"""
    zeros_like(backend, T, dims...)

Allocate a zero-initialized array of element type `T` on the backend.
"""
function zeros_like(b::ComputeBackend, ::Type{T}, dims...) where {T}
    arr = allocate(b, T, dims...)
    fill!(arr, zero(T))
    return arr
end

"""
    ones_like(backend, T, dims...)

Allocate a one-initialized array of element type `T` on the backend.
"""
function ones_like(b::ComputeBackend, ::Type{T}, dims...) where {T}
    arr = allocate(b, T, dims...)
    fill!(arr, one(T))
    return arr
end

# ============================================================
# Backend queries
# ============================================================

"""
    is_gpu(backend)

`true` if `backend` is a `GPUBackend`.
"""
is_gpu(::CPUBackend) = false
is_gpu(::GPUBackend) = true

"""
    ka_backend(backend)

Extract the underlying KernelAbstractions backend, suitable for use in
kernel launches. For `CPUBackend`, returns `KernelAbstractions.CPU()`.
"""
ka_backend(::CPUBackend) = KernelAbstractions.CPU()
ka_backend(b::GPUBackend) = b.backend
