"""
KernelAbstractions kernels for the Lagrangian subgradient solver.

These run on any backend (CPU, CUDA, AMDGPU, Metal, oneAPI) thanks to
KernelAbstractions.jl. No CUDA-specific code should appear here.
"""

using KernelAbstractions

"""
    subgradient_update_kernel!(λ, g, α, lower, upper)

In-place projected subgradient step:

    λᵢ ← clamp(λᵢ + α gᵢ, lower, upper)

For dualization of equality constraints, `lower = -Inf` and `upper = +Inf`.
For inequalities `≤`, use `lower = 0`.
"""
@kernel function subgradient_update_kernel!(λ, @Const(g), α, lower, upper)
    i = @index(Global)
    @inbounds λ[i] = clamp(λ[i] + α * g[i], lower, upper)
end

"""
    subgradient_update!(backend, λ, g, α; lower=zero, upper=Inf)

Launch the subgradient update on the given backend. Returns `λ`.
"""
function subgradient_update!(backend, λ::AbstractArray{T}, g::AbstractArray{T},
                             α::Real; lower = zero(T), upper = T(Inf)) where {T}
    kernel! = subgradient_update_kernel!(backend)
    kernel!(λ, g, T(α), T(lower), T(upper); ndrange = length(λ))
    KernelAbstractions.synchronize(backend)
    return λ
end

# ============================================================
# C3.5: mass-balance subgradient for spatial-coupling dualisation
# ============================================================

"""
    compute_subgradient_flow_kernel!(g_μ, S_traj, q_traj, σ_traj,
                                    inflow, edge_u, edge_v, n_edges)

Per `(stage, reservoir, scenario)` work item, compute the mass-balance
residual

    g_μ_{s,i,t} = (S_{i,t+1} − S_{i,t}) − inflow_{i,t}
                  − Σ_{u → i} (q_{u,t} + σ_{u,t})
                  + q_{i,t} + σ_{i,t}

At the dual optimum this is zero (mass balance is satisfied as an
equality). Subgradient on `μ` is the residual itself.

The upstream-adjacency is provided as two flat vectors
`(edge_u, edge_v)` of length `n_edges` so the kernel can scan them
without device-side hash lookups. Topologies are tiny (`n_edges`
≤ a few dozen for realistic cases); the linear scan is cheap.

# Arguments
- `g_μ::AbstractArray{T,3}`        `(T_stg, n_reg, n_scen)` — written
- `S_traj::AbstractArray{T,3}`     `(T_stg+1, n_reg, n_scen)`
- `q_traj::AbstractArray{T,3}`     `(T_stg,   n_reg, n_scen)`
- `σ_traj::AbstractArray{T,3}`     `(T_stg,   n_reg, n_scen)`
- `inflow::AbstractArray{T,3}`     `(T_stg, n_reg, n_scen)`
- `edge_u::AbstractArray{Int32,1}` `(n_edges,)`             — upstream module index
- `edge_v::AbstractArray{Int32,1}` `(n_edges,)`             — downstream module index
"""
@kernel function compute_subgradient_flow_kernel!(
        g_μ,
        @Const(S_traj), @Const(q_traj), @Const(σ_traj),
        @Const(inflow),
        @Const(edge_u), @Const(edge_v),
    )
    it, ir, is = @index(Global, NTuple)
    Tnum = eltype(g_μ)
    n_edges = length(edge_u)
    @inbounds begin
        residual = (S_traj[it + 1, ir, is] - S_traj[it, ir, is]) -
                   inflow[it, ir, is] +
                   q_traj[it, ir, is] + σ_traj[it, ir, is]
        for e in 1:n_edges
            if edge_v[e] == Int32(ir)
                u = edge_u[e]
                residual -= q_traj[it, u, is] + σ_traj[it, u, is]
            end
        end
        g_μ[it, ir, is] = residual
    end
end
