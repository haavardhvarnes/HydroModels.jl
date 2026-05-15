"""
Batched scenario subproblem solving — the core inner loop for
Lagrangian decomposition.

When the per-(scenario, reservoir) subproblem has been pre-solved
by `bellman_dp_kernel!` into a policy table indexed by storage grid,
forward simulation rolls out the realised trajectory along that
policy. Trivially parallel across `(scenario, reservoir)`; sequential
in `t`.
"""

using KernelAbstractions

"""
    forward_simulate_kernel!(S_traj, q_traj, σ_traj,
                             π_q, π_σ, π_next,
                             S0, S_grid, reg_mask)

Forward-roll the DP policy from initial storage `S0` through all
stages. The starting grid index is the one closest to `S0` (snap to
nearest grid point). Stage-by-stage, the kernel reads the chosen
`π_next[k, t, ir, is]` from the policy table and emits the
corresponding `(q, σ)` actions and next-state `S_traj[t+1, ir, is]`.

For `BufferReservoir` rows (where `reg_mask[ir] == false`), there is
no storage state; we emit `S = 0` and copy zeros to the action
trajectories. The current C3.5 milestone treats Buffer reservoirs as
a TODO (see `dp_decomposition.jl` validation) so this branch is
defensive — it ensures the kernel always writes well-defined values.

# Arguments
- `S_traj::AbstractArray{T,3}`     `(T+1, n_reg, n_scen)` — written
- `q_traj::AbstractArray{T,3}`     `(T,   n_reg, n_scen)` — written
- `σ_traj::AbstractArray{T,3}`     `(T,   n_reg, n_scen)` — written
- `π_q::AbstractArray{T,4}`        `(K, T, n_reg, n_scen)` — read
- `π_σ::AbstractArray{T,4}`        `(K, T, n_reg, n_scen)` — read
- `π_next::AbstractArray{Int32,4}` `(K, T, n_reg, n_scen)` — read
- `S0::AbstractArray{T,1}`         `(n_reg,)`              — initial volume per reservoir
- `S_grid::AbstractArray{T,2}`     `(K, n_reg)`            — storage grid
- `reg_mask::AbstractArray{Bool,1}` `(n_reg,)`
"""
@kernel function forward_simulate_kernel!(
        S_traj, q_traj, σ_traj,
        @Const(π_q), @Const(π_σ), @Const(π_next),
        @Const(S0), @Const(S_grid),
        @Const(reg_mask),
    )
    ir, is = @index(Global, NTuple)

    Tnum = eltype(S_traj)
    K    = size(S_grid, 1)
    nsta = size(q_traj, 1)

    @inbounds begin
        if reg_mask[ir]
            # Snap S0 to the nearest grid index.
            S0_i = S0[ir]
            best_k = Int32(1)
            best_d = abs(S_grid[1, ir] - S0_i)
            for k in 2:K
                d = abs(S_grid[k, ir] - S0_i)
                if d < best_d
                    best_d = d
                    best_k = Int32(k)
                end
            end
            k_curr = best_k
            S_traj[1, ir, is] = S_grid[k_curr, ir]

            for t in 1:nsta
                q_traj[t, ir, is] = π_q[k_curr, t, ir, is]
                σ_traj[t, ir, is] = π_σ[k_curr, t, ir, is]
                k_next = π_next[k_curr, t, ir, is]
                S_traj[t + 1, ir, is] = S_grid[k_next, ir]
                k_curr = k_next
            end
        else
            for t in 1:nsta
                q_traj[t, ir, is] = zero(Tnum)
                σ_traj[t, ir, is] = zero(Tnum)
                S_traj[t, ir, is] = zero(Tnum)
            end
            S_traj[nsta + 1, ir, is] = zero(Tnum)
        end
    end
end
