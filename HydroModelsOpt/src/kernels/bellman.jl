"""
KernelAbstractions kernels for Bellman recursion / dynamic programming.

Used by C3.5's GPU-native Lagrangian solver. After dualising both the
end-of-horizon storage targets (multipliers λ) and the cross-reservoir
mass-balance equalities (multipliers μ), each `(scenario, reservoir)`
subproblem decouples to a 1-D finite-horizon optimal-control problem
in storage `S`, solvable by backward Bellman recursion on a
discretised grid.

The action-space `(q, σ)` collapses: with a fixed target next-state
`S_{t+1}`, mass balance pins `q + σ = S_t + inflow − S_{t+1}`, and
since `η > 0` and the μ coefficients are identical on `q` and `σ`,
the optimal split is bang-bang on `q` first then `σ`. The DP loop
iterates only over next-state grid indices — `O(K²)` per stage per
`(scenario, reservoir)`.
"""

using KernelAbstractions

"""
    bellman_dp_kernel!(V_next, V_curr, π_q, π_σ, π_next,
                      inflow_t, price_t, μ_t, μ_d_t,
                      S_grid, η, q_max, S_min, S_max, λ,
                      reg_mask, has_downstream, t)

Single stage of the backward Bellman sweep. Each work item is a
`(scenario, reservoir, grid_k)` triple. Computes `V_curr[k, i, s]` and
the optimal policy from `V_next[:, i, s]`.

The kernel is launched once per stage by the host-side outer loop;
KA does not provide intra-kernel barriers across `ndrange`.

# Arguments (all device arrays)
- `V_next::AbstractArray{T,3}`  `(K, n_reg, n_scen)` — value at stage t+1 (read)
- `V_curr::AbstractArray{T,3}`  `(K, n_reg, n_scen)` — value at stage t (write)
- `π_q   ::AbstractArray{T,3}`  `(K, n_reg, n_scen)` — optimal turbine flow (write)
- `π_σ   ::AbstractArray{T,3}`  `(K, n_reg, n_scen)` — optimal spill (write)
- `π_next::AbstractArray{Int32,3}` `(K, n_reg, n_scen)` — chosen next-state grid index (write)
- `inflow_t::AbstractArray{T,2}` `(n_reg, n_scen)` — inflow at stage t
- `price_t::T`                   — scalar price at stage t
- `μ_t   ::AbstractArray{T,2}`   `(n_reg, n_scen)` — local mass-balance dual at stage t
- `μ_d_t ::AbstractArray{T,2}`   `(n_reg, n_scen)` — downstream's μ at stage t (the dual that
                                  "credits" sending flow downstream); zero where there is no
                                  downstream module
- `S_grid::AbstractArray{T,2}`   `(K, n_reg)` — discretised storage levels per reservoir
- `η::AbstractArray{T,1}`        `(n_reg,)`   — energy factor per reservoir
- `q_max::AbstractArray{T,1}`    `(n_reg,)`   — rated discharge per reservoir
- `S_min::AbstractArray{T,1}`    `(n_reg,)`   — minimum storage
- `S_max::AbstractArray{T,1}`    `(n_reg,)`   — maximum storage
- `reg_mask::AbstractArray{Bool,1}` `(n_reg,)` — `true` for RegulationReservoir; `false` for
                                                BufferReservoir (treated as zero-storage
                                                pass-through; V is identically 0)
"""
@kernel function bellman_dp_kernel!(
        V_curr, @Const(V_next),
        π_q, π_σ, π_next,
        @Const(inflow_t), price_t,
        @Const(μ_t), @Const(μ_d_t),
        @Const(S_grid),
        @Const(η), @Const(q_max), @Const(S_min), @Const(S_max),
        @Const(reg_mask),
        t::Int32,
    )
    ik, ir, is = @index(Global, NTuple)

    Tnum = eltype(V_curr)
    K    = size(V_curr, 1)

    @inbounds begin
        if reg_mask[ir]
            S_t      = S_grid[ik, ir]
            inflow   = inflow_t[ir, is]
            η_i      = η[ir]
            qmax_i   = q_max[ir]
            μ_self   = μ_t[ir, is]
            μ_down   = μ_d_t[ir, is]
            Smin_i   = S_min[ir]
            Smax_i   = S_max[ir]

            # Reservoir water balance:   q + σ = S_t + inflow − S_{t+1}
            # We sweep over next-state grid index kp = 1..K and pick the
            # argmax of  stage_reward + V_next[kp, ir, is].
            best_val  = Tnum(-Inf)
            best_kp   = Int32(1)
            best_q    = zero(Tnum)
            best_σ    = zero(Tnum)

            for kp in 1:K
                S_tp1 = S_grid[kp, ir]
                # Feasibility: S_{t+1} must be within bounds and the
                # implied flow (q + σ) must be non-negative.
                if S_tp1 >= Smin_i && S_tp1 <= Smax_i
                    flow_total = S_t + inflow - S_tp1
                    if flow_total >= zero(Tnum)
                        # Bang-bang split: q first, σ takes the overflow.
                        q_kp = flow_total > qmax_i ? qmax_i : flow_total
                        σ_kp = flow_total > qmax_i ?
                            (flow_total - qmax_i) : zero(Tnum)

                        # Stage reward:
                        #   revenue + μ_down · (q + σ)
                        #          − μ_self · (S_{t+1} − S_t − inflow + q + σ)
                        mb_residual = (S_tp1 - S_t - inflow) + flow_total
                        stage_r = price_t * η_i * q_kp +
                                  μ_down * flow_total -
                                  μ_self * mb_residual

                        cand = stage_r + V_next[kp, ir, is]
                        if cand > best_val
                            best_val = cand
                            best_kp  = Int32(kp)
                            best_q   = q_kp
                            best_σ   = σ_kp
                        end
                    end
                end
            end

            V_curr[ik, ir, is]    = best_val
            π_q[ik, t, ir, is]    = best_q
            π_σ[ik, t, ir, is]    = best_σ
            π_next[ik, t, ir, is] = best_kp
        else
            # BufferReservoir: no state. V is identically zero; the
            # (q, σ) decisions are filled in by the forward-simulate
            # kernel from the realised upstream flow.
            V_curr[ik, ir, is]    = zero(Tnum)
            π_q[ik, t, ir, is]    = zero(Tnum)
            π_σ[ik, t, ir, is]    = zero(Tnum)
            π_next[ik, t, ir, is] = Int32(ik)
        end
    end
end

"""
    bellman_dp_terminal!(V_curr, S_grid, λ, reg_mask)

Initialise the per-stage value buffer at stage `T+1` to the terminal
cost `V_{T+1}(S) = −λ_i · S`. For `BufferReservoir`, `V` stays zero.
Launch over `(K, n_reg, n_scen)` — the terminal cost is identical
across scenarios, but writing once per thread is simpler than
broadcasting from a (K, n_reg) buffer.
"""
@kernel function bellman_dp_terminal!(
        V_curr,
        @Const(S_grid), @Const(λ), @Const(reg_mask),
    )
    ik, ir, is = @index(Global, NTuple)
    Tnum = eltype(V_curr)
    @inbounds begin
        V_curr[ik, ir, is] = reg_mask[ir] ?
            -λ[ir] * S_grid[ik, ir] : zero(Tnum)
    end
end
