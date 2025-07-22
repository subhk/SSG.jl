# src/monge_ampere.jl
# Monge-Ampère equation solver

"""
    solve_mongeampere!(dom::Domain, fld::Fields; tol=1e-8, maxiter=30, verbose=false) -> Bool
Solve the Monge-Ampère equation for the streamfunction φ given buoyancy b.

The equation is: (1 + φ_xx)(1 + φ_yy) - φ_xy² = b/N²

We assume N² = 1 for simplicity. This is a simplified iterative solver.
For production use, implement a proper Newton-Krylov method.

# Arguments
- `dom`: Domain structure
- `fld`: Fields structure (φ is modified, b is input)
- `tol`: Convergence tolerance for residual norm
- `maxiter`: Maximum number of iterations
- `verbose`: Print convergence information

# Returns
- `Bool`: true if converged, false otherwise
"""
function solve_mongeampere!(dom::Domain, fld::Fields; 
                           tol=1e-8, maxiter=30, verbose=false)
    
    rank = MPI.Comm_rank(dom.pr.comm)
    
    for iter = 1:maxiter
        # Compute residual
        calc_MA_residual!(dom, fld)
        
        # Compute global residual norm
        res_local = sum(abs2, fld.R)
        res_global = MPI.Allreduce(res_local, MPI.SUM, dom.pr.comm)
        res_norm = sqrt(res_global) / sqrt(dom.Nx * dom.Ny)
        
        if verbose && rank == 0
            @printf("[MA] iter %2d: residual = %.3e\n", iter, res_norm)
        end
        
        # Check convergence
        if res_norm < tol
            verbose && rank == 0 && println("[MA] Converged!")
            return true
        end
        
        # Simple update: φ += α * R (very basic, but stable)
        α = 0.01  # Small step size for stability
        @. fld.φ += α * fld.R
    end
    
    if verbose && rank == 0
        @warn "[MA] Failed to converge in $maxiter iterations"
    end
    
    return false  # Did not converge, but continue anyway
end

"""
    calc_MA_residual!(dom::Domain, fld::Fields)
Compute the residual of the Monge-Ampère equation:
R = (1 + φ_xx)(1 + φ_yy) - φ_xy² - b

The residual is stored in fld.R.
"""
function calc_MA_residual!(dom::Domain, fld::Fields)
    # Use scratch arrays for second derivatives
    φ_xx = fld.tmp      # ∂²φ/∂x²
    φ_yy = fld.tmp2     # ∂²φ/∂y²
    φ_xy = fld.R        # ∂²φ/∂x∂y (temporarily use R)
    
    # Compute second derivatives
    second_derivatives!(dom, fld.φ, fld.φhat, φ_xx, φ_yy, φ_xy, fld.tmpc)
    
    # Store φ_xy values before overwriting R
    φ_xy_vals = copy(φ_xy)
    
    # Compute residual: (1 + φ_xx)(1 + φ_yy) - φ_xy² - b
    @. fld.R = (1 + φ_xx) * (1 + φ_yy) - φ_xy_vals^2 - fld.b
    
    return fld.R
end

"""
    initialize_streamfunction!(dom::Domain, fld::Fields)
Initialize streamfunction with a simple approximation.
For small buoyancy, φ ≈ -b (geostrophic approximation).
"""
function initialize_streamfunction!(dom::Domain, fld::Fields)
    # Simple initialization: φ = -b (works for small amplitude)
    @. fld.φ = -fld.b
    return fld.φ
end

"""
    MA_energy(dom::Domain, fld::Fields) -> Real
Compute the "energy" associated with the Monge-Ampère residual.
This can be used to monitor convergence.
"""
function MA_energy(dom::Domain, fld::Fields)
    calc_MA_residual!(dom, fld)
    
    # Local contribution
    energy_local = 0.5 * sum(abs2, fld.R)
    
    # Global sum
    energy_global = MPI.Allreduce(energy_local, MPI.SUM, dom.pr.comm)
    
    return energy_global / (dom.Nx * dom.Ny)
end

"""
    check_MA_constraint(dom::Domain, fld::Fields; verbose=false) -> Real
Check how well the Monge-Ampère constraint is satisfied.
Returns the normalized residual norm.
"""
function check_MA_constraint(dom::Domain, fld::Fields; verbose=false)
    calc_MA_residual!(dom, fld)
    
    # Compute global residual norm
    res_local = sum(abs2, fld.R)
    res_global = MPI.Allreduce(res_local, MPI.SUM, dom.pr.comm)
    res_norm = sqrt(res_global) / sqrt(dom.Nx * dom.Ny)
    
    if verbose && MPI.Comm_rank(dom.pr.comm) == 0
        @printf("MA constraint residual: %.3e\n", res_norm)
    end
    
    return res_norm
end