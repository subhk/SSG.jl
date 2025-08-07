
using PencilArray: range_local

"""
    set_b!(prob, b_field, domain::Domain)

Set the buoyancy field and update all derived variables for SSG solver.

# Arguments
- `prob`: Surface Semi-GeostrophicProblem structure
- `b_field`: Buoyancy field (PencilArray or Matrix)
- `domain`: Domain structure from SSG.jl

# Updates
- Sets buoyancy field with zero mean constraint
- Applies spectral dealiasing
- Solves Monge-Ampère equation for streamfunction
- Computes geostrophic velocities
- Calculates initial kinetic energy diagnostics
"""
function set_b!(prob::SemiGeostrophicProblem{T}, b_field, domain::Domain) where T

    # Ensure buoyancy field is properly sized
    @ensuresamegrid(prob.fields.bₛ, b_field)
    
    # Copy buoyancy field to problem structure
    copy_field!(prob.fields.bₛ, b_field)
    
    # Zero mean constraint for spectral methods (critical for periodic domains)
    # Transform to spectral space first
    rfft_2d!(domain, prob.fields.bₛ, prob.fields.bhat)
    
    # Set k=0 mode to zero (removes domain average)
    # Handle PencilArray structure and local ranges
    range_locals = range_local(prob.fields.bhat.pencil)
    bhat_local   = prob.fields.bhat.data
    
    # Check if this process owns the k=0 mode
    if 1 in range_locals[1] && 1 in range_locals[2]
        i_local = findfirst(x -> x == 1, range_locals[1])
        j_local = findfirst(x -> x == 1, range_locals[2])
        if i_local !== nothing && j_local !== nothing
            bhat_local[i_local, j_local, :] .= 0.0
        end
    end
    
    # Apply dealiasing in spectral space
    dealias!(domain, prob.fields.bhat)
    
    # Transform back to physical space
    irfft!(domain, prob.fields.bhat, prob.fields.bₛ)
    
    # Solve Monge-Ampère equation: det(D²φ) = b
    solve_monge_ampere_fields!(prob.fields, domain; 
                              tol=1e-10, 
                              maxiter=20, 
                              verbose=false)
    
    # Compute geostrophic velocities from streamfunction
    compute_geostrophic_velocities!(prob.fields, domain)
    
    # Calculate surface kinetic energy diagnostics
    # Using spectral energy calculation for accuracy
    ke_total = compute_kinetic_energy(prob.fields, domain)
    
    # Alternative: Direct spectral calculation as in original
    # Transform velocities to spectral space
    rfft!(domain, prob.fields.u, prob.fields.tmpc)
    ke_u = 0.5 * parsevalsum2(prob.fields.tmpc, domain)
    
    rfft!(domain, prob.fields.v, prob.fields.tmpc)
    ke_v = 0.5 * parsevalsum2(prob.fields.tmpc, domain)
    
    # Report initial energy (matching original format)
    @printf "initial surface KE 1/2 ∫(u²+v²) dx dy: %f \n" (ke_u + ke_v)
    
    # Update diagnostics if enabled
    if prob.diagnostics !== nothing
        update_diagnostics!(prob.diagnostics, prob.fields, domain, prob.clock)
    end
    
    return nothing
end

"""
    set_φ!(prob, φ_field, domain::Domain)

Set streamfunction field and compute derived buoyancy (alternative initialization).
Based on original setICs.jl set_Φ! function.
"""
function set_φ!(prob::SemiGeostrophicProblem{T}, φ_field, domain::Domain) where T
    # Copy streamfunction field
    copy_field!(prob.fields.φ, φ_field)
    
    # Zero mean constraint - handle PencilArray distribution
    rfft!(domain, prob.fields.φ, prob.fields.φhat)
    range_locals = range_local(prob.fields.φhat.pencil)
    φhat_local = prob.fields.φhat.data
    
    # Check if this process owns the k=0 mode
    if 1 in range_locals[1] && 1 in range_locals[2]
        i_local = findfirst(x -> x == 1, range_locals[1])
        j_local = findfirst(x -> x == 1, range_locals[2])
        if i_local !== nothing && j_local !== nothing
            φhat_local[i_local, j_local, :] .= 0.0
        end
    end
    irfft!(domain, prob.fields.φhat, prob.fields.φ)
    
    # Compute buoyancy from streamfunction using Monge-Ampère relation
    # This would need the inverse relationship - simplified here
    rfft!(domain, prob.fields.φ, prob.fields.φhat)
    laplacian_h!(domain, prob.fields.φhat, prob.fields.bhat)  # ∇²φ
    irfft!(domain, prob.fields.bhat, prob.fields.bₛ)
    
    # Apply zero mean to buoyancy as well - handle PencilArray distribution
    rfft!(domain, prob.fields.bₛ, prob.fields.bhat)
    range_locals = range_local(prob.fields.bhat.pencil)
    bhat_local = prob.fields.bhat.data
    
    # Check if this process owns the k=0 mode
    if 1 in range_locals[1] && 1 in range_locals[2]
        i_local = findfirst(x -> x == 1, range_locals[1])
        j_local = findfirst(x -> x == 1, range_locals[2])
        if i_local !== nothing && j_local !== nothing
            bhat_local[i_local, j_local, :] .= 0.0
        end
    end
    irfft!(domain, prob.fields.bhat, prob.fields.b)
    
    # Compute velocities and diagnostics
    compute_geostrophic_velocities!(prob.fields, domain)
    
    # Energy diagnostics
    ke_total = compute_kinetic_energy(prob.fields, domain)
    b_stats  = field_stats(prob.fields.bₛ)
    
    @printf "min/max value of u: %f %f\n" extrema(prob.fields.u.data)...
    @printf "min/max value of v: %f %f\n" extrema(prob.fields.v.data)...
    @printf "initial surface KE 1/2 ∫(u²+v²) dx dy: %f \n" ke_total
    @printf "initial surface b: %f %f \n" b_stats[3:4]...
    
    return nothing
end

"""
    solve_monge_ampere_fields!(fields::Fields, domain::Domain; kwargs...)

Solve Monge-Ampère equation using SSG.jl's multigrid solver.
Wrapper to integrate with the fields structure.
"""
function solve_monge_ampere_fields!(fields::Fields{T}, 
                                   domain::Domain;
                                   tol::T=T(1e-10),
                                   maxiter::Int=20,
                                   verbose::Bool=false,
                                   ε::T=T(0,1)) where T

    solution, diagnostics = solve_ssg_equation(fields.φ, 
                                      fields.b, 
                                      ε, 
                                      domain;
                                      tol=1e-8,
                                      verbose=(rank == 0),
                                      smoother=:spectral)
    
    # Copy solution back to fields
    copy_field!(fields.φ, solution)
    
    if verbose && !diagnostics.converged
        @warn "Monge-Ampère solver did not converge: $(diagnostics.final_residual)"
    end
    
    return diagnostics.converged
end

"""
    compute_geostrophic_velocities!(fields::Fields, domain::Domain)

Compute geostrophic velocities from streamfunction using spectral derivatives.
u = -∂φ/∂y, v = ∂φ/∂x
"""
function compute_geostrophic_velocities!(fields::Fields{T}, domain::Domain) where T
    # Transform streamfunction to spectral space
    rfft!(domain, fields.φ, fields.φhat)
    
    # Compute u = -∂φ/∂y
    ddy!(domain, fields.φhat, fields.tmpc)
    irfft!(domain, fields.tmpc, fields.u)
    fields.u.data .*= -1  # Apply negative sign
    
    # Compute v = ∂φ/∂x  
    ddx!(domain, fields.φhat, fields.tmpc)
    irfft!(domain, fields.tmpc, fields.v)
    
    return nothing
end

"""
    compute_kinetic_energy(fields::Fields, domain::Domain) -> Real

Compute total kinetic energy using spectral methods.
"""
function compute_kinetic_energy(fields::Fields{T}, domain::Domain) where T
    # Transform velocities to spectral space
    rfft!(domain, fields.u, fields.tmpc)
    ke_u = 0.5 * parsevalsum2(fields.tmpc, domain)
    
    rfft!(domain, fields.v, fields.tmpc)
    ke_v = 0.5 * parsevalsum2(fields.tmpc, domain)
    
    return ke_u + ke_v
end

# # Alternative version with more explicit type annotations and error checking
# function set_b!(prob::Problem, sol_b::AbstractArray{T,N}, grid::Grid) where {T<:Number, N}
#     # Input validation
#     @assert size(sol_b) == size(prob.sol_b) "Buoyancy field size mismatch"
    
#     # Zero mean constraint for spectral methods
#     if ndims(sol_b) >= 2
#         sol_b[1,1] = zero(T)
#     end
    
#     # Update problem state
#     prob.sol_b .= sol_b
#     dealias!(prob.sol_b, grid)
    
#     # Streamfunction calculation
#     calcΦ!(prob.sol_Φ3d, prob.sol_b, prob.vars, grid)
    
#     # Extract surface streamfunction
#     if ndims(prob.sol_Φ3d) == 3
#         prob.sol_Φ .= view(prob.sol_Φ3d, :, :, grid.nz)
#     end
    
#     # Update derived quantities
#     updatevars!(prob)
    
#     # Energy diagnostics with proper normalization
#     domain_area = grid.Lx * grid.Ly  # or appropriate grid area calculation
#     ke_total = 0.5 * (parsevalsum2(prob.vars.uh, grid) + 
#                       parsevalsum2(prob.vars.vh, grid)) / domain_area
    
#     @info "Initial surface kinetic energy density: $(ke_total)"
    
#     return ke_total  # Return for potential use in diagnostics
# end
