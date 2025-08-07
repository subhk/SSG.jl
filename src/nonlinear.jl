###############################################################################
# SURFACE SEMI-GEOSTROPHIC EQUATIONS
###############################################################################
#
# Governing equation for buoyancy:
# ∂b/∂t + J(ψ, b) = 0                    (buoyancy conservation)
# u = -∂ψ/∂y,  v = ∂ψ/∂x                 (geostrophic velocities)
#
# Where:
# - b(x,y,t): surface buoyancy anomaly
# - ψ(x,y,t): geostrophic streamfunction  
# - J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x: Jacobian (advection term)
# - D²ψ: Hessian matrix of ψ
#
# TIME INTEGRATION OPTIONS (from timestep.jl):
# 1. Low-storage 2nd order Adams-Bashforth (AB2-LS)
# 2. 3rd order Runge-Kutta (RK3)
# 3. Spectral filtering for numerical stability
# This version uses proper PencilFFT 2D operations for surface fields.
###############################################################################

"""
Compute Jacobian J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x
"""
function compute_jacobian!(db_dt::PencilArray{T, 2}, 
                          φₛ::PencilArray{T, 2}, 
                          bₛ::PencilArray{T, 2}, 
                          fields::Fields{T}, 
                          domain::Domain) where T
    
    # # getting streamfunction at the surface
    # extract_surface_to_2d!(fields.φₛ, Φ, domain)    

    #
    jacobian_2d!(db_dt, φₛ, bₛ, domain, 
                fields.tmpc, fields.tmpc2, 
                fields.tmp,  fields.tmp2)
    
    # Apply negative sign for advection: ∂b/∂t = -J(ψ,b)
    db_dt.data .*= -1
    
    return db_dt
end


"""
Compute buoyancy tendency for surface semi-geostrophic equations
Only evolves SURFACE buoyancy - streamfunction is diagnostic
```math
    ∂b/∂t + J(ψ, b) = 0
    u = -∂ψ/∂y,  v = ∂ψ/∂x 
"""
function compute_tendency!(db_dt::PencilArray{T, 2}, 
                          fields::Fields{T}, 
                          domain::Domain, 
                          params::TimeParams{T}) where T
    
    # Compute Jacobian
    compute_jacobian!(db_dt, fields.φ, fields.bₛ, fields, domain)
    
    # # Apply dealiasing
    # rfft!(domain, db_dt, fields.tmpc)
    # dealias!(domain, fields.tmpc)
    # irfft!(domain, fields.tmpc, db_dt)
    
    return db_dt
end


# """
# Compute geostrophic velocities from streamfunction
# """
# function compute_geostrophic_velocities!(fields::Fields{T}, domain::Domain) where T
#     # Transform streamfunction to spectral space
#     rfft!(domain, fields.φ, fields.φhat)
    
#     # Compute u = -∂φ/∂y
#     ddy!(domain, fields.φhat, fields.tmpc)
#     irfft!(domain, fields.tmpc, fields.u)
#     fields.u.data .*= -1
    
#     # Compute v = ∂φ/∂x  
#     ddx!(domain, fields.φhat, fields.tmpc)
#     irfft!(domain, fields.tmpc, fields.v)
    
#     return nothing
# end


"""
    compute_geostrophic_velocities_2d!(u, v, ψ, surface_domain::SurfaceDomain, tmp_spec1, tmp_spec2)

Compute 2D geostrophic velocities: u = -∂ψ/∂y, v = ∂ψ/∂x.
"""
function compute_geostrophic_velocities_2d!(u, v, ψ, domain::Domain, tmp_spec1, tmp_spec2)
    # Transform streamfunction to spectral space
    rfft_2d!(domain, ψ, tmp_spec1)  # ψ̂
    
    # Compute u = -∂ψ/∂y
    ddy_2d!( domain,  tmp_spec1, tmp_spec2)   # ∂ψ̂/∂y
    irfft_2d!(domain, tmp_spec2, u)           # u = ∂ψ/∂y
    u.data .*= -1  # Apply negative sign
    
    # Compute v = ∂ψ/∂x  
    ddx_2d!(  domain, tmp_spec1, tmp_spec2)   # ∂ψ̂/∂x
    irfft_2d!(domain, tmp_spec2, v)           # v = ∂ψ/∂x
    
    return (u, v)
end
