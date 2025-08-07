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
    compute_jacobian!(db_dt, fields.φₛ, fields.bₛ, fields, domain)
    
    # # Apply dealiasing
    # rfft!(domain, db_dt, fields.tmpc)
    # dealias!(domain, fields.tmpc)
    # irfft!(domain, fields.tmpc, db_dt)
    
    return db_dt
end


"""
Compute Jacobian J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x
"""
function compute_jacobian!(db_dt::PencilArray{T, 2}, 
                          φₛ::PencilArray{T, 2}, 
                          bₛ::PencilArray{T, 2}, 
                          fields::Fields{T}, 
                          domain::Domain) where T
    
    #
    jacobian_2d!(db_dt, φₛ, bₛ, domain, 
                fields.tmpc_2d, fields.tmpc2_2d, 
                fields.tmp,  fields.tmp2)
    
    # Apply negative sign for advection: ∂b/∂t = -J(ψ,b)
    db_dt.data .*= -1
    
    return db_dt
end


"""
    compute_surface_geostrophic_velocities_2d!(u, v, ψ, surface_domain::SurfaceDomain, tmp_spec1, tmp_spec2)

Compute 2D surface geostrophic velocities: u = -∂ψ/∂y, v = ∂ψ/∂x.
"""
function compute_surface_geostrophic_velocities!(fields::Fields{T}, 
                                        domain::Domain) where T

    # Use 2D transforms for surface fields
    rfft_2d!(domain, fields.φₛ, fields.φshat)
    
    # Compute u = -∂φ/∂y (2D)
    ddy_2d!(domain, fields.φshat, fields.tmpc_2d)
    irfft_2d!(domain, fields.tmpc_2d, fields.u)
    fields.u.data .*= -1
    
    # Compute v = ∂φ/∂x (2D)
    ddx_2d!(domain, fields.φshat, fields.tmpc_2d)
    irfft_2d!(domain, fields.tmpc_2d, fields.v)
    
    return nothing
end
