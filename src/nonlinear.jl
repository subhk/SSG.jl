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
###############################################################################

"""
Compute buoyancy tendency for surface semi-geostrophic equations
Only evolves surface buoyancy - streamfunction is diagnostic
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
    
    # Apply dealiasing
    rfft_2d!(domain, db_dt, fields.tmpc_2d)
    dealias_2d!(domain, fields.tmpc_2d)
    irfft_2d!(domain, fields.tmpc_2d, db_dt)
    
    return db_dt
end


"""
Compute Jacobian J(ψₛ,bₛ) = ∂ψₛ/∂x ∂bₛ/∂y - ∂ψₛ/∂y ∂bₛ/∂x
"""
function compute_jacobian!(db_dt::PencilArray{T, 2}, 
                          φₛ::PencilArray{T, 2}, 
                          bₛ::PencilArray{T, 2}, 
                          fields::Fields{T}, 
                          domain::Domain) where T
    
    jacobian_2d!(db_dt, φₛ, bₛ, domain, 
                fields.tmpc_2d, fields.tmpc2_2d, 
                fields.tmp,  fields.tmp2)
    
    # Apply negative sign for advection: ∂b/∂t = -J(ψ,b)
    db_dt.data .*= -1
    
    return nothing
end

