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
    
    # Apply dealiasing
    rfft_2d!(domain, db_dt, fields.tmpc_2d)
    dealias_2d!(domain, fields.tmpc_2d)
    irfft_2d!(domain, fields.tmpc_2d, db_dt)
    
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
    
    jacobian_2d!(db_dt, φₛ, bₛ, domain, 
                fields.tmpc_2d, fields.tmpc2_2d, 
                fields.tmp,  fields.tmp2)
    
    # Apply negative sign for advection: ∂b/∂t = -J(ψ,b)
    db_dt.data .*= -1
    
    return db_dt
end


"""
    compute_surface_geostrophic_velocities!(fields::Fields{T}, domain::Domain) where T

Compute surface geostrophic velocities from surface streamfunction:
u = -∂ψₛ/∂y,  v = ∂ψₛ/∂x

Stores results in the surface level of 3D velocity fields for compatibility.
"""
function compute_surface_geostrophic_velocities!(fields::Fields{T}, 
                                        domain::Domain) where T

    # Transform surface streamfunction to spectral space (2D)
    rfft_2d!(domain, fields.φₛ, fields.φshat)
    
    # Compute u = -∂φₛ/∂y (2D) - store in temp arrays first
    ddy_2d!(domain, fields.φshat, fields.tmpc_2d)
    irfft_2d!(domain, fields.tmpc_2d, fields.tmp)  # tmp is 2D
    
    # Copy 2D surface velocity u to all levels of 3D velocity field
    u_data = fields.u.data
    tmp_data = fields.tmp.data
    nz = size(u_data, 3)
    
    @inbounds for k = 1:nz
        @views u_data[:, :, k] .= -tmp_data[:, :]  # Apply negative sign
    end
    
    # Compute v = ∂φₛ/∂x (2D)
    ddx_2d!(domain, fields.φshat, fields.tmpc_2d)
    irfft_2d!(domain, fields.tmpc_2d, fields.tmp2)  # tmp2 is 2D
    
    # Copy 2D surface velocity v to all levels of 3D velocity field
    v_data = fields.v.data
    tmp2_data = fields.tmp2.data
    
    @inbounds for k = 1:nz
        @views v_data[:, :, k] .= tmp2_data[:, :]
    end
    
    return nothing
end
