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
###############################################################################

"""
Compute Jacobian J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x
"""
function compute_jacobian!(db_dt::PencilArray{T, 2}, 
                          ψ::PencilArray{T, 2}, 
                          b::PencilArray{T, 2}, 
                          fields::Fields{T}, 
                          domain::Domain) where T
    
    # Use the mutating jacobian! function from transforms.jl
    jacobian!(db_dt, ψ, b, domain, 
            fields.tmpc, fields.tmpc2, 
            fields.tmp2, fields.tmp3, 
            fields.tmpc)
    
    # Apply negative sign for advection: ∂b/∂t = -J(ψ,b)
    db_dt.data .*= -1
    
    return db_dt
end

"""
Compute buoyancy tendency for surface semi-geostrophic equations
Only evolves SURFACE buoyancy - streamfunction is diagnostic
"""
function compute_tendency!(db_dt::PencilArray{T, 2}, 
                          fields::Fields{T}, 
                          domain::Domain, 
                          params::TimeParams{T}) where T
    
    # Solve Monge-Ampère equation
    solve_monge_ampere_fields!(fields, domain)
    
    # Compute Jacobian
    compute_jacobian!(db_dt, fields.φ, fields.b, fields, domain)
    
    # Apply dealiasing
    rfft!(domain, db_dt, fields.tmpc)
    dealias!(domain, fields.tmpc)
    irfft!(domain, fields.tmpc, db_dt)
    
    return db_dt
end


"""
Compute geostrophic velocities from streamfunction
"""
function compute_geostrophic_velocities!(fields::Fields{T}, domain::Domain) where T
    # Transform streamfunction to spectral space
    rfft!(domain, fields.φ, fields.φhat)
    
    # Compute u = -∂φ/∂y
    ddy!(domain, fields.φhat, fields.tmpc)
    irfft!(domain, fields.tmpc, fields.u)
    fields.u.data .*= -1
    
    # Compute v = ∂φ/∂x  
    ddx!(domain, fields.φhat, fields.tmpc)
    irfft!(domain, fields.tmpc, fields.v)
    
    return nothing
end

