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
                          Φ::PencilArray{T, 3}, 
                          bₛ::PencilArray{T, 2}, 
                          fields::Fields{T}, 
                          domain::Domain) where T
    
    # getting streamfunction at the surface
    extract_surface_to_2d!(fields.φₛ, Φ, domain)    

    # Use the mutating jacobian! function from transforms.jl
    jacobian!(db_dt, fields.φₛ, bₛ, domain, 
            fields.tmpc, fields.tmpc2, 
            fields.tmp2, fields.tmp3, 
            fields.tmpc)
    
    # Apply negative sign for advection: ∂b/∂t = -J(ψ,b)
    db_dt.data .*= -1
    
    return db_dt
end


"""
2D version of Jacobian computation for surface fields
"""
function jacobian_2d!(output, a, b, domain::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2, jac_spec)
    # Transform fields to spectral space (2D transforms)
    rfft_2d!(domain, a, tmp_spec1)  # φ̂
    rfft_2d!(domain, b, tmp_spec2)  # b̂
    
    # Compute ∂a/∂x = ∂φ/∂x
    ddx_2d!(domain, tmp_spec1, jac_spec)
    irfft_2d!(domain, jac_spec, tmp_real1)  # ∂φ/∂x in physical space
    
    # Compute ∂a/∂y = ∂φ/∂y  
    ddy_2d!(domain, tmp_spec1, jac_spec)
    irfft_2d!(domain, jac_spec, tmp_real2)  # ∂φ/∂y in physical space
    
    # Compute ∂b/∂x
    ddx_2d!(domain, tmp_spec2, jac_spec)
    irfft_2d!(domain, jac_spec, output)     # Use output as temp for ∂b/∂x
    
    # Compute ∂b/∂y  
    ddy_2d!(domain, tmp_spec2, jac_spec)
    irfft_2d!(domain, jac_spec, tmp_spec1)  # Reuse tmp_spec1 as real temp for ∂b/∂y
    
    # Compute Jacobian: J = ∂φ/∂x * ∂b/∂y - ∂φ/∂y * ∂b/∂x
    output_data = output.data
    φx_data = tmp_real1.data      # ∂φ/∂x
    φy_data = tmp_real2.data      # ∂φ/∂y
    bx_data = output.data         # ∂b/∂x (already in output)
    by_data = tmp_spec1.data      # ∂b/∂y (reusing tmp_spec1)
    
    for i in eachindex(output_data)
        output_data[i] = φx_data[i] * by_data[i] - φy_data[i] * bx_data[i]
    end
    
    return nothing
end

"""
2D FFT wrappers for surface fields (extract first z-level from 3D operations)
"""
function rfft_2d!(domain::Domain, field_2d, field_spec_2d)
    # For now, use a simple approach - extend to 3D temporarily
    # This is not the most efficient but will work
    
    # Get data directly
    field_2d_data = field_2d.data
    field_spec_2d_data = field_spec_2d.data
    
    # Use 2D FFT directly on the local data
    # Note: This assumes we're working on a single process for now
    # For MPI, you'd need proper PencilFFTs 2D transforms
    
    plan = FFTW.plan_rfft(field_2d_data)
    field_spec_2d_data .= plan * field_2d_data
    
    return nothing
end

function irfft_2d!(domain::Domain, field_spec_2d, field_2d)
    field_2d_data = field_2d.data
    field_spec_2d_data = field_spec_2d.data
    
    # Get the size for inverse transform
    nx, ny = size(field_2d_data)
    
    plan = FFTW.plan_irfft(field_spec_2d_data, nx, 1)
    field_2d_data .= plan * field_spec_2d_data
    
    return nothing
end

function ddx_2d!(domain::Domain, field_spec_2d, output_spec_2d)
    field_data = field_spec_2d.data
    output_data = output_spec_2d.data
    
    nx, nyc = size(field_data)
    
    for j in 1:nyc
        for i in 1:nx
            kx = i <= length(domain.kx) ? domain.kx[i] : 0.0
            output_data[i, j] = im * kx * field_data[i, j]
        end
    end
    
    return nothing
end

function ddy_2d!(domain::Domain, field_spec_2d, output_spec_2d)
    field_data = field_spec_2d.data
    output_data = output_spec_2d.data
    
    nx, nyc = size(field_data)
    
    for j in 1:nyc
        ky = j <= length(domain.ky) ? domain.ky[j] : 0.0
        for i in 1:nx
            output_data[i, j] = im * ky * field_data[i, j]
        end
    end
    
    return nothing
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

