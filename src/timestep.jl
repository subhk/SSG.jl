# src/timestep.jl
# Time stepping schemes

"""
    step_RK3!(dom::Domain, fld::Fields, params::Params, dt::Real)
Advance the system by one time step using third-order Runge-Kutta.

This is a low-storage RK3 scheme that only requires one extra storage array.
The scheme is:
- Stage 1: k₁ = f(u⁰), u¹ = u⁰ + (dt/3)k₁
- Stage 2: k₂ = f(u¹), u² = u⁰ + (dt·15/16)k₂ - (dt·5/9)k₁  
- Stage 3: k₃ = f(u²), u³ = u⁰ + (dt·8/15)k₃ - (dt·153/128)k₂

# Arguments
- `dom`: Domain structure
- `fld`: Fields structure (buoyancy b is advanced)
- `params`: Parameters
- `dt`: Time step size
"""
function step_RK3!(dom::Domain, fld::Fields, params::Params, dt::Real)
    # Store initial buoyancy
    b0 = copy(fld.b)
    
    # RK3 coefficients
    a = [0.0, -5/9, -153/128]      # Coefficients for previous stages
    b = [1/3, 15/16, 8/15]         # Coefficients for current RHS
    
    # Storage for previous RHS
    rhs_prev = copy(fld.tmp)
    rhs_prev .= 0
    
    for stage = 1:3
        # Compute RHS of buoyancy equation
        rhs_buoyancy!(fld.tmp2, dom, fld, params)
        
        # Update buoyancy using RK3 formula
        if stage == 1
            @. fld.b = b0 + b[stage] * dt * fld.tmp2
            @. rhs_prev = fld.tmp2
        else
            @. fld.b = b0 + b[stage] * dt * fld.tmp2 + a[stage] * dt * rhs_prev
            @. rhs_prev = fld.tmp2
        end
        
        # Apply spectral filter if enabled
        if has_filter(params)
            apply_filter_b!(dom, fld, params.filter)
        end
        
        # Solve for streamfunction and update velocities
        solve_mongeampere!(dom, fld; 
                          tol=params.MA_tol, 
                          maxiter=params.MA_maxiter,
                          verbose=false)
        SG_velocities!(dom, fld)
    end
    
    return nothing
end

"""
    step_euler!(dom::Domain, fld::Fields, params::Params, dt::Real)
Simple forward Euler step (mainly for testing/debugging).

# Arguments  
- `dom`: Domain structure
- `fld`: Fields structure
- `params`: Parameters
- `dt`: Time step size
"""
function step_euler!(dom::Domain, fld::Fields, params::Params, dt::Real)
    # Compute RHS
    rhs_buoyancy!(fld.tmp, dom, fld, params)
    
    # Update buoyancy: b^{n+1} = b^n + dt * RHS
    @. fld.b += dt * fld.tmp
    
    # Apply filter if enabled
    if has_filter(params)
        apply_filter_b!(dom, fld, params.filter)
    end
    
    # Update streamfunction and velocities
    solve_mongeampere!(dom, fld; 
                      tol=params.MA_tol, 
                      maxiter=params.MA_maxiter,
                      verbose=false)
    SG_velocities!(dom, fld)
    
    return nothing
end

"""
    adaptive_timestep(dom::Domain, fld::Fields, params::Params, dt_current::Real; 
                     cfl_target=0.5, safety=0.9) -> Real
Compute an adaptive time step based on CFL condition.

# Arguments
- `dom`: Domain structure  
- `fld`: Fields structure
- `params`: Parameters
- `dt_current`: Current time step
- `cfl_target`: Target CFL number
- `safety`: Safety factor (< 1)

# Returns  
- New time step size
"""
function adaptive_timestep(dom::Domain, fld::Fields, params::Params, dt_current::Real;
                          cfl_target=0.5, safety=0.9)
    
    # Compute maximum velocity magnitude
    u_max_local = maximum(abs.(fld.u))
    v_max_local = maximum(abs.(fld.v))
    vel_max_local = max(u_max_local, v_max_local)
    
    # Global maximum
    vel_max_global = MPI.Allreduce(vel_max_local, MPI.MAX, dom.pr.comm)
    
    # Avoid division by zero
    if vel_max_global < 1e-12
        return dt_current
    end
    
    # Grid spacing
    dx = dom.Lx / dom.Nx
    dy = dom.Ly / dom.Ny
    h_min = min(dx, dy)
    
    # CFL-based time step
    dt_cfl = safety * cfl_target * h_min / vel_max_global
    
    # Don't change too rapidly
    dt_new = min(dt_cfl, 1.2 * dt_current)  # Don't increase by more than 20%
    dt_new = max(dt_new, 0.5 * dt_current)  # Don't decrease by more than 50%
    
    return dt_new
end

"""
    estimate_max_dt(dom::Domain, fld::Fields; cfl_max=1.0) -> Real
Estimate maximum stable time step based on current state.

# Arguments
- `dom`: Domain structure
- `fld`: Fields structure  
- `cfl_max`: Maximum CFL number

# Returns
- Estimated maximum time step
"""
function estimate_max_dt(dom::Domain, fld::Fields; cfl_max=1.0)
    # Maximum velocity
    u_max = maximum(abs.(fld.u))
    v_max = maximum(abs.(fld.v))
    vel_max = MPI.Allreduce(max(u_max, v_max), MPI.MAX, dom.pr.comm)
    
    # Grid spacing
    dx = dom.Lx / dom.Nx  
    dy = dom.Ly / dom.Ny
    h_min = min(dx, dy)
    
    # CFL constraint
    if vel_max > 1e-12
        dt_max = cfl_max * h_min / vel_max
    else
        dt_max = Inf
    end
    
    return dt_max
end