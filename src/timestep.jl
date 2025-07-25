# src/timestep.jl
# Time integration for surface semi-geostrophic equations
# Supports low-storage 2nd order Adams-Bashforth and 3rd order Runge-Kutta
# with spectral filtering and Monge-Ampère inversion

using PencilArrays
using PencilFFTs
using LinearAlgebra
using Printf

# # Import required modules
# include("transforms.jl")
# include("fields.jl")

###############################################################################
# SURFACE SEMI-GEOSTROPHIC EQUATIONS
###############################################################################
#
# GOVERNING EQUATIONS:
# ∂b/∂t + J(ψ, b) = 0                    (buoyancy conservation)
# det(D²ψ) = b                           (Monge-Ampère equation for streamfunction)
# u = -∂ψ/∂y,  v = ∂ψ/∂x               (geostrophic velocities)
#
# Where:
# - b(x,y,t): surface buoyancy anomaly
# - ψ(x,y,t): geostrophic streamfunction  
# - J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x: Jacobian (advection term)
# - D²ψ: Hessian matrix of ψ
#
# TIME INTEGRATION OPTIONS:
# 1. Low-storage 2nd order Adams-Bashforth (AB2-LS)
# 2. 3rd order Runge-Kutta (RK3)
# 3. Spectral filtering for numerical stability
#
###############################################################################

# ============================================================================
# TIME INTEGRATION PARAMETERS AND STRUCTURES
# ============================================================================

"""
Time integration scheme selector
"""
@enum TimeScheme begin
    AB2_LowStorage  # 2nd order Adams-Bashforth (low storage)
    RK3            # 3rd order Runge-Kutta
    RK3_LowStorage # 3rd order Runge-Kutta (low storage variant)
end

"""
Time integration parameters
"""
struct TimeParams{T<:AbstractFloat}
    dt::T                    # time step
    scheme::TimeScheme       # integration scheme
    filter_freq::Int         # frequency of spectral filtering (every N steps)
    filter_strength::T       # spectral filter strength (0 = no filter, 1 = strong)
    cfl_safety::T           # CFL safety factor
    max_dt::T               # maximum allowed time step
    min_dt::T               # minimum allowed time step
    adaptive_dt::Bool       # adaptive time stepping
    
    function TimeParams{T}(dt::T; 
                          scheme::TimeScheme=AB2_LowStorage,
                          filter_freq::Int=10,
                          filter_strength::T=T(0.1),
                          cfl_safety::T=T(0.5),
                          max_dt::T=T(0.1),
                          min_dt::T=T(1e-6),
                          adaptive_dt::Bool=false) where T
        new{T}(dt, scheme, filter_freq, filter_strength, 
               cfl_safety, max_dt, min_dt, adaptive_dt)
    end
end

"""
Time integration state for multi-step methods
"""
mutable struct TimeState{T, PA}
    t::T                     # current time
    step::Int               # current step number
    
    # For Adams-Bashforth (stores previous tendency)
    db_dt_old::PA           # previous time derivative of buoyancy
    
    # For Runge-Kutta (intermediate stages)
    b_stage::PA             # intermediate buoyancy for RK stages
    k1::PA                  # RK stage derivatives
    k2::PA
    k3::PA
    
    # Diagnostic
    dt_actual::T            # actual dt used (for adaptive stepping)
    cfl_max::T              # maximum CFL number
    
    function TimeState{T, PA}(initial_time::T, fields::Fields{T}) where {T, PA}
        t = initial_time
        step = 0
        
        # Allocate storage for time integration
        db_dt_old = similar(fields.b)
        b_stage = similar(fields.b)
        k1 = similar(fields.b)
        k2 = similar(fields.b)
        k3 = similar(fields.b)
        
        # Initialize
        zero_field!(db_dt_old)
        zero_field!(b_stage)
        zero_field!(k1)
        zero_field!(k2)
        zero_field!(k3)
        
        dt_actual = T(0)
        cfl_max = T(0)
        
        new{T, PA}(t, step, db_dt_old, b_stage, k1, k2, k3, dt_actual, cfl_max)
    end
end

# ============================================================================
# ADVECTION OPERATORS
# ============================================================================

"""
Compute Jacobian J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x spectrally
"""
function compute_jacobian!(db_dt::PencilArray{T, 2}, ψ::PencilArray{T, 2}, b::PencilArray{T, 2}, 
                          fld::Fields{T}, dom::Domain) where T
    
    # Compute ∂ψ/∂x
    rfft!(dom, ψ, fld.φhat)
    ddx!(dom, fld.φhat, fld.tmpc)
    irfft!(dom, fld.tmpc, fld.tmp)  # tmp = ∂ψ/∂x
    
    # Compute ∂ψ/∂y
    rfft!(dom, ψ, fld.φhat)
    ddy!(dom, fld.φhat, fld.tmpc)
    irfft!(dom, fld.tmpc, fld.tmp2)  # tmp2 = ∂ψ/∂y
    
    # Compute ∂b/∂x
    rfft!(dom, b, fld.bhat)
    ddx!(dom, fld.bhat, fld.tmpc)
    irfft!(dom, fld.tmpc, fld.tmp3)  # tmp3 = ∂b/∂x
    
    # Compute ∂b/∂y
    rfft!(dom, b, fld.bhat)
    ddy!(dom, fld.bhat, fld.tmpc)
    irfft!(dom, fld.tmpc, fld.u)  # u = ∂b/∂y (reuse velocity array)
    
    # Compute Jacobian: J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x
    tmp_data = fld.tmp.data    # ∂ψ/∂x
    tmp2_data = fld.tmp2.data  # ∂ψ/∂y
    tmp3_data = fld.tmp3.data  # ∂b/∂x
    u_data = fld.u.data        # ∂b/∂y
    db_dt_data = db_dt.data
    
    @inbounds @simd for i in eachindex(db_dt_data)
        db_dt_data[i] = tmp_data[i] * u_data[i] - tmp2_data[i] * tmp3_data[i]
    end
    
    # Apply negative sign for advection: ∂b/∂t = -J(ψ,b)
    db_dt.data .*= -1
    
    return db_dt
end

"""
Compute tendency for surface semi-geostrophic equations
"""
function compute_tendency!(db_dt::PencilArray{T, 2}, fld::Fields{T}, 
                          dom::Domain, params::TimeParams{T}) where T
    
    # Solve Monge-Ampère equation: det(D²ψ) = b
    # This updates fld.φ given current fld.b
    solve_monge_ampere_fields!(fld; tol=1e-10, verbose=false)
    
    # Compute Jacobian J(ψ,b) for advection
    compute_jacobian!(db_dt, fld.φ, fld.b, fld, dom)
    
    # Apply spectral dealiasing to tendency
    rfft!(dom, db_dt, fld.tmpc)
    dealias!(dom, fld.tmpc)
    irfft!(dom, fld.tmpc, db_dt)
    
    return db_dt
end

# ============================================================================
# CFL CONDITION AND ADAPTIVE TIME STEPPING
# ============================================================================

"""
Compute CFL number for current state
"""
function compute_cfl_number(fld::Fields{T}, dom::Domain, dt::T) where T
    # Compute geostrophic velocities
    compute_geostrophic_velocities!(fld)
    
    # Get velocity magnitudes
    u_data = fld.u.data
    v_data = fld.v.data
    
    # Compute maximum velocity
    u_max_local = maximum(abs, u_data)
    v_max_local = maximum(abs, v_data)
    
    # MPI reduction to get global maximum
    u_max = MPI.Allreduce(u_max_local, MPI.MAX, fld.domain.pc.comm)
    v_max = MPI.Allreduce(v_max_local, MPI.MAX, fld.domain.pc.comm)
    
    # Grid spacing
    dx = dom.Lx / dom.Nx
    dy = dom.Ly / dom.Ny
    
    # CFL number
    cfl_x = u_max * dt / dx
    cfl_y = v_max * dt / dy
    cfl = max(cfl_x, cfl_y)
    
    return cfl
end

"""
Adaptive time step calculation
"""
function adaptive_timestep(fld::Fields{T}, dom::Domain, 
                          params::TimeParams{T}, state::TimeState{T}) where T
    
    if !params.adaptive_dt
        return params.dt
    end
    
    # Compute current CFL number
    cfl = compute_cfl_number(fld, dom, params.dt)
    state.cfl_max = cfl
    
    # Adjust time step based on CFL condition
    if cfl > params.cfl_safety
        # Reduce time step
        dt_new = params.dt * params.cfl_safety / cfl
        dt_new = max(dt_new, params.min_dt)
    elseif cfl < 0.5 * params.cfl_safety
        # Increase time step (conservative)
        dt_new = params.dt * 1.1
        dt_new = min(dt_new, params.max_dt)
    else
        dt_new = params.dt
    end
    
    return dt_new
end

# ============================================================================
# TIME INTEGRATION SCHEMES
# ============================================================================

"""
Low-storage 2nd order Adams-Bashforth time step
"""
function timestep_ab2_ls!(fld::Fields{T}, dom::Domain, 
                         params::TimeParams{T}, state::TimeState{T}) where T
    
    dt = adaptive_timestep(fld, dom, params, state)
    state.dt_actual = dt
    
    # Compute current tendency
    compute_tendency!(fld.tmp, fld, dom, params)  # tmp = db/dt at current time
    
    if state.step == 0
        # First step: Forward Euler
        b_data = fld.b.data
        tmp_data = fld.tmp.data
        @inbounds @simd for i in eachindex(b_data)
            b_data[i] += dt * tmp_data[i]
        end
        
        # Store tendency for next step
        copy_field!(state.db_dt_old, fld.tmp)
    else
        # Adams-Bashforth step: b^{n+1} = b^n + dt(3/2 * f^n - 1/2 * f^{n-1})
        b_data = fld.b.data
        tmp_data = fld.tmp.data           # current tendency
        old_data = state.db_dt_old.data   # previous tendency
        
        @inbounds @simd for i in eachindex(b_data)
            b_data[i] += dt * (1.5 * tmp_data[i] - 0.5 * old_data[i])
        end
        
        # Update stored tendency (low-storage: reuse arrays)
        copy_field!(state.db_dt_old, fld.tmp)
    end
    
    # Update time and step
    state.t += dt
    state.step += 1
    
    return dt
end

"""
3rd order Runge-Kutta time step
"""
function timestep_rk3!(fld::Fields{T}, dom::Domain, 
                      params::TimeParams{T}, state::TimeState{T}) where T
    
    dt = adaptive_timestep(fld, dom, params, state)
    state.dt_actual = dt
    
    # Store initial state
    copy_field!(state.b_stage, fld.b)
    
    # RK3 coefficients (classical 3rd order)
    # Stage 1: k1 = f(t_n, y_n)
    compute_tendency!(state.k1, fld, dom, params)
    
    # Update to intermediate state: y1 = y_n + dt/2 * k1
    b_data = fld.b.data
    b_stage_data = state.b_stage.data
    k1_data = state.k1.data
    
    @inbounds @simd for i in eachindex(b_data)
        b_data[i] = b_stage_data[i] + 0.5 * dt * k1_data[i]
    end
    
    # Stage 2: k2 = f(t_n + dt/2, y1)
    compute_tendency!(state.k2, fld, dom, params)
    
    # Update to second intermediate state: y2 = y_n - dt * k1 + 2 * dt * k2
    k2_data = state.k2.data
    @inbounds @simd for i in eachindex(b_data)
        b_data[i] = b_stage_data[i] - dt * k1_data[i] + 2.0 * dt * k2_data[i]
    end
    
    # Stage 3: k3 = f(t_n + dt, y2)
    compute_tendency!(state.k3, fld, dom, params)
    
    # Final update: y_{n+1} = y_n + dt/6 * (k1 + 4*k2 + k3)
    k3_data = state.k3.data
    @inbounds @simd for i in eachindex(b_data)
        b_data[i] = b_stage_data[i] + (dt/6.0) * (k1_data[i] + 4.0*k2_data[i] + k3_data[i])
    end
    
    # Update time and step
    state.t += dt
    state.step += 1
    
    return dt
end

"""
Low-storage 3rd order Runge-Kutta (Williamson variant)
"""
function timestep_rk3_ls!(fld::Fields{T}, dom::Domain, 
                         params::TimeParams{T}, state::TimeState{T}) where T
    
    dt = adaptive_timestep(fld, dom, params, state)
    state.dt_actual = dt
    
    # Low-storage RK3 coefficients (Williamson)
    α = [T(0), T(-5/9), T(-153/128)]
    β = [T(1/3), T(15/16), T(8/15)]
    
    # Initialize: S = 0, U = b^n
    zero_field!(state.k1)  # Use k1 as S (accumulator)
    copy_field!(state.b_stage, fld.b)  # Use b_stage as U
    
    for stage = 1:3
        # Compute tendency: k = f(U)
        compute_tendency!(fld.tmp, fld, dom, params)
        
        # Update accumulator: S = α[stage] * S + k
        k1_data = state.k1.data      # S
        tmp_data = fld.tmp.data      # k
        @inbounds @simd for i in eachindex(k1_data)
            k1_data[i] = α[stage] * k1_data[i] + tmp_data[i]
        end
        
        # Update state: U = U + β[stage] * dt * S
        b_stage_data = state.b_stage.data  # U
        b_data = fld.b.data
        @inbounds @simd for i in eachindex(b_data)
            u_new = b_stage_data[i] + β[stage] * dt * k1_data[i]
            b_stage_data[i] = u_new
            b_data[i] = u_new  # Update working solution
        end
    end
    
    # Update time and step
    state.t += dt
    state.step += 1
    
    return dt
end

# ============================================================================
# SPECTRAL FILTERING
# ============================================================================

"""
Apply spectral filter to remove high-frequency noise
"""
function apply_spectral_filter!(fld::Fields{T}, dom::Domain, 
                               filter_strength::T) where T
    
    # Transform buoyancy to spectral space
    rfft!(dom, fld.b, fld.bhat)
    
    # Apply exponential filter
    apply_exponential_filter!(fld.bhat, dom, filter_strength)
    
    # Transform back to physical space
    irfft!(dom, fld.bhat, fld.b)
    
    return nothing
end

"""
Exponential spectral filter
"""
function apply_exponential_filter!(bhat::PencilArray{Complex{T}, 2}, 
                                  dom::Domain, strength::T) where T
    
    bhat_data = bhat.data
    
    # Get local wavenumber ranges
    local_ranges = local_range(bhat.pencil)
    
    # Filter parameters
    kx_max = π * dom.Nx / dom.Lx
    ky_max = π * dom.Ny / dom.Ly
    k_cutoff = 0.65 * min(kx_max, ky_max)  # Filter starts at 65% of Nyquist
    
    @inbounds for (j_local, j_global) in enumerate(local_ranges[2])
        ky = dom.ky[j_global]
        for (i_local, i_global) in enumerate(local_ranges[1])
            kx = dom.kx[i_global] 
            if i_global <= length(dom.kx) else 0
            
            k_mag = sqrt(kx^2 + ky^2)
            
            if k_mag > k_cutoff
                # Exponential filter
                filter_factor = exp(-strength * ((k_mag - k_cutoff) / (kx_max - k_cutoff))^2)
                bhat_data[i_local, j_local] *= filter_factor
            end
        end
    end
    
    return nothing
end

# ============================================================================
# MAIN TIME STEPPING INTERFACE
# ============================================================================

"""
Take one time step using specified scheme
"""
function timestep!(fld::Fields{T}, dom::Domain, 
                  params::TimeParams{T}, state::TimeState{T}) where T
    
    # Choose time integration scheme
    if params.scheme == AB2_LowStorage
        dt = timestep_ab2_ls!(fld, dom, params, state)
    elseif params.scheme == RK3
        dt = timestep_rk3!(fld, dom, params, state)
    elseif params.scheme == RK3_LowStorage
        dt = timestep_rk3_ls!(fld, dom, params, state)
    else
        error("Unknown time integration scheme: $(params.scheme)")
    end
    
    # Apply spectral filter periodically
    if params.filter_freq > 0 && state.step % params.filter_freq == 0
        apply_spectral_filter!(fld, dom, params.filter_strength)
    end
    
    return dt
end

"""
Time integration loop with diagnostics
"""
function integrate!(fld::Fields{T}, dom::Domain, 
                   params::TimeParams{T}, state::TimeState{T};
                   t_final::T, output_freq::Int=100,
                   verbose::Bool=true) where T
    
    n_steps = 0
    
    while state.t < t_final
        # Take time step
        dt_actual = timestep!(fld, dom, params, state)
        n_steps += 1
        
        # Ensure we don't overshoot final time
        if state.t + dt_actual > t_final
            # Adjust final step
            dt_remaining = t_final - state.t
            params = TimeParams{T}(dt_remaining; 
                                 scheme=params.scheme,
                                 filter_freq=params.filter_freq,
                                 filter_strength=params.filter_strength,
                                 adaptive_dt=false)
            timestep!(fld, dom, params, state)
            break
        end
        
        # Output diagnostics
        if verbose && n_steps % output_freq == 0
            cfl = compute_cfl_number(fld, dom, dt_actual)
            b_stats = enhanced_field_stats(fld)[:b]
            
            @printf("Step %6d: t = %8.4f, dt = %8.6f, CFL = %6.3f, |b|_max = %8.4e\n",
                   state.step, state.t, dt_actual, cfl, b_stats.max)
        end
    end
    
    if verbose
        @printf("Integration complete: %d steps, final time = %.6f\n", n_steps, state.t)
    end
    
    return state.t, n_steps
end

# ============================================================================
# ADVANCED DIAGNOSTICS AND UTILITIES
# ============================================================================

"""
Compute kinetic energy: KE = 0.5 * ∫(u² + v²) dA
"""
function compute_kinetic_energy(fld::Fields{T}, dom::Domain) where T
    # Ensure velocities are up to date
    compute_geostrophic_velocities!(fld)
    
    # Compute local kinetic energy
    u_data = fld.u.data
    v_data = fld.v.data
    
    ke_local = zero(T)
    @inbounds @simd for i in eachindex(u_data)
        ke_local += 0.5 * (u_data[i]^2 + v_data[i]^2)
    end
    
    # Global sum via MPI
    ke_global = MPI.Allreduce(ke_local, MPI.SUM, fld.domain.pc.comm)
    
    # Normalize by domain area
    total_points = dom.Nx * dom.Ny
    ke_mean = ke_global / total_points
    
    return ke_mean
end

"""
Compute enstrophy: ENS = 0.5 * ∫ω² dA where ω = ∇²ψ
"""
function compute_enstrophy(fld::Fields{T}, dom::Domain) where T
    # Compute vorticity ω = ∇²ψ using spectral method
    rfft!(dom, fld.φ, fld.φhat)
    laplacian_h!(dom, fld.φhat, fld.tmpc)  # ∇²ψ in spectral space
    irfft!(dom, fld.tmpc, fld.tmp)         # ω in physical space
    
    # Compute local enstrophy
    ω_data = fld.tmp.data
    ens_local = zero(T)
    @inbounds @simd for i in eachindex(ω_data)
        ens_local += 0.5 * ω_data[i]^2
    end
    
    # Global sum and normalize
    ens_global = MPI.Allreduce(ens_local, MPI.SUM, fld.domain.pc.comm)
    ens_mean = ens_global / (dom.Nx * dom.Ny)
    
    return ens_mean
end

"""
Compute total buoyancy (should be conserved)
"""
function compute_total_buoyancy(fld::Fields{T}, dom::Domain) where T
    b_data = fld.b.data
    b_sum_local = sum(b_data)
    b_sum_global = MPI.Allreduce(b_sum_local, MPI.SUM, fld.domain.pc.comm)
    
    # Normalize by domain area
    dx = dom.Lx / dom.Nx
    dy = dom.Ly / dom.Ny
    total_buoyancy = b_sum_global * dx * dy
    
    return total_buoyancy
end

"""
Compute maximum divergence (should be small for geostrophic flow)
"""
function compute_max_divergence(fld::Fields{T}, dom::Domain) where T
    # Ensure velocities are current
    compute_geostrophic_velocities!(fld)
    
    # Compute divergence: ∇·u = ∂u/∂x + ∂v/∂y
    rfft!(dom, fld.u, fld.tmpc)
    ddx!(dom, fld.tmpc, fld.tmpc)          # ∂u/∂x in spectral
    irfft!(dom, fld.tmpc, fld.tmp)         # ∂u/∂x in physical
    
    rfft!(dom, fld.v, fld.tmpc2)
    ddy!(dom, fld.tmpc2, fld.tmpc2)        # ∂v/∂y in spectral  
    irfft!(dom, fld.tmpc2, fld.tmp2)       # ∂v/∂y in physical
    
    # Total divergence
    div_data = fld.tmp.data     # ∂u/∂x
    dv_dy_data = fld.tmp2.data  # ∂v/∂y
    
    max_div_local = zero(T)
    @inbounds for i in eachindex(div_data)
        div_total = div_data[i] + dv_dy_data[i]
        max_div_local = max(max_div_local, abs(div_total))
    end
    
    # Global maximum
    max_div_global = MPI.Allreduce(max_div_local, MPI.MAX, fld.domain.pc.comm)
    
    return max_div_global
end

"""
Initialize Gaussian vortex for testing
"""
function initialize_gaussian_vortex!(fld::Fields{T}, dom::Domain;
                                    amplitude::T=T(1.0),
                                    center_x::T=T(0.5)*dom.Lx,
                                    center_y::T=T(0.5)*dom.Ly,
                                    width::T=T(0.1)*min(dom.Lx, dom.Ly)) where T
    
    # Get local grid points
    local_ranges = local_range(fld.b.pencil)
    b_data = fld.b.data
    
    # Create Gaussian buoyancy anomaly
    @inbounds for (j_local, j_global) in enumerate(local_ranges[2])
        y = (j_global - 1) * dom.Ly / dom.Ny
        for (i_local, i_global) in enumerate(local_ranges[1])
            x = (i_global - 1) * dom.Lx / dom.Nx
            
            # Distance from center
            dx = x - center_x
            dy = y - center_y
            
            # Apply periodic boundary conditions for distance
            dx = dx - dom.Lx * round(dx / dom.Lx)
            dy = dy - dom.Ly * round(dy / dom.Ly)
            
            r2 = dx^2 + dy^2
            b_data[i_local, j_local] = amplitude * exp(-r2 / (2 * width^2))
        end
    end
    
    # Ensure zero mean for periodic domain
    b_mean = compute_total_buoyancy(fld, dom) / (dom.Lx * dom.Ly)
    fld.b.data .-= b_mean
    
    return fld
end

"""
Initialize Taylor-Green vortex
"""
function initialize_taylor_green!(fld::Fields{T}, dom::Domain;
                                 amplitude::T=T(1.0)) where T
    
    local_ranges = local_range(fld.b.pencil)
    b_data = fld.b.data
    
    @inbounds for (j_local, j_global) in enumerate(local_ranges[2])
        y = (j_global - 1) * 2π / dom.Ny
        for (i_local, i_global) in enumerate(local_ranges[1])
            x = (i_global - 1) * 2π / dom.Nx
            
            # Taylor-Green pattern
            b_data[i_local, j_local] = amplitude * sin(x) * cos(y)
        end
    end
    
    return fld
end

"""
Apply random perturbations for turbulence studies
"""
function add_random_noise!(fld::Fields{T}, dom::Domain;
                          noise_amplitude::T=T(0.01)) where T
    
    b_data = fld.b.data
    
    # Add random noise (different on each process)
    @inbounds for i in eachindex(b_data)
        b_data[i] += noise_amplitude * (2*rand(T) - 1)
    end
    
    # Remove mean to maintain conservation
    b_mean = compute_total_buoyancy(fld, dom) / (dom.Lx * dom.Ly)
    fld.b.data .-= b_mean
    
    return fld
end

# ============================================================================
# OUTPUT AND VISUALIZATION UTILITIES
# ============================================================================

"""
Save fields to HDF5 file (placeholder - would need HDF5.jl)
"""
function save_fields(filename::String, fld::Fields{T}, time::T) where T
    # This would save to HDF5 format
    # h5open(filename, "w") do file
    #     file["time"] = time
    #     file["buoyancy"] = Array(fld.b)
    #     file["streamfunction"] = Array(fld.φ)
    #     file["u_velocity"] = Array(fld.u)
    #     file["v_velocity"] = Array(fld.v)
    # end
    
    println("Would save fields to $filename at time $time")
    return filename
end

"""
Time series diagnostics structure
"""
mutable struct DiagnosticTimeSeries{T}
    times::Vector{T}
    kinetic_energy::Vector{T}
    enstrophy::Vector{T}
    total_buoyancy::Vector{T}
    max_divergence::Vector{T}
    max_cfl::Vector{T}
    
    DiagnosticTimeSeries{T}() where T = new{T}(T[], T[], T[], T[], T[], T[])
end

"""
Update diagnostic time series
"""
function update_diagnostics!(diag::DiagnosticTimeSeries{T}, 
                            fld::Fields{T}, dom::Domain, 
                            state::TimeState{T}) where T
    
    push!(diag.times, state.t)
    push!(diag.kinetic_energy, compute_kinetic_energy(fld, dom))
    push!(diag.enstrophy, compute_enstrophy(fld, dom))
    push!(diag.total_buoyancy, compute_total_buoyancy(fld, dom))
    push!(diag.max_divergence, compute_max_divergence(fld, dom))
    push!(diag.max_cfl, state.cfl_max)
    
    return diag
end

"""
Print diagnostic summary
"""
function print_diagnostics(diag::DiagnosticTimeSeries{T}, step::Int) where T
    if length(diag.times) == 0
        return
    end
    
    idx = length(diag.times)
    @printf("Step %6d: t=%8.4f | KE=%8.4e | ENS=%8.4e | B_tot=%8.4e | div_max=%8.2e | CFL=%6.3f\n",
           step, diag.times[idx], diag.kinetic_energy[idx], diag.enstrophy[idx],
           diag.total_buoyancy[idx], diag.max_divergence[idx], diag.max_cfl[idx])
end

# ============================================================================
# ENHANCED INTEGRATION LOOP WITH FULL DIAGNOSTICS
# ============================================================================

"""
Advanced integration with comprehensive diagnostics and output
"""
function integrate_with_diagnostics!(fld::Fields{T}, dom::Domain, 
                                    params::TimeParams{T}, state::TimeState{T};
                                    t_final::T, 
                                    output_freq::Int=100,
                                    diag_freq::Int=10,
                                    save_freq::Int=1000,
                                    output_dir::String="output",
                                    verbose::Bool=true) where T
    
    # Initialize diagnostics
    diag = DiagnosticTimeSeries{T}()
    
    # Create output directory
    if MPI.Comm_rank(fld.domain.pc.comm) == 0
        mkpath(output_dir)
    end
    
    n_steps = 0
    save_counter = 0
    
    # Initial diagnostics
    update_diagnostics!(diag, fld, dom, state)
    
    if verbose && MPI.Comm_rank(fld.domain.pc.comm) == 0
        println("Starting semi-geostrophic integration...")
        println("Scheme: $(params.scheme), dt: $(params.dt), t_final: $t_final")
        print_diagnostics(diag, 0)
    end
    
    while state.t < t_final
        # Take time step
        dt_actual = timestep!(fld, dom, params, state)
        n_steps += 1
        
        # Update diagnostics
        if n_steps % diag_freq == 0
            update_diagnostics!(diag, fld, dom, state)
        end
        
        # Print progress
        if verbose && n_steps % output_freq == 0 && MPI.Comm_rank(fld.domain.pc.comm) == 0
            print_diagnostics(diag, n_steps)
        end
        
        # Save fields
        if n_steps % save_freq == 0
            save_counter += 1
            filename = joinpath(output_dir, "fields_$(save_counter).h5")
            save_fields(filename, fld, state.t)
        end
        
        # Check for final time
        if state.t >= t_final
            break
        end
    end
    
    # Final diagnostics and summary
    if verbose && MPI.Comm_rank(fld.domain.pc.comm) == 0
        update_diagnostics!(diag, fld, dom, state)
        println("\nIntegration completed!")
        @printf("Final time: %.6f, Total steps: %d\n", state.t, n_steps)
        @printf("Average dt: %.6f, Final CFL: %.4f\n", 
               (state.t - diag.times[1])/n_steps, diag.max_cfl[end])
        
        # Conservation check
        b_initial = diag.total_buoyancy[1]
        b_final = diag.total_buoyancy[end]
        b_drift = abs(b_final - b_initial) / abs(b_initial)
        @printf("Buoyancy conservation: %.2e relative drift\n", b_drift)
    end
    
    return diag, n_steps
end

# ============================================================================
# PROBLEM STRUCTURE AND STEP_UNTIL! INTERFACE
# ============================================================================

"""
Problem structure that encapsulates the entire semi-geostrophic system
"""
mutable struct SemiGeostrophicProblem{T<:AbstractFloat}
    # Core components
    fields::Fields{T}
    domain::Domain
    timestepper::TimeParams{T}
    clock::TimeState{T}
    
    # Optional components
    diagnostics::Union{DiagnosticTimeSeries{T}, Nothing}
    output_settings::NamedTuple
    
    function SemiGeostrophicProblem{T}(fields::Fields{T}, 
                                     domain::Domain,
                                     timestepper::TimeParams{T},
                                     clock::TimeState{T};
                                     diagnostics::Union{DiagnosticTimeSeries{T}, Nothing}=nothing,
                                     output_settings::NamedTuple=NamedTuple()) where T
        new{T}(fields, domain, timestepper, clock, diagnostics, output_settings)
    end
end

"""
Convenience constructor for SemiGeostrophicProblem
"""
function SemiGeostrophicProblem(domain::Domain{T}; 
                              scheme::TimeScheme=RK3,
                              dt::T=T(0.01),
                              initial_time::T=T(0.0),
                              adaptive_dt::Bool=false,
                              filter_freq::Int=10,
                              filter_strength::T=T(0.1),
                              enable_diagnostics::Bool=true,
                              output_dir::String="output") where T
    
    # Initialize components
    fields = allocate_enhanced_fields(domain)
    
    timestepper = TimeParams{T}(dt; 
                              scheme=scheme,
                              adaptive_dt=adaptive_dt,
                              filter_freq=filter_freq,
                              filter_strength=filter_strength)
    
    clock = TimeState{T, typeof(fields.b)}(initial_time, fields)
    
    # Optional diagnostics
    diagnostics = enable_diagnostics ? DiagnosticTimeSeries{T}() : nothing
    
    # Output settings
    output_settings = (
        dir = output_dir,
        save_freq = 100,
        diag_freq = 10,
        verbose = true
    )
    
    return SemiGeostrophicProblem{T}(fields, domain, timestepper, clock;
                                   diagnostics=diagnostics,
                                   output_settings=output_settings)
end

# ============================================================================
# STEP_UNTIL! IMPLEMENTATION
# ============================================================================

"""
Step forward until a specified stop time (convenience method)
"""
step_until!(prob::SemiGeostrophicProblem, stop_time) = step_until!(prob, prob.timestepper, stop_time)

"""
Step forward until a specified stop time with specified timestepper
"""
function step_until!(prob::SemiGeostrophicProblem{T}, 
                    timestepper::TimeParams{T}, 
                    stop_time::T) where T
    
    # Validation
    stop_time > prob.clock.t || error("stop_time ($stop_time) must be greater than current time ($(prob.clock.t))")
    
    # Store original time step for restoration
    original_dt = timestepper.dt
    original_adaptive = timestepper.adaptive_dt
    
    try
        # Extract current time step (may be adaptive)
        current_dt = prob.clock.dt_actual > 0 ? prob.clock.dt_actual : timestepper.dt
        
        # Calculate time interval and number of full steps
        time_interval = stop_time - prob.clock.t
        nsteps = floor(Int, time_interval / current_dt)
        
        # Take full steps
        if nsteps > 0
            stepforward!(prob, nsteps; preserve_timestepper=true)
        end
        
        # Calculate remaining time
        t_remaining = stop_time - prob.clock.t
        
        # Take final partial step if needed
        if t_remaining > eps(T)
            # Temporarily disable adaptive stepping for exact final time
            temp_timestepper = TimeParams{T}(t_remaining;
                                           scheme=timestepper.scheme,
                                           filter_freq=timestepper.filter_freq,
                                           filter_strength=timestepper.filter_strength,
                                           adaptive_dt=false)
            
            # Take the final step
            stepforward!(prob, 1; custom_timestepper=temp_timestepper)
        end
        
        # Ensure exact final time (handle floating point precision)
        prob.clock.t = stop_time
        
    finally
        # Always restore original timestepper parameters
        prob.timestepper = TimeParams{T}(original_dt;
                                       scheme=timestepper.scheme,
                                       filter_freq=timestepper.filter_freq,
                                       filter_strength=timestepper.filter_strength,
                                       adaptive_dt=original_adaptive,
                                       cfl_safety=timestepper.cfl_safety,
                                       max_dt=timestepper.max_dt,
                                       min_dt=timestepper.min_dt)
    end
    
    return nothing
end

"""
Step forward a specified number of steps
"""
function stepforward!(prob::SemiGeostrophicProblem{T}, nsteps::Int=1;
                     preserve_timestepper::Bool=false,
                     custom_timestepper::Union{TimeParams{T}, Nothing}=nothing) where T
    
    # Choose timestepper
    active_timestepper = custom_timestepper !== nothing ? custom_timestepper : prob.timestepper
    
    # Store original if preserving
    original_timestepper = preserve_timestepper ? deepcopy(prob.timestepper) : nothing
    
    try
        # Update problem timestepper if using custom
        if custom_timestepper !== nothing
            prob.timestepper = custom_timestepper
        end
        
        # Take the specified number of steps
        for step = 1:nsteps
            # Take one time step
            dt_actual = timestep!(prob.fields, prob.domain, active_timestepper, prob.clock)
            
            # Update diagnostics if enabled
            if prob.diagnostics !== nothing && 
               (prob.clock.step % prob.output_settings.diag_freq == 0)
                update_diagnostics!(prob.diagnostics, prob.fields, prob.domain, prob.clock)
            end
            
            # Output progress if requested
            if prob.output_settings.verbose && 
               (prob.clock.step % prob.output_settings.save_freq == 0) &&
               MPI.Comm_rank(prob.domain.pc.comm) == 0
                
                if prob.diagnostics !== nothing
                    print_diagnostics(prob.diagnostics, prob.clock.step)
                else
                    @printf("Step %6d: t = %8.4f, dt = %8.6f\n", 
                           prob.clock.step, prob.clock.t, dt_actual)
                end
            end
        end
        
    finally
        # Restore original timestepper if preserving
        if preserve_timestepper && original_timestepper !== nothing
            prob.timestepper = original_timestepper
        end
    end
    
    return nothing
end

"""
Step forward for a specified time duration
"""
function stepforward!(prob::SemiGeostrophicProblem{T}, duration::T) where T
    target_time = prob.clock.t + duration
    step_until!(prob, target_time)
    return nothing
end

# ============================================================================
# ENHANCED PROBLEM INTERFACE
# ============================================================================

"""
Set initial conditions for the problem
"""
function set_initial_conditions!(prob::SemiGeostrophicProblem{T}, 
                                init_func::Function, args...; kwargs...) where T
    init_func(prob.fields, prob.domain, args...; kwargs...)
    
    # Update initial diagnostics
    if prob.diagnostics !== nothing
        update_diagnostics!(prob.diagnostics, prob.fields, prob.domain, prob.clock)
    end
    
    return prob
end

"""
Run simulation for a specified duration with automatic output
"""
function run!(prob::SemiGeostrophicProblem{T}, duration::T;
             output_freq::Union{T, Nothing}=nothing,
             save_fields::Bool=false) where T
    
    final_time = prob.clock.t + duration
    
    if output_freq === nothing
        # Run without intermediate output
        step_until!(prob, final_time)
    else
        # Run with periodic output
        current_time = prob.clock.t
        while current_time < final_time
            next_output_time = min(current_time + output_freq, final_time)
            step_until!(prob, next_output_time)
            current_time = prob.clock.t
            
            # Save fields if requested
            if save_fields && MPI.Comm_rank(prob.domain.pc.comm) == 0
                filename = joinpath(prob.output_settings.dir, 
                                  "fields_t$(round(current_time, digits=4)).h5")
                save_fields(filename, prob.fields, current_time)
            end
        end
    end
    
    return prob
end

"""
Get current simulation state summary
"""
function simulation_summary(prob::SemiGeostrophicProblem{T}) where T
    summary = Dict{String, Any}()
    
    # Basic state
    summary["current_time"] = prob.clock.t
    summary["total_steps"] = prob.clock.step
    summary["time_scheme"] = prob.timestepper.scheme
    summary["current_dt"] = prob.clock.dt_actual > 0 ? prob.clock.dt_actual : prob.timestepper.dt
    
    # Physics
    if prob.diagnostics !== nothing && length(prob.diagnostics.times) > 0
        idx = length(prob.diagnostics.times)
        summary["kinetic_energy"] = prob.diagnostics.kinetic_energy[idx]
        summary["enstrophy"] = prob.diagnostics.enstrophy[idx]
        summary["total_buoyancy"] = prob.diagnostics.total_buoyancy[idx]
        summary["max_divergence"] = prob.diagnostics.max_divergence[idx]
        summary["max_cfl"] = prob.diagnostics.max_cfl[idx]
        
        # Conservation checks
        if length(prob.diagnostics.kinetic_energy) > 1
            ke_initial = prob.diagnostics.kinetic_energy[1]
            ke_current = prob.diagnostics.kinetic_energy[idx]
            summary["energy_conservation"] = abs(ke_current - ke_initial) / ke_initial
            
            b_initial = prob.diagnostics.total_buoyancy[1]
            b_current = prob.diagnostics.total_buoyancy[idx]
            summary["buoyancy_conservation"] = abs(b_current - b_initial) / abs(b_initial)
        end
    end
    
    return summary
end

"""
Pretty print simulation summary
"""
function Base.show(io::IO, prob::SemiGeostrophicProblem{T}) where T
    println(io, "SemiGeostrophicProblem{$T}:")
    println(io, "  Current time: $(prob.clock.t)")
    println(io, "  Total steps:  $(prob.clock.step)")
    println(io, "  Time scheme:  $(prob.timestepper.scheme)")
    println(io, "  Grid size:    $(prob.domain.Nx)×$(prob.domain.Ny)")
    println(io, "  Domain size:  $(prob.domain.Lx)×$(prob.domain.Ly)")
    println(io, "  Diagnostics:  $(prob.diagnostics !== nothing ? "enabled" : "disabled")")
    
    if prob.diagnostics !== nothing && length(prob.diagnostics.times) > 0
        idx = length(prob.diagnostics.times)
        println(io, "  Kinetic energy: $(prob.diagnostics.kinetic_energy[idx])")
        println(io, "  Max CFL:        $(prob.diagnostics.max_cfl[idx])")
    end
end


"""
 Example:
 ========

# Initialize
dom = Domain(512, 512, 2π, 2π, MPI.COMM_WORLD)
prob = SemiGeostrophicProblem(dom; scheme=RK3, dt=0.005, adaptive_dt=true)

# Initial conditions
set_initial_conditions!(prob, initialize_taylor_green!; amplitude=2.0)
set_initial_conditions!(prob, add_random_noise!; noise_amplitude=0.1)

# Phase 1: Careful spin-up
prob.timestepper.dt = 0.001
step_until!(prob, 2.0)
println("Spin-up complete: $(simulation_summary(prob))")

# Phase 2: Main evolution with output
prob.timestepper.dt = 0.01
run!(prob, 18.0; output_freq=2.0, save_fields=true)

# Phase 3: High-resolution final analysis
high_res = TimeParams{Float64}(0.002; scheme=RK3, filter_freq=5)
step_until!(prob, high_res, 25.0)

# Final summary
println("Simulation complete:")
println(prob)  # Pretty-printed summary

step_until!(prob, 10.0)                     # ✅ Matches interface
step_until!(prob, custom_timestepper, 15.0) # ✅ Extended version
"""
