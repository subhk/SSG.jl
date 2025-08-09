# src/timestep.jl
# Time integration for surface semi-geostrophic equations
# Supports low-storage 2nd order Adams-Bashforth and 3rd order Runge-Kutta with spectral filtering 

using PencilArrays: range_local

"""
Time integration scheme selector
"""
@enum TimeScheme begin
    AB2_LowStorage  # 2nd order Adams-Bashforth (low storage)
    RK3             # 3rd order Runge-Kutta
    RK3_LowStorage  # 3rd order Runge-Kutta (low storage variant)
end

"""
Time integration parameters
"""
struct TimeParams{T<:AbstractFloat}
    dt::T                    # time step
    scheme::TimeScheme       # integration scheme
    filter_freq::Int          # frequency of spectral filtering (every N steps)
    filter_strength::T        # spectral filter strength (0 = no filter, 1 = strong)
    cfl_safety::T             # CFL safety factor
    max_dt::T                # maximum allowed time step
    min_dt::T                # minimum allowed time step
    adaptive_dt::Bool        # adaptive time stepping
    
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
mutable struct TimeState{T, PA2D}
    t::T                     # current time
    step::Int                # current step number
    
    # For Adams-Bashforth (stores previous tendency)
    db_dt_old::PA2D           # previous time derivative of buoyancy
    
    # For Runge-Kutta (intermediate stages)
    b_stage::PA2D             # intermediate buoyancy for RK stages
    k1::PA2D                  # RK stage derivatives
    k2::PA2D
    k3::PA2D
    
    # Diagnostic
    dt_actual::T            # actual dt used (for adaptive stepping)
    cfl_max::T              # maximum CFL number
    
    function TimeState{T, PA2D}(initial_time::T, fields::Fields{T}) where {T, PA2D}
        t = initial_time
        step = 0
        
        # Allocate storage for time integration
        db_dt_old = similar(fields.bₛ)
        b_stage   = similar(fields.bₛ)

        k1 = similar(fields.bₛ)
        k2 = similar(fields.bₛ)
        k3 = similar(fields.bₛ)
        
        # Initialize
        zero_field!(db_dt_old)
        zero_field!(b_stage)
        zero_field!(k1)
        zero_field!(k2)
        zero_field!(k3)
        
        dt_actual = T(0)
        cfl_max = T(0)
        
        new{T, PA2D}(t, step, db_dt_old, b_stage, k1, k2, k3, dt_actual, cfl_max)
    end
end

"""
Compute CFL number for current state
"""
function compute_cfl_number(fields::Fields{T}, domain::Domain, dt::T) where T
    # Compute geostrophic velocities
    compute_geostrophic_velocities!(fields, domain)
    
    # Get velocity magnitudes
    u_data = fields.u.data
    v_data = fields.v.data
    
    # Compute maximum velocity
    u_max_local = maximum(abs, u_data)
    v_max_local = maximum(abs, v_data)
    
    # MPI reduction to get global maximum
    u_max = MPI.Allreduce(u_max_local, MPI.MAX, fields.u.pencil.comm)
    v_max = MPI.Allreduce(v_max_local, MPI.MAX, fields.v.pencil.comm)
    
    # Grid spacing
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    
    # CFL number
    cfl_x = u_max * dt / dx
    cfl_y = v_max * dt / dy
    cfl = max(cfl_x, cfl_y)
    
    return cfl
end

"""
Adaptive time step calculation
"""
function adaptive_timestep(fields::Fields{T}, domain::Domain, 
                          params::TimeParams{T}, state::TimeState{T}) where T
    
    if !params.adaptive_dt
        return params.dt
    end
    
    # Compute current CFL number
    cfl = compute_cfl_number(fields, domain, params.dt)
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

"""
Low-storage 2nd order Adams-Bashforth time step
"""
function timestep_ab2_ls!(fields::Fields{T}, domain::Domain, 
                         params::TimeParams{T}, state::TimeState{T}) where T
    
    dt = adaptive_timestep(fields, domain, params, state)
    state.dt_actual = dt
    
    # Compute current tendency
    compute_tendency!(fields.tmp2, fields, domain, params)  # tmp2 = db/dt at current time
    
    if state.step == 0
        # First step: Forward Euler
        b_data = fields.bₛ.data
        tmp_data = fields.tmp2.data
        @inbounds @simd for i in eachindex(b_data)
            b_data[i] += dt * tmp_data[i]
        end
        
        # Store tendency for next step
        copy_field!(state.db_dt_old, fields.tmp2)
    else
        # Adams-Bashforth step: bₛ^{n+1} = bₛ^n + dt(3/2 * f^n - 1/2 * f^{n-1})
        b_data   = fields.bₛ.data
        tmp_data = fields.tmp2.data           # current tendency
        old_data = state.db_dt_old.data     # previous tendency
        
        @inbounds @simd for i in eachindex(b_data)
            b_data[i] += dt * (1.5 * tmp_data[i] - 0.5 * old_data[i])
        end
        
        # Update stored tendency (low-storage: reuse arrays)
        copy_field!(state.db_dt_old, fields.tmp2)
    end
    
    # Update time and step
    state.t += dt
    state.step += 1
    
    return dt
end

"""
3rd order Runge-Kutta time step
"""
function timestep_rk3!(fields::Fields{T}, domain::Domain, 
                      params::TimeParams{T}, state::TimeState{T}) where T
    
    dt = adaptive_timestep(fields, domain, params, state)
    state.dt_actual = dt
    
    # Store initial state
    copy_field!(state.b_stage, fields.bₛ)
    
    # Classical 3rd order Runge-Kutta (RK3) coefficients
    # This is the standard Heun's third-order method
    
    # Stage 1: k1 = f(t_n, y_n)
    compute_tendency!(state.k1, fields, domain, params)
    
    # Update to intermediate state: y1 = y_n + dt/3 * k1
    b_data = fields.bₛ.data
    b_stage_data = state.b_stage.data
    k1_data = state.k1.data
    
    @inbounds @simd for i in eachindex(b_data)
        b_data[i] = b_stage_data[i] + (dt/3.0) * k1_data[i]
    end
    
    # Stage 2: k2 = f(t_n + dt/3, y1)
    compute_tendency!(state.k2, fields, domain, params)
    
    # Update to second intermediate state: y2 = y_n + 2*dt/3 * k2
    k2_data = state.k2.data
    @inbounds @simd for i in eachindex(b_data)
        b_data[i] = b_stage_data[i] + (2.0*dt/3.0) * k2_data[i]
    end
    
    # Stage 3: k3 = f(t_n + 2*dt/3, y2)
    compute_tendency!(state.k3, fields, domain, params)
    
    # Final update: y_{n+1} = y_n + dt/4 * (k1 + 3*k3)
    k3_data = state.k3.data
    @inbounds @simd for i in eachindex(b_data)
        b_data[i] = b_stage_data[i] + (dt/4.0) * (k1_data[i] + 3.0*k3_data[i])
    end
    
    # Update time and step
    state.t += dt
    state.step += 1
    
    return dt
end

"""
Apply spectral filter to remove high-frequency noise from surface buoyancy field
"""
function apply_spectral_filter!(fields::Fields{T}, domain::Domain, 
                               filter_strength::T) where T
    
    # Transform surface buoyancy to spectral space (2D)
    rfft_2d!(domain, fields.bₛ, fields.bhat)
    
    # Apply exponential filter to spectral field
    apply_exponential_filter!(fields.bhat, domain, filter_strength)
    
    # Transform back to physical space (2D)
    irfft_2d!(domain, fields.bhat, fields.bₛ)
    
    return nothing
end

"""
Exponential spectral filter for 2D spectral fields
"""
function apply_exponential_filter!(bhat::PencilArray{Complex{T}, 2}, 
                                  domain::Domain, strength::T) where T
    
    bhat_data = bhat.data
    
    # Get local wavenumber ranges for 2D pencil
    range_locals = range_local(bhat.pencil)
    
    # Filter parameters
    kx_max = π * domain.Nx / domain.Lx
    ky_max = π * domain.Ny / domain.Ly
    k_cutoff = 0.65 * min(kx_max, ky_max)  # Filter starts at 65% of Nyquist
    k_range = kx_max - k_cutoff
    
    # Avoid division by zero
    if k_range < 1e-14
        return nothing
    end
    
    @inbounds for (j_local, j_global) in enumerate(range_locals[2])
        if j_global <= length(domain.ky)
            ky = domain.ky[j_global]
            for (i_local, i_global) in enumerate(range_locals[1])
                if i_global <= length(domain.kx)
                    kx = domain.kx[i_global]
                    
                    k_mag = sqrt(kx^2 + ky^2)
                    
                    if k_mag > k_cutoff
                        # Exponential filter with proper normalization
                        filter_param = (k_mag - k_cutoff) / k_range
                        filter_factor = exp(-strength * filter_param^2)
                        bhat_data[i_local, j_local] *= filter_factor
                    end
                end
            end
        end
    end
    
    return nothing
end

"""
Take one time step using specified scheme
"""
function timestep!(fields::Fields{T}, domain::Domain, 
                  params::TimeParams{T}, state::TimeState{T}) where T
    
    # Choose time integration scheme
    if params.scheme == AB2_LowStorage
        dt = timestep_ab2_ls!(fields, domain, params, state)
    elseif params.scheme == RK3
        dt = timestep_rk3!(fields, domain, params, state)
    else
        error("Time integration scheme $(params.scheme) not implemented yet")
    end
    
    # Apply spectral filter periodically
    if params.filter_freq > 0 && state.step % params.filter_freq == 0
        apply_spectral_filter!(fields, domain, params.filter_strength)
    end
    
    return dt
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
                            fields::Fields{T}, domain::Domain, 
                            state::TimeState{T}) where T
    
    push!(diag.times, state.t)
    
    # Compute kinetic energy
    compute_geostrophic_velocities!(fields, domain)
    ke = 0.5 * (parsevalsum2(fields.u, domain) + parsevalsum2(fields.v, domain))
    push!(diag.kinetic_energy, ke)
    
    # Compute enstrophy (simplified)
    push!(diag.enstrophy, 0.0)  # Placeholder
    
    # Compute total buoyancy  
    b_sum = parsevalsum(fields.bₛ, domain)
    push!(diag.total_buoyancy, b_sum)
    
    # Max divergence (placeholder)
    push!(diag.max_divergence, 0.0)
    
    # CFL
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
    @printf("Step %6d: t=%8.4f | KE=%8.4e | B_tot=%8.4e | CFL=%6.3f\n",
           step, diag.times[idx], diag.kinetic_energy[idx],
           diag.total_buoyancy[idx], diag.max_cfl[idx])
end

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
end

"""
Convenience constructor for SemiGeostrophicProblem
"""
function SemiGeostrophicProblem(domain::Domain; 
                              scheme::TimeScheme=RK3,
                              dt::Real=0.01,
                              initial_time::Real=0.0,
                              adaptive_dt::Bool=false,
                              filter_freq::Int=10,
                              filter_strength::Real=0.1,
                              enable_diagnostics::Bool=true,
                              output_dir::String="output")
    
    T = FT  # Use the global FT constant
    
    # Initialize components
    fields = allocate_fields(domain)
    
    timestepper = TimeParams{T}(T(dt); 
                              scheme=scheme,
                              adaptive_dt=adaptive_dt,
                              filter_freq=filter_freq,
                              filter_strength=T(filter_strength))
    
    clock = TimeState{T, typeof(fields.bₛ)}(T(initial_time), fields)
    
    # Optional diagnostics
    diagnostics = enable_diagnostics ? DiagnosticTimeSeries{T}() : nothing
    
    # Output settings
    output_settings = (
        dir = output_dir,
        save_freq = 100,
        diag_freq = 10,
        verbose = true
    )
    
    return SemiGeostrophicProblem{T}(fields, domain, timestepper, 
                                clock, diagnostics, output_settings)
end

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
    
    # Simple implementation: step until we reach the target time
    while prob.clock.t < stop_time
        # Calculate remaining time
        t_remaining = stop_time - prob.clock.t
        
        # Adjust time step if needed
        dt_use = min(prob.timestepper.dt, t_remaining)
        
        # Create temporary timestepper with adjusted dt
        temp_params = TimeParams{T}(dt_use;
                                   scheme=timestepper.scheme,
                                   filter_freq=timestepper.filter_freq,
                                   filter_strength=timestepper.filter_strength,
                                   adaptive_dt=false)
        
        # Take time step
        timestep!(prob.fields, prob.domain, temp_params, prob.clock)
        
        # Update diagnostics if enabled
        if prob.diagnostics !== nothing
            update_diagnostics!(prob.diagnostics, prob.fields, prob.domain, prob.clock)
        end
    end
    
    # Ensure exact final time
    prob.clock.t = stop_time
    
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
           MPI.Comm_rank(prob.domain.pc3d.comm) == 0
            
            if prob.diagnostics !== nothing
                print_diagnostics(prob.diagnostics, prob.clock.step)
            else
                @printf("Step %6d: t = %8.4f, dt = %8.6f\n", 
                       prob.clock.step, prob.clock.t, dt_actual)
            end
        end
    end
    
    return nothing
end