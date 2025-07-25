# src/timestep.jl
# Time stepping schemes


"""
    apply_spectral_filter!(field, field_spec, filter, domain)

Apply spectral filtering to a single field.
"""
function apply_spectral_filter!(field, field_spec, filter, domain)
    # Transform to spectral space
    rfft!(domain, field, field_spec)
    
    # Apply dealiasing and filter
    dealias!(domain, field_spec)
    apply_filter!(field_spec, filter)
    
    # Transform back to physical space
    irfft!(domain, field_spec, field)
    
    return nothing
end

"""
    apply_filter!(field_spec, filter)

Apply spectral filter to a field in spectral space.
"""
function apply_filter!(field_spec, filter)
    field_local = field_spec.data
    @. field_local *= filter
    return nothing
end

"""
    solve_mongeampere!(domain::Domain, fields::Fields; kwargs...)

Solve the Monge-Ampère equation for the streamfunction.
This is a placeholder - the actual implementation would depend on 
the specific semigeostrophic formulation.
"""
function solve_mongeampere!(domain::Domain, fields::Fields; 
                           tol=1e-10, maxiter=100, verbose=false)
    # Placeholder for Monge-Ampère solver
    # This would typically involve:
    # 1. Setting up the nonlinear elliptic equation
    # 2. Iterative solution (Newton, Picard, etc.)
    # 3. Updating the streamfunction field φ
    
    # For now, set φ = 0 (this should be replaced with actual solver)
    zero_field!(fields.φ)
    
    return nothing
end

"""
    SG_velocities!(domain::Domain, fields::Fields, params)

Compute semigeostrophic velocities from the streamfunction and buoyancy.
"""# src/timesteppers.jl
# Time-stepping schemes for surface semigeostrophic equations

"""
    stepforward!(prob::Problem)

Step forward `prob` one time step.
"""
stepforward!(prob::Problem) =
  stepforward!(prob.fields, prob.clock, prob.timestepper, prob.equation, prob.params, prob.domain)

"""
    stepforward!(prob::Problem, nsteps::Int)

Step forward `prob` for `nsteps`.
"""
function stepforward!(prob::Problem, nsteps::Int)
  for _ in 1:nsteps
    stepforward!(prob)
  end
  
  return nothing
end

"""
    stepforward!(prob::Problem, diags, nsteps::Int)

Step forward `prob` for `nsteps`, incrementing `diags` along the way. `diags` may be a 
single `Diagnostic` or a `Vector` of `Diagnostic`s.
"""
function stepforward!(prob::Problem, diags, nsteps::Int)
  for _ in 1:nsteps
    stepforward!(prob)
    increment!(diags)
  end
  
  return nothing
end

const fullyexplicitsteppers = [
                            :ForwardEuler,
                            :RK4,
                            :RK3,
                            :RK2,
                            :AB3,
                            :LSRK54,
                            :FilteredForwardEuler,
                            :FilteredRK4,
                            :FilteredRK3,
                            :FilteredRK2,
                            :FilteredAB3,
                            :FilteredLSRK54
                        ]

isexplicit(stepper) = any(Symbol(stepper) .== fullyexplicitsteppers)

"""
    TimeStepper(stepper, equation, dt=nothing, dev=CPU(); kw...)

Instantiate the Time`stepper` for `equation` with timestep `dt` and
on the `dev`ice. The keyword arguments, `kw`, are passed to the
timestepper constructor.
"""
function TimeStepper(stepper, equation, dt=nothing, dev::Device=CPU(); kw...)
  fullsteppername = Symbol(stepper, :TimeStepper)

  # Create expression that instantiates the time-stepper, depending on whether 
  # timestepper is explicit or not.
  expr = isexplicit(stepper) ? Expr(:call, fullsteppername, equation, dev) :
                               Expr(:call, fullsteppername, equation, dt, dev)

  # Add keyword arguments    
  length(kw) > 0 && push!(expr.args, Tuple(Expr(:kw, p.first, p.second) for p in kw)...)

  return eval(expr)
end

# =============================================================================
# FORWARD EULER
# =============================================================================

"""
    struct ForwardEulerTimeStepper{T} <: AbstractTimeStepper{T}

A forward Euler timestepper for surface semigeostrophic equations:

```julia
b^{n+1} = b^n + dt * RHS(b^n, t^n)
```
"""
struct ForwardEulerTimeStepper{T} <: AbstractTimeStepper{T}
  N :: T # Explicit nonlinear terms for buoyancy equation
  ForwardEulerTimeStepper(N::T) where T = new{T}(0N)
end

"""
    ForwardEulerTimeStepper(equation::Equation, dev::Device=CPU())

Construct a forward Euler timestepper for `equation` on device `dev`.
"""
ForwardEulerTimeStepper(equation::Equation, dev::Device=CPU()) = 
  ForwardEulerTimeStepper(create_real_field(equation.domain, equation.T))

function stepforward!(fields, clock, ts::ForwardEulerTimeStepper, equation, params, domain)
  # Compute nonlinear terms for buoyancy equation
  equation.calcN!(ts.N, fields, clock.t, clock, params, domain)
  
  # Update buoyancy: b^{n+1} = b^n + dt * (L*b + N)
  b_local = fields.b.data
  N_local = ts.N.data
  @. b_local += clock.dt * N_local
  
  # Apply linear operator if needed (e.g., diffusion)
  if hasfield(typeof(equation), :L) && equation.L !== nothing
    apply_linear_operator!(fields.b, equation.L, domain, clock.dt)
  end
  
  # Solve for streamfunction and velocities after buoyancy update
  solve_mongeampere!(domain, fields; 
                    tol=params.MA_tol, 
                    maxiter=params.MA_maxiter,
                    verbose=false)
  SG_velocities!(domain, fields, params)
  
  clock.t += clock.dt
  clock.step += 1
  
  return nothing
end

"""
    struct FilteredForwardEulerTimeStepper{T,Tf} <: AbstractTimeStepper{T}

A forward Euler timestepper with spectral filtering for surface semigeostrophic equations.
"""
struct FilteredForwardEulerTimeStepper{T,Tf} <: AbstractTimeStepper{T}
       N :: T
  filter :: Tf
end

"""
    FilteredForwardEulerTimeStepper(equation, dev; filterkwargs...)

Construct a Forward Euler timestepper with spectral filtering for `equation` on device `dev`.
"""
function FilteredForwardEulerTimeStepper(equation::Equation, dev::Device=CPU(); filterkwargs...)
  filter = makefilter(equation; filterkwargs...)
  
  return FilteredForwardEulerTimeStepper(create_real_field(equation.domain, equation.T), filter)
end

function stepforward!(fields, clock, ts::FilteredForwardEulerTimeStepper, equation, params, domain)
  # Compute nonlinear terms for buoyancy equation
  equation.calcN!(ts.N, fields, clock.t, clock, params, domain)
  
  # Update buoyancy field
  b_local = fields.b.data
  N_local = ts.N.data
  @. b_local += clock.dt * N_local
  
  # Apply spectral filter to buoyancy
  apply_spectral_filter!(fields.b, fields.b̂, ts.filter, domain)
  
  # Solve for streamfunction and velocities
  solve_mongeampere!(domain, fields; 
                    tol=params.MA_tol, 
                    maxiter=params.MA_maxiter,
                    verbose=false)
  SG_velocities!(domain, fields, params)

  clock.t += clock.dt
  clock.step += 1

  return nothing
end

# =============================================================================
# RK4
# =============================================================================

"""
    struct RK4TimeStepper{T} <: AbstractTimeStepper{T}

A 4th-order Runge-Kutta timestepper for surface semigeostrophic equations.
"""
struct RK4TimeStepper{T} <: AbstractTimeStepper{T}
  b₁ :: T      # Intermediate buoyancy storage
  RHS₁ :: T    # RHS at stage 1
  RHS₂ :: T    # RHS at stage 2
  RHS₃ :: T    # RHS at stage 3
  RHS₄ :: T    # RHS at stage 4
end

"""
    RK4TimeStepper(equation::Equation, dev::Device=CPU())

Construct a 4th-order Runge-Kutta timestepper for `equation` on device `dev`.
"""
function RK4TimeStepper(equation::Equation, dev::Device=CPU())
  # Create temporary storage for buoyancy field only
  b₁ = create_real_field(equation.domain, equation.T)
  RHS₁ = create_real_field(equation.domain, equation.T)
  RHS₂ = create_real_field(equation.domain, equation.T)
  RHS₃ = create_real_field(equation.domain, equation.T)
  RHS₄ = create_real_field(equation.domain, equation.T)
  
  return RK4TimeStepper(b₁, RHS₁, RHS₂, RHS₃, RHS₄)
end

"""
    struct FilteredRK4TimeStepper{T,Tf} <: AbstractTimeStepper{T}

A 4th-order Runge-Kutta timestepper with spectral filtering.
"""
struct FilteredRK4TimeStepper{T,Tf} <: AbstractTimeStepper{T}
  b₁ :: T
  RHS₁ :: T
  RHS₂ :: T
  RHS₃ :: T
  RHS₄ :: T
  filter :: Tf
end

"""
    FilteredRK4TimeStepper(equation::Equation, dev::Device=CPU(); filterkwargs...)

Construct a 4th-order Runge-Kutta timestepper with spectral filtering.
"""
function FilteredRK4TimeStepper(equation::Equation, dev::Device=CPU(); filterkwargs...)
  ts = RK4TimeStepper(equation, dev)
  filter = makefilter(equation; filterkwargs...)

  return FilteredRK4TimeStepper(ts.b₁, ts.RHS₁, ts.RHS₂, ts.RHS₃, ts.RHS₄, filter)
end

function addlinearterm!(RHS, equation, fields, domain)
    if hasfield(typeof(equation), :L) && equation.L !== nothing
        # Add linear operator terms (e.g., diffusion)
        add_horizontal_diffusion!(RHS, domain, fields, equation.κ_h)
    end
    return nothing
end
  
function substep_buoyancy!(b_new, b_old, RHS, dt)
    # Update buoyancy: b_new = b_old + dt * RHS
    b_new_local = b_new.data
    b_old_local = b_old.data
    RHS_local = RHS.data
    @. b_new_local = b_old_local + dt * RHS_local
    
    return nothing
end

function RK4substeps!(fields, clock, ts, equation, params, domain, t, dt)
    # Store initial buoyancy
    copy_field!(ts.b₁, fields.b)  # b₀ = b^n
    
    # Substep 1
    equation.calcN!(ts.RHS₁, fields, t, clock, params, domain)
    addlinearterm!(ts.RHS₁, equation, fields, domain)

    # Substep 2: b₁ = b₀ + (dt/2) * k₁
    substep_buoyancy!(fields.b, ts.b₁, ts.RHS₁, dt/2)
    solve_mongeampere!(domain, fields; tol=params.MA_tol, maxiter=params.MA_maxiter, verbose=false)
    SG_velocities!(domain, fields, params)
    
    equation.calcN!(ts.RHS₂, fields, t+dt/2, clock, params, domain)
    addlinearterm!(ts.RHS₂, equation, fields, domain)

    # Substep 3: b₂ = b₀ + (dt/2) * k₂
    substep_buoyancy!(fields.b, ts.b₁, ts.RHS₂, dt/2)
    solve_mongeampere!(domain, fields; tol=params.MA_tol, maxiter=params.MA_maxiter, verbose=false)
    SG_velocities!(domain, fields, params)
    
    equation.calcN!(ts.RHS₃, fields, t+dt/2, clock, params, domain)
    addlinearterm!(ts.RHS₃, equation, fields, domain)

    # Substep 4: b₃ = b₀ + dt * k₃
    substep_buoyancy!(fields.b, ts.b₁, ts.RHS₃, dt)
    solve_mongeampere!(domain, fields; tol=params.MA_tol, maxiter=params.MA_maxiter, verbose=false)
    SG_velocities!(domain, fields, params)
    
    equation.calcN!(ts.RHS₄, fields, t+dt, clock, params, domain)
    addlinearterm!(ts.RHS₄, equation, fields, domain)

    return nothing
end
  
function RK4update!(fields, b₀, RHS₁, RHS₂, RHS₃, RHS₄, dt)
    # Final update: b^{n+1} = b₀ + dt/6 * (k₁ + 2k₂ + 2k₃ + k₄)
    b_local = fields.b.data
    b₀_local = b₀.data
    RHS₁_local = RHS₁.data
    RHS₂_local = RHS₂.data  
    RHS₃_local = RHS₃.data
    RHS₄_local = RHS₄.data
    
    @. b_local = b₀_local + dt/6 * (RHS₁_local + 2 * RHS₂_local + 2 * RHS₃_local + RHS₄_local)
    return nothing
end
  
function stepforward!(fields, clock, ts::RK4TimeStepper, equation, params, domain)
    RK4substeps!(fields, clock, ts, equation, params, domain, clock.t, clock.dt)
    RK4update!(fields, ts.b₁, ts.RHS₁, ts.RHS₂, ts.RHS₃, ts.RHS₄, clock.dt)
    
    # Final solve for streamfunction and velocities
    solve_mongeampere!(domain, fields; 
                      tol=params.MA_tol, 
                      maxiter=params.MA_maxiter, 
                      verbose=false)
    SG_velocities!(domain, fields, params)
  
    clock.t += clock.dt
    clock.step += 1
  
    return nothing
end

function stepforward!(fields, clock, ts::FilteredRK4TimeStepper, equation, params, domain)
    RK4substeps!(fields, clock, ts, equation, params, domain, clock.t, clock.dt)
    RK4update!(fields, ts.b₁, ts.RHS₁, ts.RHS₂, ts.RHS₃, ts.RHS₄, clock.dt)
    
    # Apply spectral filter to buoyancy
    apply_spectral_filter!(fields.b, fields.b̂, ts.filter, domain)
    
    # Final solve for streamfunction and velocities
    solve_mongeampere!(domain, fields; 
                      tol=params.MA_tol, 
                      maxiter=params.MA_maxiter, 
                      verbose=false)
    SG_velocities!(domain, fields, params)

    clock.t += clock.dt
    clock.step += 1
  
    return nothing
end

# =============================================================================
# RK3
# =============================================================================

"""
  struct RK3TimeStepper{T} <: AbstractTimeStepper{T}

A 3rd-order Runge-Kutta timestepper for surface semigeostrophic equations.
"""
struct RK3TimeStepper{T} <: AbstractTimeStepper{T}
  fields₁ :: T
  RHS₁ :: T
  RHS₂ :: T
  RHS₃ :: T
end

"""
    RK3TimeStepper(equation::Equation, dev::Device=CPU())

Construct a 3rd-order Runge-Kutta timestepper for `equation` on device `dev`.
"""
function RK3TimeStepper(equation::Equation, dev::Device=CPU())
  fields₁ = allocate_fields(equation.domain)
  RHS₁ = create_real_field(equation.domain, equation.T)
  RHS₂ = create_real_field(equation.domain, equation.T)
  RHS₃ = create_real_field(equation.domain, equation.T)
  
  return RK3TimeStepper(fields₁, RHS₁, RHS₂, RHS₃)
end

"""
    struct FilteredRK3TimeStepper{T,Tf} <: AbstractTimeStepper{T}

A 3rd-order Runge-Kutta timestepper with spectral filtering.
"""
struct FilteredRK3TimeStepper{T,Tf} <: AbstractTimeStepper{T}
  fields₁ :: T
  RHS₁ :: T
  RHS₂ :: T
  RHS₃ :: T
  filter :: Tf
end

"""
    FilteredRK3TimeStepper(equation::Equation, dev::Device=CPU(); filterkwargs...)

Construct a 3rd-order Runge-Kutta timestepper with spectral filtering.
"""
function FilteredRK3TimeStepper(equation::Equation, dev::Device=CPU(); filterkwargs...)
  ts = RK3TimeStepper(equation, dev)
  filter = makefilter(equation; filterkwargs...)

  return FilteredRK3TimeStepper(ts.fields₁, ts.RHS₁, ts.RHS₂, ts.RHS₃, filter)
end

function RK3substeps!(fields, clock, ts, equation, params, domain, t, dt)
  # Substep 1
  equation.calcN!(ts.RHS₁, fields, t, clock, params, domain)
  add_linear_terms!(ts.RHS₁, fields, equation, domain)

  # Substep 2
  substep_fields!(ts.fields₁, fields, ts.RHS₁, dt/2)
  equation.calcN!(ts.RHS₂, ts.fields₁, t+dt/2, clock, params, domain)
  add_linear_terms!(ts.RHS₂, ts.fields₁, equation, domain)

  # Substep 3: note the 2*RHS₂ - RHS₁ combination
  # Create temporary RHS for this substep
  RHS_temp_local = ts.RHS₁.data  # Reuse RHS₁ storage
  RHS₁_local = ts.RHS₁.data
  RHS₂_local = ts.RHS₂.data
  @. RHS_temp_local = 2.0 * RHS₂_local - RHS₁_local
  
  substep_fields!(ts.fields₁, fields, ts.RHS₁, dt)  # Using modified RHS₁
  equation.calcN!(ts.RHS₃, ts.fields₁, t+dt, clock, params, domain)
  add_linear_terms!(ts.RHS₃, ts.fields₁, equation, domain)

  return nothing
end

function RK3update!(fields, RHS₁, RHS₂, RHS₃, dt)
  # Update: b^{n+1} = b^n + dt/6 * (k₁ + 4k₂ + k₃)
  b_local = fields.b.data
  RHS₁_local = RHS₁.data
  RHS₂_local = RHS₂.data
  RHS₃_local = RHS₃.data
  
  @. b_local += dt/6 * (RHS₁_local + 4 * RHS₂_local + RHS₃_local)
  return nothing
end

function stepforward!(fields, clock, ts::RK3TimeStepper, equation, params, domain)
  RK3substeps!(fields, clock, ts, equation, params, domain, clock.t, clock.dt)
  RK3update!(fields, ts.RHS₁, ts.RHS₂, ts.RHS₃, clock.dt)

  clock.t += clock.dt
  clock.step += 1

  return nothing
end

function stepforward!(fields, clock, ts::FilteredRK3TimeStepper, equation, params, domain)
  RK3substeps!(fields, clock, ts, equation, params, domain, clock.t, clock.dt)
  RK3update!(fields, ts.RHS₁, ts.RHS₂, ts.RHS₃, clock.dt)
  
  # Apply spectral filters
  apply_spectral_filters!(fields, ts.filter, domain)

  clock.t += clock.dt
  clock.step += 1

  return nothing
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    apply_spectral_filters!(fields, filter, domain)

Apply spectral filtering to prognostic fields.
"""
function apply_spectral_filters!(fields, filter, domain)
    # Filter buoyancy
    rfft!(domain, fields.b, fields.b̂)
    apply_filter!(fields.b̂, filter)
    irfft!(domain, fields.b̂, fields.b)
    
    # Filter streamfunction if present
    if hasfield(typeof(fields), :φ)
        rfft!(domain, fields.φ, fields.φ̂)
        apply_filter!(fields.φ̂, filter)
        irfft!(domain, fields.φ̂, fields.φ)
    end
    
    return nothing
end

"""
    apply_filter!(field_spec, filter)

Apply spectral filter to a field in spectral space.
"""
function apply_filter!(field_spec, filter)
    field_local = field_spec.data
    @. field_local *= filter
    return nothing
end

"""
    apply_linear_operator!(field, L, domain, dt)

Apply linear operator (e.g., diffusion) to a field.
"""
function apply_linear_operator!(field, L, domain, dt)
    # Transform to spectral space
    field_spec = create_spectral_field(domain)
    rfft!(domain, field, field_spec)
    
    # Apply linear operator in spectral space
    field_spec_local = field_spec.data
    @. field_spec_local *= exp(dt * L)  # Assuming L contains eigenvalues
    
    # Transform back
    irfft!(domain, field_spec, field)
    
    return nothing
end

"""
    step_until!(prob, stop_time)

Step forward `prob` until `stop_time`.

!!! warn "Fully-explicit timestepping schemes are required"
    We cannot use `step_until!` with implicit time steppers.

See also: [`stepforward!`](@ref)
"""
step_until!(prob, stop_time) = step_until!(prob, prob.timestepper, stop_time)

function step_until!(prob, timestepper, stop_time)
  # Throw an error if stop_time is not greater than the current problem time
  stop_time > prob.clock.t || error("stop_time must be greater than prob.clock.t")

  # Extract current time step
  dt = prob.clock.dt

  # Step forward until just before stop_time
  time_interval = stop_time - prob.clock.t
  nsteps = floor(Int, time_interval / dt)
  stepforward!(prob, nsteps)

  # Take one final small step so that prob.clock.t = stop_time
  t_remaining = stop_time - prob.clock.t
  prob.clock.dt = t_remaining
  stepforward!(prob)

  # Restore previous time-step
  prob.clock.dt = dt

  return nothing
end