# API Reference

##  Overview

This document provides a comprehensive reference for all **public functions** and **types** in SSG.jl. Functions are organized by module and functionality, with complete parameter descriptions, return values, and usage examples.

##  Table of Contents

1. [Domain Creation](#domain-creation)
2. [Field Operations](#field-operations)
3. [Transform Operations](#transform-operations)
4. [Time Integration](#time-integration)
5. [Solver Operations](#solver-operations)
6. [Initial Conditions](#initial-conditions)
7. [Diagnostics](#diagnostics)
8. [Utility Functions](#utility-functions)
9. [I/O Operations](#io-operations)

---

## Domain Creation

### `make_domain`

```julia
make_domain(Nx, Ny, Nz; kwargs...) -> Domain
```

Creates a 3D computational domain with spectral methods support.

**Arguments:**
- `Nx::Int`: Number of grid points in x-direction
- `Ny::Int`: Number of grid points in y-direction  
- `Nz::Int`: Number of grid points in z-direction

**Keyword Arguments:**
- `Lx::Real = 2π`: Domain size in x-direction
- `Ly::Real = 2π`: Domain size in y-direction
- `Lz::Real = 1.0`: Domain depth (z ∈ [-Lz, 0])
- `z_boundary::Symbol = :dirichlet`: Vertical boundary condition
- `z_grid::Symbol = :uniform`: Vertical grid type
- `stretch_params::NamedTuple = nothing`: Grid stretching parameters
- `comm::MPI.Comm = MPI.COMM_WORLD`: MPI communicator

**Valid `z_boundary` options:**
- `:dirichlet`: Fixed values at boundaries
- `:neumann`: Fixed derivatives at boundaries  
- `:periodic`: Periodic boundary conditions
- `:free_slip`: Free-slip boundary conditions

**Valid `z_grid` options:**
- `:uniform`: Equally spaced grid points
- `:stretched`: Non-uniform spacing with stretching
- `:custom`: User-specified coordinates

**Stretch parameters for `z_grid = :stretched`:**
- `(type=:tanh, β=2.0, surface_concentration=true)`: Tanh stretching
- `(type=:sinh, β=2.0)`: Sinh stretching  
- `(type=:exponential, β=2.0)`: Exponential clustering
- `(type=:power, α=1.5)`: Power law stretching

**Returns:** `Domain{T}` structure containing grid information and FFT plans.

**Examples:**
```julia
# Basic uniform domain
domain = make_domain(64, 64, 16; Lx=4π, Ly=4π, Lz=2.0)

# Ocean-scale domain with surface stretching
domain = make_domain(256, 256, 64; 
                    Lx=100e3, Ly=100e3, Lz=1000.0,  # 100km×100km×1km
                    z_grid=:stretched,
                    stretch_params=(type=:tanh, β=2.5, surface_concentration=true))

# Custom vertical grid
custom_z = [-1.0, -0.7, -0.4, -0.2, -0.05, 0.0]
domain = make_domain(128, 128, 6;
                    z_grid=:custom,
                    stretch_params=(z_coords=custom_z,))
```

---

### `gridpoints`

```julia  
gridpoints(domain::Domain, direction::Symbol) -> Vector
```

Extract coordinate vector for specified direction.

**Arguments:**
- `domain::Domain`: Domain structure
- `direction::Symbol`: Coordinate direction (`:x`, `:y`, `:z`)

**Returns:** Vector of coordinate values.

**Example:**
```julia
x_coords = gridpoints(domain, :x)
z_coords = gridpoints(domain, :z)
```

---

### `gridpoints_2d`

```julia
gridpoints_2d(domain::Domain) -> (Vector, Vector)
```

Extract 2D surface coordinate vectors.

**Returns:** Tuple `(x, y)` of coordinate vectors.

**Example:**
```julia
x, y = gridpoints_2d(domain)
```

---

## Field Operations

### `allocate_fields`

```julia
allocate_fields(domain::Domain{T}) -> Fields{T}
```

Allocates all field arrays for the given domain.

**Arguments:**
- `domain::Domain{T}`: Domain structure

**Returns:** `Fields{T}` structure with all arrays allocated.

**Example:**
```julia
fields = allocate_fields(domain)
```

---

### `zero_fields!`

```julia
zero_fields!(fields::Fields) -> Nothing
```

Sets all field arrays to zero.

**Arguments:**
- `fields::Fields`: Field structure

**Example:**
```julia
zero_fields!(fields)
```

---

### `copy_field!`

```julia
copy_field!(dest::PencilArray, src::PencilArray) -> Nothing
```

Copies data from source to destination field.

**Arguments:**
- `dest::PencilArray`: Destination array
- `src::PencilArray`: Source array

**Requirements:** Arrays must have same size and distribution.

**Example:**
```julia
copy_field!(fields.tmp, fields.bₛ)
```

---

### `zero_field!`

```julia
zero_field!(field::PencilArray) -> Nothing  
```

Sets field array to zero.

**Arguments:**
- `field::PencilArray`: Field to zero

**Example:**
```julia
zero_field!(fields.tmp)
```

---

### `@ensuresamegrid`

```julia
@ensuresamegrid(field1, field2)
```

Macro to verify that two fields have compatible grid dimensions.

**Arguments:**
- `field1, field2`: PencilArrays to compare

**Throws:** `ArgumentError` if grids don't match.

**Example:**
```julia
@ensuresamegrid(fields.bₛ, fields.φₛ)  # ✓ Both 2D surface fields
@ensuresamegrid(fields.bₛ, fields.φ)   # ✗ 2D vs 3D mismatch
```

---

## Transform Operations

### FFT Operations

#### `rfft!`

```julia
rfft!(domain::Domain, realfield::PencilArray, specfield::PencilArray) -> Nothing
```

Forward real FFT: real space → spectral space (horizontal directions).

**Arguments:**
- `domain::Domain`: Domain structure with FFT plans
- `realfield::PencilArray{T, N}`: Input real field
- `specfield::PencilArray{Complex{T}, N}`: Output spectral field

**Example:**
```julia
rfft!(domain, fields.bₛ, fields.bshat)  # 2D surface FFT
rfft!(domain, fields.φ, fields.φhat)    # 3D volume FFT
```

---

#### `irfft!`

```julia
irfft!(domain::Domain, specfield::PencilArray, realfield::PencilArray) -> Nothing
```

Inverse real FFT: spectral space → real space (horizontal directions).

**Arguments:**
- `domain::Domain`: Domain structure with FFT plans
- `specfield::PencilArray{Complex{T}, N}`: Input spectral field
- `realfield::PencilArray{T, N}`: Output real field

**Example:**
```julia
irfft!(domain, fields.bshat, fields.bₛ)  # 2D surface inverse FFT
irfft!(domain, fields.φhat, fields.φ)    # 3D volume inverse FFT
```

---

#### `rfft_2d!`, `irfft_2d!`

```julia
rfft_2d!(domain::Domain, realfield_2d::PencilArray{T,2}, specfield_2d::PencilArray{Complex{T},2}) -> Nothing
irfft_2d!(domain::Domain, specfield_2d::PencilArray{Complex{T},2}, realfield_2d::PencilArray{T,2}) -> Nothing
```

Specialized 2D FFT operations for surface fields.

**Example:**
```julia
rfft_2d!(domain, fields.bₛ, fields.bshat)
irfft_2d!(domain, fields.bshat, fields.bₛ)
```

---

### Spectral Derivatives

#### `ddx!`, `ddy!`

```julia
ddx!(domain::Domain, Â::PencilArray, out̂::PencilArray) -> Nothing
ddy!(domain::Domain, Â::PencilArray, out̂::PencilArray) -> Nothing
```

Spectral derivatives ∂/∂x and ∂/∂y: multiply by ikₓ, ikᵧ.

**Arguments:**
- `domain::Domain`: Domain with wavenumber arrays
- `Â::PencilArray{Complex{T}, N}`: Input spectral field
- `out̂::PencilArray{Complex{T}, N}`: Output derivative field

**Example:**
```julia
# Compute ∂b/∂x in spectral space
rfft_2d!(domain, fields.bₛ, fields.bshat)
ddx_2d!(domain, fields.bshat, fields.tmpc_2d)  # tmpc_2d = ∂̂b/∂x
irfft_2d!(domain, fields.tmpc_2d, fields.tmp)  # tmp = ∂b/∂x
```

---

#### `ddx_2d!`, `ddy_2d!`

```julia
ddx_2d!(domain::Domain, Â::PencilArray{Complex{T},2}, out̂::PencilArray{Complex{T},2}) -> Nothing
ddy_2d!(domain::Domain, Â::PencilArray{Complex{T},2}, out̂::PencilArray{Complex{T},2}) -> Nothing
```

2D spectral derivatives for surface fields.

---

#### `laplacian_h!`

```julia
laplacian_h!(domain::Domain, Â::PencilArray, out̂::PencilArray) -> Nothing
```

Horizontal Laplacian: multiply by -(kₓ² + kᵧ²).

**Example:**
```julia
rfft!(domain, fields.φ, fields.φhat)
laplacian_h!(domain, fields.φhat, fields.tmpc_3d)  # ∇²ₕφ̂
irfft!(domain, fields.tmpc_3d, fields.tmp)         # ∇²ₕφ
```

---

#### Vertical Derivatives

#### `ddz!`

```julia
ddz!(domain::Domain, field::PencilArray{T,3}, out::PencilArray{T,3}) -> Nothing
```

Vertical derivative ∂/∂z using finite differences.

**Arguments:**
- `domain::Domain`: Domain with vertical grid spacing
- `field::PencilArray{T,3}`: Input 3D field
- `out::PencilArray{T,3}`: Output derivative field

---

#### `d2dz2!`

```julia
d2dz2!(domain::Domain, field::PencilArray{T,3}, out::PencilArray{T,3}) -> Nothing
```

Second vertical derivative ∂²/∂z² using finite differences.

---

### Dealiasing

#### `dealias!`

```julia
dealias!(domain::Domain, Â::PencilArray) -> Nothing
```

Apply two-thirds dealiasing rule to spectral field.

**Arguments:**
- `domain::Domain`: Domain with dealiasing mask
- `Â::PencilArray{Complex{T}, N}`: Spectral field to dealias

**Example:**
```julia
rfft_2d!(domain, fields.bₛ, fields.bshat)
dealias!(domain, fields.bshat)  # Remove aliased modes
irfft_2d!(domain, fields.bshat, fields.bₛ)
```

---

#### `dealias_2d!`

```julia
dealias_2d!(domain::Domain, Â::PencilArray{Complex{T},2}) -> Nothing
```

2D dealiasing for surface fields.

---

### Jacobian Operations

#### `jacobian!`

```julia
jacobian!(J::PencilArray{T,3}, ψ::PencilArray{T,3}, b::PencilArray{T,3}, 
         domain::Domain, workspace...) -> Nothing
```

Compute 3D Jacobian J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x.

---

#### `jacobian_2d!`

```julia
jacobian_2d!(J::PencilArray{T,2}, ψ::PencilArray{T,2}, b::PencilArray{T,2},
            domain::Domain, workspace...) -> Nothing
```

Compute 2D surface Jacobian.

**Example:**
```julia
jacobian_2d!(fields.tmp, fields.φₛ, fields.bₛ, domain,
            fields.tmpc_2d, fields.tmpc2_2d, fields.tmp2, fields.tmp3)
```

---

## Time Integration

### Problem Setup

#### `SemiGeostrophicProblem`

```julia
SemiGeostrophicProblem(domain::Domain; kwargs...) -> SemiGeostrophicProblem
```

Main problem structure encapsulating complete simulation.

**Keyword Arguments:**
- `scheme::TimeScheme = RK3`: Time integration scheme
- `dt::Real = 0.01`: Base time step size
- `initial_time::Real = 0.0`: Starting time
- `adaptive_dt::Bool = false`: Enable adaptive time stepping
- `filter_freq::Int = 10`: Spectral filtering frequency (steps)
- `filter_strength::Real = 0.1`: Filter strength (0-1)
- `enable_diagnostics::Bool = true`: Enable diagnostic computation
- `output_dir::String = "output"`: Output directory

**Valid `scheme` options:**
- `AB2_LowStorage`: 2nd order Adams-Bashforth (low memory)
- `RK3`: 3rd order Runge-Kutta
- `RK3_LowStorage`: 3rd order RK (low memory variant)

**Returns:** `SemiGeostrophicProblem{T}` structure.

**Example:**
```julia
# Basic problem setup
prob = SemiGeostrophicProblem(domain; dt=0.005, scheme=RK3)

# High-performance setup with filtering
prob = SemiGeostrophicProblem(domain;
                             dt=0.01,
                             scheme=AB2_LowStorage,
                             adaptive_dt=true,
                             filter_freq=20,
                             filter_strength=0.05,
                             enable_diagnostics=true)
```

---

### Time Stepping

#### `timestep!`

```julia
timestep!(fields::Fields, domain::Domain, params::TimeParams, state::TimeState) -> Real
```

Advances solution by one time step.

**Arguments:**
- `fields::Fields`: All field arrays
- `domain::Domain`: Grid and spectral information
- `params::TimeParams`: Integration parameters
- `state::TimeState`: Current time and step information

**Returns:** Actual time step used (may differ from `params.dt` if adaptive).

**Example:**
```julia
dt_used = timestep!(prob.fields, prob.domain, prob.timestepper, prob.clock)
```

---

#### `step_until!`

```julia
step_until!(prob::SemiGeostrophicProblem, stop_time::Real) -> Nothing
```

Integrates until specified time is reached.

**Arguments:**
- `prob::SemiGeostrophicProblem`: Complete problem structure
- `stop_time::Real`: Target simulation time

**Example:**
```julia
# Run simulation for 5 time units
step_until!(prob, 5.0)

# Continue for another 10 time units
step_until!(prob, 15.0)
```

---

#### `stepforward!`

```julia
stepforward!(prob::SemiGeostrophicProblem, nsteps::Int=1; kwargs...) -> Nothing
```

Takes specified number of time steps.

**Arguments:**
- `prob::SemiGeostrophicProblem`: Problem structure
- `nsteps::Int`: Number of steps to take

**Keyword Arguments:**
- `preserve_timestepper::Bool = false`: Don't modify original timestepper
- `custom_timestepper::Union{TimeParams, Nothing} = nothing`: Temporary timestepper

**Example:**
```julia
# Take 1000 steps with progress reporting
stepforward!(prob, 1000)

# Take 100 steps with custom timestepper
custom_params = TimeParams{Float64}(0.001; scheme=RK3, adaptive_dt=false)
stepforward!(prob, 100; custom_timestepper=custom_params)
```

---

### Time Parameters

#### `TimeParams`

```julia
TimeParams{T}(dt::T; kwargs...) -> TimeParams{T}
```

Time integration parameter structure.

**Arguments:**
- `dt::T`: Base time step size

**Keyword Arguments:**
- `scheme::TimeScheme = AB2_LowStorage`: Integration method
- `filter_freq::Int = 10`: Filtering frequency  
- `filter_strength::T = T(0.1)`: Filter strength
- `cfl_safety::T = T(0.5)`: CFL safety factor
- `max_dt::T = T(0.1)`: Maximum time step
- `min_dt::T = T(1e-6)`: Minimum time step
- `adaptive_dt::Bool = false`: Enable adaptive stepping

**Example:**
```julia
# Conservative parameters
params = TimeParams{Float64}(0.01;
                            scheme=RK3,
                            adaptive_dt=true,
                            cfl_safety=0.3,
                            filter_freq=5)
```

---

## Solver Operations

### SSG Equation Solver

#### `solve_ssg_equation`

```julia
solve_ssg_equation(Φ_initial, b_rhs, ε, domain; kwargs...) -> (solution, diagnostics)
```

Solves 3D Semi-Geostrophic equation: ∇²Φ = εDΦ with multigrid methods.

**Arguments:**
- `Φ_initial::PencilArray{T,3}`: Initial guess for streamfunction
- `b_rhs::PencilArray{T,3}`: Right-hand side (extended buoyancy field)
- `ε::Real`: SSG parameter (Rossby number measure)
- `domain::Domain`: Computational domain

**Keyword Arguments:**
- `tol::Real = 1e-8`: Convergence tolerance
- `maxiter::Int = 50`: Maximum iterations
- `verbose::Bool = false`: Enable progress output
- `n_levels::Int = 3`: Number of multigrid levels
- `smoother::Symbol = :spectral`: Smoother type

**Valid `smoother` options:**
- `:spectral`: Spectral preconditioning (recommended)
- `:adaptive`: Automatic smoother selection
- `:sor`: Successive over-relaxation
- `:enhanced`: Enhanced SOR with metric terms

**Returns:** 
- `solution::PencilArray{T,3}`: Converged streamfunction
- `diagnostics::NamedTuple`: Convergence information

**Diagnostics fields:**
- `converged::Bool`: Whether iteration converged
- `iterations::Int`: Number of iterations used
- `final_residual::Real`: Final residual norm
- `convergence_history::Vector`: Residual history
- `ε_parameter::Real`: Parameter used
- `solve_time::Real`: Wall-clock time

**Example:**
```julia
# Solve SSG equation
Φ₀ = create_real_field(domain, Float64)  # Initial guess
b_3d = extend_2d_to_3d(fields.bₛ, domain)  # RHS from surface buoyancy

solution, diag = solve_ssg_equation(Φ₀, b_3d, 0.1, domain;
                                   tol=1e-10,
                                   maxiter=100,
                                   smoother=:adaptive,
                                   verbose=true)

if diag.converged
    println("Converged in $(diag.iterations) iterations")
    println("Final residual: $(diag.final_residual)")
else
    @warn "Failed to converge after $(diag.iterations) iterations"
end
```

---

### Interface Solvers

#### `solve_monge_ampere_fields!`

```julia
solve_monge_ampere_fields!(fields::Fields, domain::Domain; kwargs...) -> Bool
```

Interface-compatible Monge-Ampère solver for existing SSG.jl code.

**Arguments:**
- `fields::Fields`: Field structure (buoyancy → streamfunction)
- `domain::Domain`: Computational domain

**Keyword Arguments:**
- `tol::Real = 1e-10`: Convergence tolerance
- `verbose::Bool = false`: Enable output
- `ε::Real = 0.1`: SSG parameter

**Returns:** `Bool` indicating convergence success.

**Example:**
```julia
# Set surface buoyancy
# ... initialize fields.bₛ ...

# Solve for streamfunction
success = solve_monge_ampere_fields!(fields, domain; tol=1e-8, verbose=true)

if success
    # fields.φₛ now contains surface streamfunction
    compute_surface_geostrophic_velocities!(fields, domain)
end
```

---

#### `solve_poisson_simple`

```julia
solve_poisson_simple(Φ_initial, b_rhs, domain; kwargs...) -> (solution, diagnostics)
```

Simple Poisson solver: ∇²φ = b (fallback when ε → 0).

**Arguments:**
- `Φ_initial::PencilArray{T,3}`: Initial guess (ignored)
- `b_rhs::PencilArray{T,3}`: Right-hand side
- `domain::Domain`: Domain structure

**Returns:** Solution and diagnostics (similar to `solve_ssg_equation`).

---

## Initial Conditions

### `set_b!`

```julia
set_b!(prob::SemiGeostrophicProblem, b_field, domain::Domain) -> Nothing
```

Sets buoyancy field and computes derived quantities.

**Arguments:**
- `prob::SemiGeostrophicProblem`: Problem structure
- `b_field`: Buoyancy field (PencilArray or compatible)
- `domain::Domain`: Domain structure

**Operations performed:**
1. Copy buoyancy to `prob.fields.bₛ`
2. Enforce zero mean constraint
3. Apply spectral dealiasing
4. Solve Monge-Ampère equation for streamfunction
5. Compute geostrophic velocities
6. Calculate initial energy diagnostics
7. Update diagnostics if enabled

**Example:**
```julia
# Initialize from array
b_init = zeros(size_local(domain.pr2d)...)
# ... fill b_init with initial condition ...
b_field = PencilArray(domain.pr2d, b_init)

set_b!(prob, b_field, domain)
```

---

### `set_φ!`

```julia
set_φ!(prob::SemiGeostrophicProblem, φ_field, domain::Domain) -> Nothing
```

Sets streamfunction and computes derived buoyancy.

**Arguments:**
- `prob::SemiGeostrophicProblem`: Problem structure  
- `φ_field`: Streamfunction field
- `domain::Domain`: Domain structure

**Example:**
```julia
# Alternative initialization via streamfunction
φ_init = PencilArray(domain.pr2d, ...)
set_φ!(prob, φ_init, domain)
```

---

### Built-in Initial Conditions

#### `initialize_taylor_green!`

```julia
initialize_taylor_green!(fields::Fields, domain::Domain; amplitude::Real=1.0) -> Nothing
```

Classical Taylor-Green vortex initial condition.

**Formula:** b(x,y) = amplitude × sin(x)cos(y)

---

#### `initialize_gaussian_vortex!`

```julia
initialize_gaussian_vortex!(fields::Fields, domain::Domain; 
                           amplitude::Real=1.0, width::Real=0.5, 
                           center_x::Real=π, center_y::Real=π) -> Nothing
```

Gaussian vortex with specified center and width.

---

#### `initialize_random_field!`

```julia
initialize_random_field!(fields::Fields, domain::Domain;
                         amplitude::Real=0.1, seed::Int=1234,
                         k_min::Int=1, k_max::Int=10) -> Nothing
```

Random initial condition with controlled spectral content.

---

#### `initialize_baroclinic_jet!`

```julia
initialize_baroclinic_jet!(fields::Fields, domain::Domain;
                          amplitude::Real=1.0, width::Real=π/4,
                          center_y::Real=π) -> Nothing
```

Baroclinic jet profile (ocean/atmospheric applications).

---

### Convenience Function

#### `set_initial_conditions!`

```julia
set_initial_conditions!(prob::SemiGeostrophicProblem, init_func::Function; kwargs...) -> Nothing
```

Applies initial condition function with parameters.

**Arguments:**
- `prob::SemiGeostrophicProblem`: Problem structure
- `init_func::Function`: Initialization function
- `kwargs...`: Parameters passed to `init_func`

**Example:**
```julia
# Apply Taylor-Green vortex with custom amplitude
set_initial_conditions!(prob, initialize_taylor_green!; amplitude=2.0)

# Apply Gaussian vortex with custom parameters  
set_initial_conditions!(prob, initialize_gaussian_vortex!; 
                       amplitude=1.5, width=0.3, center_x=π/2)
```

---

## Diagnostics

### Energy and Enstrophy

#### `compute_energy`

```julia
compute_energy(field::PencilArray, domain::Domain) -> Real
```

Computes spectral energy: ½∫|field|² dx dy using Parseval's theorem.

**Example:**
```julia
ke = compute_energy(fields.u, domain) + compute_energy(fields.v, domain)
```

---

#### `compute_enstrophy`

```julia
compute_enstrophy(field::PencilArray, domain::Domain) -> Real
```

Computes enstrophy of vorticity field.

---

#### `compute_kinetic_energy`

```julia
compute_kinetic_energy(fields::Fields, domain::Domain) -> Real
```

Total 3D kinetic energy: ½∫(u² + v²) dV.

---

#### `compute_surface_kinetic_energy`

```julia
compute_surface_kinetic_energy(fields::Fields, domain::Domain) -> Real
```

Surface kinetic energy using 2D spectral methods.

---

### Spectral Integrals

#### `parsevalsum`

```julia
parsevalsum(field::PencilArray, domain::Domain) -> Real
```

Spectral integral: ∫field dx dy using Parseval's theorem.

---

#### `parsevalsum2`

```julia
parsevalsum2(field::PencilArray, domain::Domain) -> Real  
```

Spectral squared integral: ∫field² dx dy.

**Example:**
```julia
total_buoyancy = parsevalsum(fields.bₛ, domain)
buoyancy_variance = parsevalsum2(fields.bₛ, domain)
```

---

### Velocity Computations

#### `compute_geostrophic_velocities!`

```julia
compute_geostrophic_velocities!(fields::Fields, domain::Domain) -> Nothing
```

Computes 3D geostrophic velocities from streamfunction.

**Formula:** u = -∂φ/∂y, v = ∂φ/∂x

---

#### `compute_surface_geostrophic_velocities!`

```julia
compute_surface_geostrophic_velocities!(fields::Fields, domain::Domain) -> Nothing
```

Computes surface velocities from 2D streamfunction.

---

### Field Statistics

#### `field_stats`

```julia
field_stats(field::PencilArray, name::String="field") -> NamedTuple
```

Basic field statistics: min, max, mean, std.

**Example:**
```julia
stats = field_stats(fields.bₛ, "surface buoyancy")
println("Min: $(stats.min), Max: $(stats.max), Mean: $(stats.mean)")
```

---

#### `enhanced_field_stats`

```julia
enhanced_field_stats(field::PencilArray, name::String="field") -> NamedTuple
```

Extended statistics including percentiles and extrema.

---

## Utility Functions

### Field Creation

#### `create_real_field`

```julia
create_real_field(domain::Domain, ::Type{T}, dims::Int=3) -> PencilArray{T}
```

Creates empty real field with proper dimensions.

**Arguments:**
- `domain::Domain`: Domain structure
- `T::Type`: Element type (Float32, Float64)
- `dims::Int`: Dimensions (2 or 3)

**Example:**
```julia
scratch_2d = create_real_field(domain, Float64, 2)
scratch_3d = create_real_field(domain, Float64, 3)
```

---

#### `create_spectral_field`

```julia
create_spectral_field(domain::Domain, ::Type{T}, dims::Int=3) -> PencilArray{Complex{T}}
```

Creates empty spectral field.

---

### Array Operations

#### `norm_field`

```julia
norm_field(field::PencilArray) -> Real
```

Computes L2 norm with MPI reduction.

**Example:**
```julia
residual_norm = norm_field(fields.R)
```

---

#### `inner_product`

```julia
inner_product(field1::PencilArray, field2::PencilArray) -> Real
```

Inner product ⟨field1, field2⟩ with MPI reduction.

---

### Vorticity

#### `compute_vorticity!`

```julia
compute_vorticity!(ω::PencilArray, u::PencilArray, v::PencilArray, domain::Domain) -> Nothing
```

Computes vorticity: ω = ∂v/∂x - ∂u/∂y.

**Example:**
```julia
ω = create_real_field(domain, Float64, 2)
compute_vorticity!(ω, fields.u, fields.v, domain)
```

---

## I/O Operations

### Simulation State

#### `save_simulation_state_full`

```julia
save_simulation_state_full(filename::String, fields::Fields, domain::Domain,
                           time::Real, step::Int; kwargs...) -> Nothing
```

Saves complete simulation state to file.

**Arguments:**
- `filename::String`: Output file path (.jld2 format)
- `fields::Fields`: All field data
- `domain::Domain`: Grid information
- `time::Real`: Current simulation time
- `step::Int`: Current step number

**Keyword Arguments:**
- `save_spectral::Bool = false`: Include spectral arrays
- `compression::Bool = true`: Enable compression

**Example:**
```julia
save_simulation_state_full("checkpoint_$(prob.clock.step).jld2",
                          prob.fields, prob.domain, 
                          prob.clock.t, prob.clock.step;
                          save_spectral=false)
```

---

#### `load_simulation_state_full`

```julia
load_simulation_state_full(filename::String) -> (fields, domain, time, step)
```

Loads complete simulation state from file.

**Returns:** Tuple of loaded data structures.

**Example:**
```julia
fields, domain, time, step = load_simulation_state_full("checkpoint_1000.jld2")
```

---

### Output Manager

#### `OutputManager`

```julia
OutputManager(prob::SemiGeostrophicProblem; kwargs...) -> OutputManager
```

Manages simulation output and checkpointing.

**Keyword Arguments:**
- `output_dir::String = "output"`: Output directory
- `save_freq::Int = 100`: Field saving frequency
- `checkpoint_freq::Int = 1000`: Checkpoint frequency
- `diagnostic_freq::Int = 10`: Diagnostic output frequency
