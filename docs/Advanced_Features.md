# Advanced Features

##  Overview

This document covers the **most sophisticated capabilities** of SSG.jl, including advanced solvers, high-performance computing features, and cutting-edge numerical methods. These features are designed for **research applications** and **production-scale simulations**.

##  Table of Contents

1. [3D Semi-Geostrophic Solver](#3d-semi-geostrophic-solver)
2. [Multigrid Methods](#multigrid-methods)
3. [MPI Parallelization](#mpi-parallelization)
4. [Spectral Filtering](#spectral-filtering)
5. [Non-Uniform Grids](#non-uniform-grids)
6. [Adaptive Time Stepping](#adaptive-time-stepping)
7. [Advanced Diagnostics](#advanced-diagnostics)
8. [Performance Optimization](#performance-optimization)
9. [Custom Extensions](#custom-extensions)

---

## 3D Semi-Geostrophic Solver

### Overview

The **3D SSG solver** is the most advanced component of SSG.jl, implementing a sophisticated **multigrid method** for the nonlinear Semi-Geostrophic equation:

```math
\nabla^2 \Phi = \varepsilon D\Phi
```

where **DΦ** is the highly nonlinear **Monge-Ampère operator**.

### Core Solver Interface

```julia
# High-level interface
solution, diagnostics = solve_ssg_equation(Φ_initial, b_rhs, ε, domain;
                                          tol=1e-8,
                                          maxiter=50,
                                          smoother=:adaptive,
                                          verbose=true)

# Advanced configuration
solution, diag = solve_ssg_equation(Φ₀, b_3d, 0.1, domain;
                                   tol=1e-12,           # Ultra-high precision
                                   maxiter=200,         # More iterations  
                                   n_levels=5,          # Deep multigrid
                                   smoother=:spectral,  # Spectral preconditioning
                                   ω=0.7,              # Custom relaxation
                                   verbose=true)
```

### Solver Components

#### Nonlinear Operator Implementation
```julia
# The DΦ operator: (∂²Φ/∂X²)(∂²Φ/∂Y²) - (∂²Φ/∂X∂Y)²
function compute_d_operator!(level::SSGLevel{T}, result::PencilArray{T, 3}) where T
    domain = level.domain
    
    # Spectral computation of mixed derivative ∂²Φ/∂X∂Y
    rfft!(domain, level.Φ, level.Φ_hat)
    ddx!(domain, level.Φ_hat, level.tmp_spec)
    ddy!(domain, level.tmp_spec, level.tmp_spec)
    irfft!(domain, level.tmp_spec, level.Φ_xy)
    
    # Pure second derivatives
    # ... [implementation details] ...
    
    # Combine: DΦ = Φ_xx * Φ_yy - (Φ_xy)²
    @inbounds for k in axes(result_local, 3)
        for j in axes(result_local, 2)
            @simd for i in axes(result_local, 1)
                result_local[i,j,k] = Φ_xx_local[i,j,k] * Φ_yy_local[i,j,k] - Φ_xy_local[i,j,k]^2
            end
        end
    end
end
```

#### Boundary Condition Handling
```julia
# Advanced boundary conditions with non-uniform grids
function apply_neumann_surface!(Φ_local::Array{T,3}, i::Int, j::Int, k::Int, 
                                bc_value::T, dz::Vector{T}, nz_local::Int) where T
    # Second-order backward difference for non-uniform grid
    h1 = dz[k-1]  # Spacing to level below
    h2 = dz[k-2]  # Spacing two levels below
    
    # Compute finite difference coefficients
    α = h1 + h2
    a = (2*h1 + h2) / (h1 * α)
    b = -α / (h1 * h2)  
    c = h1 / (h2 * α)
    
    # Apply BC: ∂Φ/∂Z = bc_value
    return (bc_value - b*Φ_local[i,j,k-1] - c*Φ_local[i,j,k-2]) / a
end
```

### Interface Compatibility

```julia
# Drop-in replacement for existing Monge-Ampère solver
function solve_monge_ampere_fields!(fields::Fields{T}, domain::Domain; 
                                   tol::T=T(1e-10), ε::T=T(0.1)) where T
    
    # Convert 2D surface problem to 3D
    b_3d = extend_2d_to_3d(fields.bₛ, domain)
    
    # Solve full 3D SSG equation
    solution, diag = solve_ssg_equation(fields.φ, b_3d, ε, domain; 
                                       tol=tol, smoother=:spectral)
    
    # Extract surface solution
    extract_surface_to_2d!(fields.φₛ, solution, domain)
    
    return diag.converged
end
```

---

## Multigrid Methods

### Geometric Multigrid Hierarchy

SSG.jl implements a **sophisticated V-cycle multigrid** method with multiple smoothing options:

```julia
# Multigrid solver structure
mutable struct SSGMultigridSolver{T<:AbstractFloat}
    levels::Vector{SSGLevel{T}}      # Grid hierarchy
    n_levels::Int                    # Number of levels
    ε::T                             # SSG parameter
    smoother_type::Symbol            # Smoother selection
    ω::T                            # Relaxation parameter
    convergence_history::Vector{T}   # Residual history
end
```

### Advanced Smoothing Strategies

#### 1. Spectral Smoother (Recommended)
```julia
function ssg_spectral_smoother!(level::SSGLevel{T}, iters::Int, ω::T, ε::T) where T
    for iter = 1:iters
        # Compute residual
        compute_ssg_residual!(level, ε)
        
        # Transform to spectral space
        rfft!(domain, level.r, level.r_hat)
        
        # Spectral preconditioning
        @inbounds for k in axes(level.Φ_hat.data, 3)
            for j in range_local_j
                for i in range_local_i
                    kx, ky = domain.kx[i], domain.ky[j]
                    k_mag_sq = kx^2 + ky^2
                    
                    if k_mag_sq > 1e-14
                        # Preconditioner: 1/(k² + ε·k⁴)
                        ε_factor = ε * k_mag_sq
                        preconditioner = 1.0 / (k_mag_sq + ε_factor + 1e-12)
                        
                        correction = level.r_hat.data[i,j,k] * preconditioner
                        level.Φ_hat.data[i,j,k] += ω * correction
                    end
                end
            end
        end
        
        # Transform back and apply boundary conditions
        irfft!(domain, level.Φ_hat, level.Φ)
        apply_ssg_boundary_conditions!(level)
    end
end
```

#### 2. Adaptive SOR Smoother
```julia
function ssg_sor_smoother_adaptive!(level::SSGLevel{T}, iters::Int, ω::T, ε::T) where T
    # Analyze grid stretching
    dz = level.domain.dz
    max_stretch = maximum(dz[2:end] ./ dz[1:end-1])
    
    # Choose method based on grid properties
    if max_stretch < 2.0
        ssg_sor_smoother!(level, iters, ω, ε)          # Standard SOR
    elseif max_stretch < 5.0
        ssg_sor_smoother_enhanced!(level, iters, ω*0.9, ε)  # Enhanced SOR
    else
        ssg_sor_smoother_enhanced!(level, iters, ω*0.8, ε; use_metrics=true)  # Full metrics
    end
end
```

#### 3. Enhanced SOR for Stretched Grids
```julia
function ssg_sor_smoother_enhanced!(level::SSGLevel{T}, iters::Int, ω::T, ε::T;
                                   use_metrics::Bool=true) where T
    
    # Precompute finite difference coefficients for efficiency
    α_coeff = zeros(T, nz_local)
    β_coeff = zeros(T, nz_local) 
    γ_coeff = zeros(T, nz_local)
    
    @inbounds for k = 2:nz_local-1
        h_below = dz[k-1]
        h_above = dz[k]
        h_total = h_below + h_above
        
        # Second-order finite difference coefficients
        α_coeff[k] = 2.0 / (h_below * h_total)      # Lower neighbor
        β_coeff[k] = -2.0 / (h_below * h_above)     # Center point
        γ_coeff[k] = 2.0 / (h_above * h_total)      # Upper neighbor
        
        # Optional: Metric correction for highly stretched grids
        if use_metrics
            stretch_ratio = max(h_above, h_below) / min(h_above, h_below)
            if stretch_ratio > 5.0
                smooth_factor = min(0.9, 5.0 / stretch_ratio)
                α_coeff[k] *= smooth_factor
                β_coeff[k] *= smooth_factor  
                γ_coeff[k] *= smooth_factor
            end
        end
    end
    
    # Red-black Gauss-Seidel with precomputed coefficients
    for iter = 1:iters
        for color = 0:1
            @inbounds for k = 2:nz_local-1
                for j = 2:ny_local-1
                    for i = 2:nx_local-1
                        if (i + j + k) % 2 != color
                            continue
                        end
                        
                        # Current and neighbor values
                        Φ_c = Φ_local[i,j,k]
                        Φ_e, Φ_w = Φ_local[i+1,j,k], Φ_local[i-1,j,k]
                        Φ_n, Φ_s = Φ_local[i,j+1,k], Φ_local[i,j-1,k]
                        Φ_u, Φ_d = Φ_local[i,j,k+1], Φ_local[i,j,k-1]
                        
                        # Diagonal coefficient (including linearization)
                        diag_coeff = -2*inv_dx2 - 2*inv_dy2 + β_coeff[k] - ε*0.1
                        
                        # Off-diagonal contributions
                        off_diag = (Φ_e + Φ_w)*inv_dx2 + (Φ_n + Φ_s)*inv_dy2 + 
                                  α_coeff[k]*Φ_d + γ_coeff[k]*Φ_u
                        
                        # SOR update with adaptive relaxation
                        if abs(diag_coeff) > 1e-14
                            Φ_new = -off_diag / diag_coeff
                            
                            # Adaptive relaxation for stretched grids
                            relax_factor = ω
                            if use_metrics
                                stretch_ratio = max(dz[k], dz[k-1]) / min(dz[k], dz[k-1])
                                if stretch_ratio > 3.0
                                    relax_factor *= min(1.0, 3.0 / stretch_ratio)
                                end
                            end
                            
                            Φ_local[i,j,k] = Φ_c + relax_factor * (Φ_new - Φ_c)
                        end
                    end
                end
            end
        end
    end
end
```

### V-Cycle Implementation

```julia
function ssg_v_cycle!(mg::SSGMultigridSolver{T}, level::Int=1) where T
    if level == mg.n_levels
        # Coarsest level - direct solve with many iterations
        if mg.smoother_type == :spectral
            ssg_spectral_smoother!(mg.levels[level], 50, mg.ω, mg.ε)
        else
            ssg_sor_smoother_adaptive!(mg.levels[level], 50, mg.ω, mg.ε)
        end
        return
    end
    
    current = mg.levels[level]
    coarser = mg.levels[level + 1]
    
    # Pre-smoothing (3-5 iterations)
    n_pre = mg.smoother_type == :spectral ? 3 : 5
    apply_smoother!(current, n_pre, mg)
    
    # Compute and restrict residual
    compute_ssg_residual!(current, mg.ε)
    restrict_ssg_3d!(coarser, current)
    
    # Store current solution for correction computation
    copy_field!(coarser.Φ_old, coarser.Φ)
    zero_field!(coarser.Φ)  # Zero initial guess for correction
    
    # Recursive V-cycle call
    ssg_v_cycle!(mg, level + 1)
    
    # Compute correction and prolongate
    coarser.Φ.data .-= coarser.Φ_old.data  # Correction = new - old
    prolongate_ssg_3d!(current, coarser)
    
    # Post-smoothing (3-5 iterations)
    n_post = mg.smoother_type == :spectral ? 3 : 5
    apply_smoother!(current, n_post, mg)
end
```

---

## MPI Parallelization

### Domain Decomposition Strategy

SSG.jl uses **PencilArrays.jl** for optimal **2D domain decomposition**:

```julia
# Automatic process grid optimization
function create_pencil_decomposition(Nx::Int, Ny::Int, Nz::Int, comm::MPI.Comm)
    nprocs = MPI.Comm_size(comm)
    
    # Create 3D pencils (X-Y decomposition, full Z on each process)
    pencil_3d_real = Pencil((Nx, Ny, Nz), comm; 
                           decomp_dims=(true, true, false))  # Decompose X,Y only
    
    # Create 2D surface pencils
    pencil_2d_real = Pencil((Nx, Ny), comm;
                           decomp_dims=(true, true))
    
    return pencil_3d_real, pencil_2d_real
end
```

### Advanced MPI Operations

#### Custom Reductions for Diagnostics
```julia
function compute_global_energy_spectrum(field::PencilArray{Complex{T},2}, domain::Domain) where T
    # Local energy spectrum computation
    local_spectrum = zeros(T, div(domain.Nx, 2) + 1)
    range_locals = range_local(field.pencil)
    
    for (j_local, j_global) in enumerate(range_locals[2])
        ky = domain.ky[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            kx = domain.kx[i_global]
            k_mag = sqrt(kx^2 + ky^2)
            k_idx = min(round(Int, k_mag) + 1, length(local_spectrum))
            
            if k_idx > 0
                local_spectrum[k_idx] += abs2(field.data[i_local, j_local])
            end
        end
    end
    
    # MPI reduction to combine spectra from all processes
    global_spectrum = MPI.Allreduce(local_spectrum, MPI.SUM, field.pencil.comm)
    return global_spectrum
end
```

#### Scalable I/O Operations
```julia
function parallel_field_output(filename::String, field::PencilArray, domain::Domain)
    comm = field.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Use MPI-IO for parallel writing
    if rank == 0
        # Master process writes metadata
        jldopen(filename, "w") do file
            file["domain/Nx"] = domain.Nx
            file["domain/Ny"] = domain.Ny
            file["domain/Nz"] = domain.Nz
            file["domain/Lx"] = domain.Lx
            file["domain/Ly"] = domain.Ly
            file["domain/Lz"] = domain.Lz
        end
    end
    
    MPI.Barrier(comm)  # Ensure metadata written first
    
    # Each process writes its local data
    jldopen(filename, "r+") do file
        range_locals = range_local(field.pencil)
        local_data = field.data
        
        # Create unique key for this process
        process_key = "data/process_$rank"
        file[process_key * "/data"] = local_data
        file[process_key * "/ranges"] = collect.(range_locals)
    end
end
```

### Load Balancing Optimization

```julia
function optimize_process_grid(Nx::Int, Ny::Int, nprocs::Int)
    # Find optimal factorization of nprocs into px × py
    best_px, best_py = 1, nprocs
    min_communication = Inf
    
    for px in 1:nprocs
        if nprocs % px == 0
            py = div(nprocs, px)
            
            # Estimate communication cost
            # Horizontal slices: communication proportional to Ny/py
            # Vertical slices: communication proportional to Nx/px
            comm_cost = (Ny / py) + (Nx / px)
            
            # Prefer balanced decomposition
            aspect_penalty = abs(log(px/py))  # Penalize extreme aspect ratios
            total_cost = comm_cost + aspect_penalty
            
            if total_cost < min_communication
                min_communication = total_cost
                best_px, best_py = px, py
            end
        end
    end
    
    return best_px, best_py
end
```

---

## Spectral Filtering

### Advanced Filter Types

#### 1. Exponential Filter
```julia
struct ExponentialFilter <: AbstractSpectralFilter
    strength::Float64        # Filter strength (0-1)
    order::Int              # Filter order (higher = sharper cutoff)
    cutoff_fraction::Float64 # Fraction of Nyquist frequency
end

function apply_filter!(field_hat::PencilArray{Complex{T},N}, 
                      filter::ExponentialFilter, domain::Domain) where {T,N}
    
    range_locals = range_local(field_hat.pencil)
    field_local = field_hat.data
    
    kx_max = maximum(abs, domain.kx)
    ky_max = maximum(abs, domain.ky)
    k_cutoff = filter.cutoff_fraction * min(kx_max, ky_max)
    
    @inbounds for k in axes(field_local, 3)
        for (j_local, j_global) in enumerate(range_locals[2])
            ky = domain.ky[j_global]
            for (i_local, i_global) in enumerate(range_locals[1])
                kx = domain.kx[i_global]
                
                k_mag = sqrt(kx^2 + ky^2)
                if k_mag > k_cutoff
                    # Exponential filter: exp(-α((k-k_c)/k_c)^n)
                    ratio = (k_mag - k_cutoff) / k_cutoff
                    filter_factor = exp(-filter.strength * ratio^filter.order)
                    field_local[i_local, j_local, k] *= filter_factor
                end
            end
        end
    end
end
```

#### 2. Hyperviscosity Filter
```julia
struct HyperviscosityFilter <: AbstractSpectralFilter
    viscosity::Float64       # Hyperviscosity coefficient
    order::Int              # Derivative order (4, 6, 8, ...)
end

function apply_filter!(field_hat::PencilArray{Complex{T},N}, 
                      filter::HyperviscosityFilter, domain::Domain) where {T,N}
    
    range_locals = range_local(field_hat.pencil)
    field_local = field_hat.data
    
    @inbounds for k in axes(field_local, 3)
        for (j_local, j_global) in enumerate(range_locals[2])
            ky = domain.ky[j_global]
            for (i_local, i_global) in enumerate(range_locals[1])
                kx = domain.kx[i_global]
                
                k_mag_sq = kx^2 + ky^2
                if k_mag_sq > 1e-14
                    # Hyperviscous damping: exp(-ν k^n Δt)
                    damping = exp(-filter.viscosity * k_mag_sq^(filter.order/2))
                    field_local[i_local, j_local, k] *= damping
                end
            end
        end
    end
end
```

#### 3. Cesàro Filter (Smooth Cutoff)
```julia
struct CesaroFilter <: AbstractSpectralFilter
    cutoff_wavenumber::Float64
    transition_width::Float64
end

function apply_filter!(field_hat::PencilArray{Complex{T},N}, 
                      filter::CesaroFilter, domain::Domain) where {T,N}
    
    @inbounds for k in axes(field_hat.data, 3)
        for (j_local, j_global) in enumerate(range_local(field_hat.pencil)[2])
            ky = domain.ky[j_global]
            for (i_local, i_global) in enumerate(range_local(field_hat.pencil)[1])
                kx = domain.kx[i_global]
                
                k_mag = sqrt(kx^2 + ky^2)
                
                # Smooth transition using tanh function
                transition = 0.5 * (1.0 - tanh((k_mag - filter.cutoff_wavenumber) / filter.transition_width))
                field_hat.data[i_local, j_local, k] *= transition
            end
        end
    end
end
```

### Adaptive Filtering

```julia
function adaptive_spectral_filter!(fields::Fields{T}, domain::Domain; 
                                  target_cfl::T=T(0.5)) where T
    
    # Compute current maximum velocity
    compute_geostrophic_velocities!(fields, domain)
    u_max = maximum(abs, fields.u.data)
    v_max = maximum(abs, fields.v.data)
    
    # Get global maximum
    comm = fields.u.pencil.comm
    u_max_global = MPI.Allreduce(u_max, MPI.MAX, comm)
    v_max_global = MPI.Allreduce(v_max, MPI.MAX, comm)
    
    # Estimate current CFL
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    current_cfl = max(u_max_global * dx, v_max_global * dy)  # Assuming dt=1 for estimate
    
    # Adjust filter strength based on CFL
    if current_cfl > target_cfl
        # Strong filtering needed
        filter_strength = 0.2 * (current_cfl / target_cfl)^2
        filter = ExponentialFilter(min(filter_strength, 0.8), 8, 0.6)
    else
        # Mild filtering sufficient
        filter_strength = 0.05
        filter = ExponentialFilter(filter_strength, 4, 0.8)
    end
    
    # Apply adaptive filter
    rfft_2d!(domain, fields.bₛ, fields.bshat)
    apply_filter!(fields.bshat, filter, domain)
    irfft_2d!(domain, fields.bshat, fields.bₛ)
end
```

---

## Non-Uniform Grids

### Advanced Grid Generation

#### 1. Hyperbolic Tangent Stretching
```julia
function create_tanh_grid(Nz::Int, Lz::Float64; β::Float64=2.0, 
                         surface_concentration::Bool=true)
    
    # Uniform coordinate in computational space
    η = range(0, 1, length=Nz)
    
    if surface_concentration
        # Concentrate points near surface (η=1 → z=0)
        z = zeros(Float64, Nz)
        for i = 1:Nz
            ξ = 2*η[i] - 1  # Map to [-1, 1]
            z[i] = -Lz * (1 + tanh(β * ξ) / tanh(β)) / 2
        end
    else
        # Concentrate points near bottom (η=0 → z=-Lz)
        z = -Lz .* (1 .- (1 .+ tanh.(β .* (2 .* η .- 1))) ./ (2 * tanh(β)))
    end
    
    # Compute grid spacing
    dz = zeros(Float64, Nz-1)
    for i = 1:Nz-1
        dz[i] = z[i+1] - z[i]
    end
    
    return z, dz
end
```

#### 2. Power Law Stretching
```julia
function create_power_grid(Nz::Int, Lz::Float64; α::Float64=1.5)
    
    η = range(0, 1, length=Nz)
    
    # Power law transformation
    z = -Lz .* (1 .- η.^α)
    
    # Grid spacing
    dz = zeros(Float64, Nz-1)
    for i = 1:Nz-1
        dz[i] = z[i+1] - z[i]
    end
    
    return z, dz
end
```

#### 3. Custom Grid from Function
```julia
function create_custom_grid(Nz::Int, Lz::Float64, 
                           grid_function::Function; kwargs...)
    
    η = range(0, 1, length=Nz)
    z = [-Lz * grid_function(η_i; kwargs...) for η_i in η]
    
    # Ensure monotonicity and proper bounds
    z = sort(z, rev=true)  # Decreasing from 0 to -Lz
    z[end] = 0.0    # Exact surface
    z[1] = -Lz      # Exact bottom
    
    dz = diff(z)
    return z, dz
end

# Example: Exponential clustering
exponential_grid(η; β=2.0) = (exp(β*η) - 1) / (exp(β) - 1)
z, dz = create_custom_grid(64, 1000.0, exponential_grid; β=3.0)
```

### Grid Quality Assessment

```julia
function analyze_grid_quality(z::Vector{T}, dz::Vector{T}) where T
    Nz = length(z)
    
    # Stretching ratios
    stretch_ratios = dz[2:end] ./ dz[1:end-1]
    max_stretch = maximum(stretch_ratios)
    avg_stretch = mean(stretch_ratios)
    
    # Grid smoothness (second derivative of spacing)
    d2z = diff(dz)
    smoothness = norm(d2z) / norm(dz[1:end-1])
    
    # Aspect ratios (for 3D grid analysis)
    min_dz = minimum(dz)
    max_dz = maximum(dz)
    aspect_ratio = max_dz / min_dz
    
    return (
        max_stretch = max_stretch,
        avg_stretch = avg_stretch,
        aspect_ratio = aspect_ratio,
        smoothness = smoothness,
        min_spacing = min_dz,
        max_spacing = max_dz,
        quality_score = 1.0 / (1.0 + 0.1*max_stretch + 0.05*aspect_ratio + 0.1*smoothness)
    )
end

# Usage
quality = analyze_grid_quality(domain.z, domain.dz)
if quality.max_stretch > 5.0
    @warn "High grid stretching detected ($(round(quality.max_stretch, digits=2))). Consider enhanced smoothers."
end
```

---

## Adaptive Time Stepping

### Advanced CFL-Based Adaptation

```julia
mutable struct AdaptiveTimestepper{T}
    dt_base::T                    # Base time step
    dt_current::T                # Current time step
    dt_min::T                    # Minimum allowed
    dt_max::T                    # Maximum allowed
    
    cfl_target::T                # Target CFL number
    cfl_safety::T                # Safety factor
    
    # Adaptation parameters
    increase_factor::T           # Max increase per step
    decrease_factor::T           # Max decrease per step
    stability_margin::T          # Extra safety for stability
    
    # History tracking
    dt_history::CircularBuffer{T}
    cfl_history::CircularBuffer{T}
    rejected_steps::Int
    
    function AdaptiveTimestepper{T}(dt_base::T; kwargs...) where T
        # Set defaults
        dt_min = get(kwargs, :dt_min, dt_base / 100)
        dt_max = get(kwargs, :dt_max, dt_base * 10)
        cfl_target = get(kwargs, :cfl_target, T(0.4))
        cfl_safety = get(kwargs, :cfl_safety, T(0.8))
        
        # Conservative adaptation rates
        increase_factor = get(kwargs, :increase_factor, T(1.1))
        decrease_factor = get(kwargs, :decrease_factor, T(0.5))
        stability_margin = get(kwargs, :stability_margin, T(0.1))
        
        # Initialize history
        history_length = get(kwargs, :history_length, 10)
        dt_history = CircularBuffer{T}(history_length)
        cfl_history = CircularBuffer{T}(history_length)
        
        new{T}(dt_base, dt_base, dt_min, dt_max,
               cfl_target, cfl_safety,
               increase_factor, decrease_factor, stability_margin,
               dt_history, cfl_history, 0)
    end
end

function adapt_timestep!(stepper::AdaptiveTimestepper{T}, 
                        fields::Fields{T}, domain::Domain) where T
    
    # Compute current CFL number
    cfl = compute_cfl_number(fields, domain, stepper.dt_current)
    
    # Record history
    push!(stepper.cfl_history, cfl)
    push!(stepper.dt_history, stepper.dt_current)
    
    # Compute new time step
    dt_new = stepper.dt_current
    
    if cfl > stepper.cfl_safety
        # Decrease time step aggressively
        reduction_factor = min(stepper.decrease_factor, 
                              stepper.cfl_target / cfl)
        dt_new *= reduction_factor
        stepper.rejected_steps += 1
        
    elseif cfl < stepper.cfl_target * (1 - stepper.stability_margin)
        # Cautiously increase time step
        if length(stepper.cfl_history) >= 3
            # Check that CFL has been stable
            recent_cfls = stepper.cfl_history[end-2:end]
            if all(c -> c < stepper.cfl_target * 0.9, recent_cfls)
                increase_factor = min(stepper.increase_factor,
                                     stepper.cfl_target / cfl * 0.9)  # Conservative
                dt_new *= increase_factor
            end
        end
    end
    
    # Apply bounds
    dt_new = clamp(dt_new, stepper.dt_min, stepper.dt_max)
    
    # Additional stability checks
    if length(stepper.dt_history) >= 2
        # Prevent oscillatory behavior
        recent_change = stepper.dt_history[end] / stepper.dt_history[end-1]
        if recent_change < 0.8 && dt_new > stepper.dt_current
            dt_new = stepper.dt_current  # Don't increase after recent decrease
        end
    end
    
    stepper.dt_current = dt_new
    return dt_new
end
```

### Embedded Error Estimation

```julia
function timestep_with_error_control!(fields::Fields{T}, domain::Domain,
                                     stepper::AdaptiveTimestepper{T},
                                     state::TimeState{T}) where T
    
    # Take full step
    fields_full = copy_fields(fields)
    take_full_step!(fields_full, domain, stepper.dt_current, state)
    
    # Take two half steps  
    fields_half = copy_fields(fields)
    take_half_step!(fields_half, domain, stepper.dt_current/2, state)
    take_half_step!(fields_half, domain, stepper.dt_current/2, state)
    
    # Estimate error
    error_field = similar(fields.bₛ)
    error_field.data .= fields_full.bₛ.data .- fields_half.bₛ.data
    
    # Compute error norm
    error_norm = norm_field(error_field)
    solution_norm = norm_field(fields_half.bₛ)
    relative_error = error_norm / (solution_norm + 1e-14)
    
    # Error-based time step adaptation
    tolerance = 1e-6
    if relative_error > tolerance
        # Error too large - reject step and reduce dt
        dt_new = stepper.dt_current * 0.8 * (tolerance / relative_error)^(1/3)
        stepper.dt_current = max(dt_new, stepper.dt_min)
        stepper.rejected_steps += 1
        
        # Retry with smaller step
        return timestep_with_error_control!(fields, domain, stepper, state)
    else
        # Accept the more accurate solution
        copy_fields!(fields, fields_half)
        
        # Potentially increase time step for next iteration
        if relative_error < tolerance * 0.1
            dt_new = stepper.dt_current * min(1.2, (tolerance / relative_error)^(1/4))
            stepper.dt_current = min(dt_new, stepper.dt_max)
        end
        
        return stepper.dt_current
    end
end
```

---

## Advanced Diagnostics

### Spectral Energy Analysis

```julia
function compute_energy_spectrum_2d(field::PencilArray{T,2}, domain::Domain) where T
    # Transform to spectral space
    field_hat = create_spectral_field(domain, T, 2)
    rfft_2d!(domain, field, field_hat)
    
    # Compute radial energy spectrum
    kmax = min(domain.Nx ÷ 2, domain.Ny ÷ 2)
    energy_spectrum = zeros(T, kmax)
    
    range_locals = range_local(field_hat.pencil)
    field_hat_local = field_hat.data
    
    # Local contribution to spectrum
    for (j_local, j_global) in enumerate(range_locals[2])
        ky = domain.ky[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            kx = domain.kx[i_global]
            
            k_mag = sqrt(kx^2 + ky^2)
            k_idx = round(Int, k_mag) + 1
            
            if 1 <= k_idx <= kmax
                # Energy density in wavenumber space
                energy_density = abs2(field_hat_local[i_local, j_local])
                energy_spectrum[k_idx] += energy_density
            end
        end
    end
    
    # MPI reduction to get global spectrum
    global_spectrum = MPI.Allreduce(energy_spectrum, MPI.SUM, field_hat.pencil.comm)
    
    # Normalize by shell area (2πk for 2D)
    k_values = collect(0:kmax-1)
    for k = 2:kmax
        global_spectrum[k] /= (2π * k_values[k])
    end
    
    return k_values, global_spectrum
end

# Usage example
k, E_k = compute_energy_spectrum_2d(fields.bₛ, domain)
```

### Cascade Analysis

```julia
function compute_energy_flux(fields::Fields{T}, domain::Domain; 
                           shell_width::Int=1) where T
    
    # Compute nonlinear term in spectral space
    compute_jacobian!(fields.tmp, fields.φₛ, fields.bₛ, fields, domain)
    rfft_2d!(domain, fields.tmp, fields.tmpc_2d)
    
    # Also need buoyancy in spectral space
    rfft_2d!(domain, fields.bₛ, fields.bshat)
    
    # Energy flux computation
    kmax = min(domain.Nx ÷ 2, domain.Ny ÷ 2)
    energy_flux = zeros(T, kmax)
    
    range_locals = range_local(fields.bshat.pencil)
    
    for (j_local, j_global) in enumerate(range_locals[2])
        ky = domain.ky[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            kx = domain.kx[i_global]
            
            k_mag = sqrt(kx^2 + ky^2)
            k_idx = round(Int, k_mag) + 1
            
            if 1 <= k_idx <= kmax
                # Flux = -Re(b̂* · Ĵ) where Ĵ is Jacobian in spectral space
                b_hat = fields.bshat.data[i_local, j_local]
                J_hat = fields.tmpc_2d.data[i_local, j_local]
                
                flux_contribution = -real(conj(b_hat) * J_hat)
                energy_flux[k_idx] += flux_contribution
            end
        end
    end
    
    # MPI reduction
    global_flux = MPI.Allreduce(energy_flux, MPI.SUM, fields.bshat.pencil.comm)
    
    return global_flux
end
```

### Lagrangian Diagnostics

```julia
struct LagrangianTracer{T}
    x::Vector{T}                 # Current positions
    y::Vector{T}
    x0::Vector{T}                # Initial positions  
    y0::Vector{T}
    active::Vector{Bool}         # Whether tracers are in domain
    
    # Accumulated properties
    displacement_sq::Vector{T}   # |r(t) - r(0)|²
    trajectory_length::Vector{T} # ∫|dr/dt|dt
    
    function LagrangianTracer{T}(x0::Vector{T}, y0::Vector{T}) where T
        n = length(x0)
        new{T}(copy(x0), copy(y0), x0, y0, 
               fill(true, n), zeros(T, n), zeros(T, n))
    end
end

function advect_tracers!(tracers::LagrangianTracer{T}, 
                        fields::Fields{T}, domain::Domain, dt::T) where T
    
    # Compute velocities if needed
    compute_surface_geostrophic_velocities!(fields, domain)
    
    n_tracers = length(tracers.x)
    u_local = fields.u.data[:,:,1]  # Surface level
    v_local = fields.v.data[:,:,1]
    
    for i = 1:n_tracers
        if !tracers.active[i]
            continue
        end
        
        # Interpolate velocity at tracer position
        u_interp = interpolate_field(u_local, tracers.x[i], tracers.y[i], domain)
        v_interp = interpolate_field(v_local, tracers.x[i], tracers.y[i], domain)
        
        # Update position (simple Euler - could use RK4 for accuracy)
        dx = u_interp * dt
        dy = v_interp * dt
        
        tracers.x[i] += dx
        tracers.y[i] += dy
        
        # Update diagnostics
        velocity_mag = sqrt(u_interp^2 + v_interp^2)
        tracers.trajectory_length[i] += velocity_mag * dt
        
        displacement_x = tracers.x[i] - tracers.x0[i]
        displacement_y = tracers.y[i] - tracers.y0[i]
        tracers.displacement_sq[i] = displacement_x^2 + displacement_y^2
        
        # Check if tracer left domain (with periodic boundaries)
        if tracers.x[i] < 0
            tracers.x[i] += domain.Lx
        elseif tracers.x[i] >= domain.Lx
            tracers.x[i] -= domain.Lx
        end
        
        if tracers.y[i] < 0
            tracers.y[i] += domain.Ly
        elseif tracers.y[i] >= domain.Ly
            tracers.y[i] -= domain.Ly
        end
    end
end

# Diffusivity estimation
function compute_effective_diffusivity(tracers::LagrangianTracer{T}, t::T) where T
    active_tracers = tracers.active
    if sum(active_tracers) == 0
        return T(0)
    end
    
    # Mean squared displacement
    mean_disp_sq = mean(tracers.displacement_sq[active_tracers])
    
    # Effective diffusivity: κ = <|r(t) - r(0)|²> / (4t)
    return mean_disp_sq / (4 * t)
end
```

---

## Performance Optimization

### Memory-Efficient Operations

```julia
# Zero-allocation field operations using views
function jacobian_2d_optimized!(J::PencilArray{T,2}, ψ::PencilArray{T,2}, b::PencilArray{T,2},
                               workspace::NTuple{4, PencilArray}) where T
    
    # Unpack workspace to avoid allocations
    ψ_hat, b_hat, tmp1_hat, tmp2_hat = workspace
    
    # Transform fields to spectral space (reusing pre-allocated arrays)
    rfft_2d!(domain, ψ, ψ_hat)
    rfft_2d!(domain, b, b_hat)
    
    # Compute ∂ψ/∂x and ∂b/∂y
    ddx_2d!(domain, ψ_hat, tmp1_hat)    # tmp1_hat = ∂ψ/∂x
    ddy_2d!(domain, b_hat, tmp2_hat)    # tmp2_hat = ∂b/∂y
    
    # Transform back to physical space (overwriting input arrays safely)
    irfft_2d!(domain, tmp1_hat, ψ)      # ψ now contains ∂ψ/∂x
    irfft_2d!(domain, tmp2_hat, tmp)    # tmp contains ∂b/∂y
    
    # First part of Jacobian: (∂ψ/∂x)(∂b/∂y)
    @views @. J.data = ψ.data * tmp.data
    
    # Compute ∂ψ/∂y and ∂b/∂x (reusing spectral arrays)
    rfft_2d!(domain, ψ, ψ_hat)     # Transform original ψ again
    rfft_2d!(domain, b, b_hat)     # Transform original b again
    
    ddy_2d!(domain, ψ_hat, tmp1_hat)    # tmp1_hat = ∂ψ/∂y
    ddx_2d!(domain, b_hat, tmp2_hat)    # tmp2_hat = ∂b/∂x
    
    irfft_2d!(domain, tmp1_hat, ψ)      # ψ now contains ∂ψ/∂y
    irfft_2d!(domain, tmp2_hat, tmp)    # tmp contains ∂b/∂x
    
    # Complete Jacobian: J = (∂ψ/∂x)(∂b/∂y) - (∂ψ/∂y)(∂b/∂x)
    @views @. J.data -= ψ.data * tmp.data
    
    return nothing
end
```

### SIMD-Optimized Loops

```julia
using LoopVectorization

function compute_nonlinear_term_simd!(result::Array{T,3}, 
                                     Φ_xx::Array{T,3}, Φ_yy::Array{T,3}, 
                                     Φ_xy::Array{T,3}) where T
    
    @turbo for k in axes(result, 3)
        for j in axes(result, 2)
            for i in axes(result, 1)
                result[i,j,k] = Φ_xx[i,j,k] * Φ_yy[i,j,k] - Φ_xy[i,j,k]^2
            end
        end
    end
end

# Benchmarking utilities
function benchmark_operation(op_func::Function, args...; warmup=3, samples=10)
    # Warmup
    for _ = 1:warmup
        op_func(args...)
    end
    
    # Timing
    times = zeros(samples)
    for i = 1:samples
        times[i] = @elapsed op_func(args...)
    end
    
    return (
        min_time = minimum(times),
        median_time = median(times),
        mean_time = mean(times),
        std_time = std(times)
    )
end
```

### Cache-Aware Algorithms

```julia
# Blocked matrix operations for better cache utilization
function blocked_spectral_multiply!(output::Array{Complex{T},3}, 
                                   input::Array{Complex{T},3},
                                   multipliers::Array{T,2};
                                   block_size::Int=32) where T
    
    nx, ny, nz = size(input)
    
    # Process in blocks to improve cache locality
    for k_block = 1:block_size:nz
        k_end = min(k_block + block_size - 1, nz)
        
        for j_block = 1:block_size:ny  
            j_end = min(j_block + block_size - 1, ny)
            
            for i_block = 1:block_size:nx
                i_end = min(i_block + block_size - 1, nx)
                
                # Process block
                @inbounds for k = k_block:k_end
                    for j = j_block:j_end
                        mult_j = multipliers[1, j]  # Assuming multipliers are 2D
                        for i = i_block:i_end
                            mult_i = multipliers[i, 1]
                            combined_mult = mult_i * mult_j
                            output[i,j,k] = input[i,j,k] * combined_mult
                        end
                    end
                end
            end
        end
    end
end
```

---

## Custom Extensions

### Plugin Architecture

```julia
# Abstract interfaces for extensibility
abstract type AbstractSolver end
abstract type AbstractInitialCondition end
abstract type AbstractDiagnostic end

# Example: Custom solver interface
struct CustomSSGSolver <: AbstractSolver
    name::String
    parameters::Dict{Symbol, Any}
    solve_function::Function
end

function solve!(solver::CustomSSGSolver, problem::SemiGeostrophicProblem)
    return solver.solve_function(problem, solver.parameters)
end

# Plugin registry
const SOLVER_REGISTRY = Dict{String, Type{<:AbstractSolver}}()

function register_solver!(name::String, solver_type::Type{<:AbstractSolver})
    SOLVER_REGISTRY[name] = solver_type
    @info "Registered custom solver: $name"
end

# Usage
register_solver!("MyCustomSolver", CustomSSGSolver)
```

### Domain-Specific Languages

```julia
# Macro for simplified problem setup
macro ssg_problem(domain_expr, setup_block)
    quote
        # Create domain from expression
        domain = $(esc(domain_expr))
        
        # Initialize problem
        prob = SemiGeostrophicProblem(domain)
        
        # Execute setup block with problem in scope
        let prob = prob
            $(esc(setup_block))
        end
        
        prob
    end
end

# Usage example
prob = @ssg_problem make_domain(128, 128, 16) begin
    # Custom setup code with access to 'prob'
    set_initial_conditions!(prob, initialize_taylor_green!; amplitude=2.0)
    prob.timestepper.dt = 0.005
    prob.timestepper.scheme = RK3
    prob.timestepper.adaptive_dt = true
end
```

### Research Extensions

```julia
# Framework for experimental features
module ExperimentalFeatures

using ..SSG

"""
Experimental: Higher-order time integration schemes
"""
function timestep_rk4!(fields, domain, params, state)
    # 4th order Runge-Kutta implementation
    # ... experimental code ...
end

"""
Experimental: Spectral element methods for complex domains
"""
function create_spectral_element_domain(...)
    # Spectral element domain setup
    # ... experimental code ...
end

"""
Experimental: Machine learning-enhanced filtering
"""
mutable struct MLFilter <: AbstractSpectralFilter
    model::Any  # ML model for adaptive filtering
    training_data::Vector{Array{Float64}}
    
    MLFilter() = new(nothing, [])
end

function train_filter!(filter::MLFilter, simulation_data)
    # Train ML model on simulation data
    # ... experimental ML code ...
end

end  # module ExperimentalFeatures
```
