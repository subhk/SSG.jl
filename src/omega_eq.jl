###############################################################################
# omega_eq.jl
#
# computing vertical velocities in geostrophic coordinates for geophysical flows.
#
# ==============================================================================
# PHYSICAL BACKGROUND
# ==============================================================================
#
# This code solves coupled equations for geostrophic flow dynamics:
#
# 1. MONGE-AMPÈRE EQUATION:
#    ∇²Φ = ε·DΦ where DΦ = ∂ₓₓΦ·∂ᵧᵧΦ - (∂ₓᵧΦ)²
#    Φ is the Montgomery potential (geostrophic streamfunction)
#
# 2. VERTICAL VELOCITY EQUATION:
#    (∂ₓₓ + ∂ᵧᵧ + ∂ᵤᵤ)w* = -2ε∇·Q
#    where Q represents forcing terms from geostrophic momentum coordinates
#
# 3. GREEN'S FUNCTION SOLUTION:
#    Uses vertical Green's functions to solve the elliptic boundary value problem
#    with appropriate boundary conditions at top/bottom surfaces
#
# NUMERICAL APPROACH:
# - Spectral methods (FFT) for horizontal derivatives (high accuracy)
# - Green's function approach for vertical structure
# - Iterative solution of nonlinear Monge-Ampère equation
# - Thread-parallel implementation for performance
#
###############################################################################

using FFTW
using Statistics: mean
using LinearAlgebra
using Base.Threads
using JLD2
using Printf
using Trapz

# ============================================================================
# UTILITY FUNCTIONS AND TYPES
# ============================================================================

"""Optimized utility functions with @inbounds for better performance"""
@inline dropmean(A; dims=:) = dropdims(mean(A; dims=dims); dims=dims)
@inline dropsum(A; dims=:)  = dropdims(sum(A; dims=dims); dims=dims)

"""
Setup FFT plans with proper threading and device support
"""
function setup_fft_plans(dev, T, nx, ny, nthreads; effort=FFTW.MEASURE)
    # FFT plans
    FFTW.set_num_threads(nthreads)
    fftplan = plan_flows_fft(device_array(dev){Complex{T}, 2}(undef, nx, ny), flags=effort)
    rfftplan = plan_flows_rfft(device_array(dev){T, 2}(undef, nx, ny), flags=effort)
    
    return fftplan, rfftplan
end

"""
Computational workspace to avoid repeated allocations
"""
struct ComputationWorkspace{T<:AbstractFloat}
    # Temporary arrays for derivatives
    temp_complex1::Matrix{Complex{T}}
    temp_complex2::Matrix{Complex{T}} 
    temp_real1::Matrix{T}
    temp_real2::Matrix{T}
    temp_real3::Matrix{T}
    
    # 3D workspace arrays
    temp_3d_complex::Array{Complex{T}, 3}
    temp_3d_real::Array{T, 3}
    
    # Precomputed arrays
    K::Matrix{T}              # Horizontal wavenumber magnitude
    dfact::Matrix{T}          # Green's function denominator factor
    cosh_cache::Matrix{T}     # Cache for expensive cosh calculations
    
    function ComputationWorkspace(grid)
        T = eltype(grid)
        
        # Setup FFT plans if not already available in grid
        if !hasfield(typeof(grid), :fftplan) || !hasfield(typeof(grid), :rfftplan)
            @warn "Grid missing FFT plans. Consider using setup_fft_plans() during grid initialization."
        end
        
        # 2D temporary arrays
        temp_complex1 = zeros(Complex{T}, grid.nkr, grid.nl)
        temp_complex2 = zeros(Complex{T}, grid.nkr, grid.nl)
        temp_real1 = zeros(T, grid.nx, grid.ny)
        temp_real2 = zeros(T, grid.nx, grid.ny)
        temp_real3 = zeros(T, grid.nx, grid.ny)
        
        # 3D temporary arrays
        temp_3d_complex = zeros(Complex{T}, grid.nkr, grid.nl, grid.nz)
        temp_3d_real = zeros(T, grid.nx, grid.ny, grid.nz)
        
        # Precompute expensive operations
        K = sqrt.(grid.Krsq)
        dfact = similar(K)
        
        @threads for j in 1:grid.nl
            for i in 1:grid.nkr
                k_val = K[i,j]
                if k_val > 1e-14  # Avoid division by zero
                    @inbounds dfact[i,j] = sqrt(grid.invKrsq[i,j]) / sinh(k_val)
                else
                    @inbounds dfact[i,j] = 0.0
                end
            end
        end
        dfact[1,1] = 0.0
        
        # Cache for cosh calculations
        cosh_cache = zeros(T, grid.nkr, grid.nz)
        
        new{T}(temp_complex1, temp_complex2, temp_real1, temp_real2, temp_real3,
               temp_3d_complex, temp_3d_real, K, dfact, cosh_cache)
    end
end

# ============================================================================
# OPTIMIZED CORE FUNCTIONS
# ============================================================================

"""
Optimized Green's Function with precomputed values and better memory access
"""
function green_function!(Gₖ::AbstractMatrix{Complex{T}}, dfact::AbstractMatrix{T}, 
                        grid, K::AbstractMatrix{T}, z::T, z′::T) where {T}
    
    Lz = grid.Lz
    
    if z ≥ z′
        factor_z = z
        factor_z′ = z′ + Lz
    else
        factor_z = z + Lz  
        factor_z′ = z′
    end
    
    @inbounds @threads for j in 1:size(Gₖ, 2)
        for i in 1:size(Gₖ, 1)
            k_val = K[i,j]
            Gₖ[i,j] *= cosh(k_val * factor_z) * cosh(k_val * factor_z′) * dfact[i,j]
        end
    end
    
    return nothing
end

"""
Optimized forcing calculation for Monge-Ampère equation
Computes DΦ = ∂ₓₓΦ·∂ᵧᵧΦ - (∂ₓᵧΦ)² efficiently
"""
function compute_monge_ampere_forcing!(DΦₕ::AbstractMatrix{Complex{T}}, 
                                      sol::AbstractMatrix{Complex{T}},
                                      workspace::ComputationWorkspace{T}, 
                                      grid) where {T}
    
    # Extract workspace arrays for clarity
    temp1, temp2 = workspace.temp_complex1, workspace.temp_complex2
    Φxx, Φyy, Φxy = workspace.temp_real1, workspace.temp_real2, workspace.temp_real3
    
    # Compute ∂ₓₓΦ
    @inbounds @threads for j in 1:size(sol, 2)
        for i in 1:size(sol, 1)
            temp1[i,j] = -grid.kr[i,j]^2 * sol[i,j]
        end
    end
    ldiv!(Φxx, grid.rfftplan, temp1)

    # Compute ∂ᵧᵧΦ  
    @inbounds @threads for j in 1:size(sol, 2)
        for i in 1:size(sol, 1)
            temp2[i,j] = -grid.l[i,j]^2 * sol[i,j]
        end
    end
    ldiv!(Φyy, grid.rfftplan, temp2)
    
    # Compute ∂ₓᵧΦ
    @inbounds @threads for j in 1:size(sol, 2)
        for i in 1:size(sol, 1)
            temp1[i,j] = -grid.kr[i,j] * grid.l[i,j] * sol[i,j]
        end
    end
    ldiv!(Φxy, grid.rfftplan, temp1)

    # Compute DΦ = ∂ₓₓΦ·∂ᵧᵧΦ - (∂ₓᵧΦ)²
    @inbounds @threads for j in 1:size(Φxx, 2)
        for i in 1:size(Φxx, 1)
            Φxx[i,j] = Φxx[i,j] * Φyy[i,j] - Φxy[i,j]^2
        end
    end
    
    # Transform back to spectral space
    mul!(DΦₕ, grid.rfftplan, Φxx)
    
    return nothing
end

"""
Optimized Φ calculation with improved convergence monitoring and memory management
"""
function compute_phi!(sol_Φ3d::Array{Complex{T}, 3}, sol_b::AbstractMatrix{Complex{T}}, 
                     workspace::ComputationWorkspace{T}, grid;
                     max_iterations::Int=15, tolerance::T=1e-5) where {T}
    
    # Extract precomputed arrays
    K, dfact = workspace.K, workspace.dfact
    
    # Temporary arrays
    Φᵖₕ = copy(sol_Φ3d)  # Previous iteration
    DΦₕⁿ⁻¹ = similar(sol_Φ3d)
    Iₙ = workspace.temp_complex1
    
    # Initialize solution using Green's function
    @threads for kt in 1:grid.nz
        z_val = grid.z[kt] + grid.Lz
        for j in 1:grid.nl
            for i in 1:grid.nkr
                k_val = K[i,j]
                @inbounds sol_Φ3d[i,j,kt] = sol_b[i,j] * cosh(k_val * z_val) * dfact[i,j]
            end
        end
    end

    # Iterative solution
    converged = false
    
    for iteration in 1:max_iterations
        copyto!(Φᵖₕ, sol_Φ3d)
        
        # Compute forcing at each z-level
        @threads for kt in 1:grid.nz
            z = grid.z[kt]
            fill!(Iₙ, 0)  # Reset integrand
            
            # Integrate over all z′ levels
            for kt′ in 1:grid.nz
                z′ = grid.z[kt′]
                sol_view = @view Φᵖₕ[:,:,kt′]
                
                # Compute Monge-Ampère forcing
                local_forcing = similar(Iₙ)
                compute_monge_ampere_forcing!(local_forcing, sol_view, workspace, grid)
                
                # Apply Green's function
                green_function!(local_forcing, dfact, grid, K, z, z′)
                
                # Store for integration
                DΦₕⁿ⁻¹[:,:,kt′] .= local_forcing
            end

            # Trapezoidal integration over z′
            Iₙ_integrated = trapz(grid.z, DΦₕⁿ⁻¹, dims=3)

            # Update solution
            for j in 1:grid.nl
                for i in 1:grid.nkr
                    z_val = grid.z[kt] + grid.Lz
                    k_val = K[i,j]
                    @inbounds sol_Φ3d[i,j,kt] = (sol_b[i,j] * cosh(k_val * z_val) * dfact[i,j] + 
                                                 grid.ε * Iₙ_integrated[i,j])
                end
            end
        end

        # Convergence check using surface values
        error_estimate = compute_relative_error(sol_Φ3d[:,:,end], Φᵖₕ[:,:,end], grid)
        
        @printf "Iteration %d: Relative error = %.2e\n" iteration error_estimate
        
        if error_estimate < tolerance
            @printf "Converged after %d iterations\n" iteration
            converged = true
            break
        end
    end
    
    if !converged
        @warn "Φ calculation did not converge after $max_iterations iterations"
    end
    
    return converged
end

"""
Efficient relative error computation in physical space
"""
function compute_relative_error(sol_new::AbstractMatrix{Complex{T}}, 
                               sol_old::AbstractMatrix{Complex{T}}, grid) where {T}
    
    # Convert to physical space for meaningful error metric
    Φ_new = similar(sol_new, real(T), grid.nx, grid.ny)
    Φ_old = similar(sol_old, real(T), grid.nx, grid.ny)
    
    ldiv!(Φ_new, grid.rfftplan, sol_new)
    ldiv!(Φ_old, grid.rfftplan, sol_old)
    
    # Compute relative L2 error
    numerator = sum(abs2, Φ_new .- Φ_old)
    denominator = sum(abs2, Φ_new)
    
    return sqrt(numerator / (denominator + eps(T)))
end

"""
Optimized divergence calculation for Q vector
Q₁ = ∂ˣuᵍ·∂ˣb + ∂ˣvᵍ·∂ʸb  
Q₂ = ∂ʸuᵍ·∂ˣb + ∂ʸvᵍ·∂ʸb
"""
function compute_divergence_Q!(divQ_h::Array{Complex{T}, 3}, 
                              sol_Φ3d::Array{Complex{T}, 3},
                              sol_b::AbstractMatrix{Complex{T}},
                              workspace::ComputationWorkspace{T}, 
                              grid) where {T}
    
    # Precompute buoyancy gradients (independent of z)
    temp1, temp2 = workspace.temp_complex1, workspace.temp_complex2
    ∂ˣb, ∂ʸb = workspace.temp_real1, workspace.temp_real2
    
    # ∂ˣb = ik_r * b̂
    @inbounds @threads for j in 1:grid.nl
        for i in 1:grid.nkr
            temp1[i,j] = 1im * grid.kr[i,j] * sol_b[i,j]
        end
    end
    ldiv!(∂ˣb, grid.rfftplan, temp1)
    
    # ∂ʸb = il * b̂
    @inbounds @threads for j in 1:grid.nl
        for i in 1:grid.nkr
            temp1[i,j] = 1im * grid.l[i,j] * sol_b[i,j]
        end
    end
    ldiv!(∂ʸb, grid.rfftplan, temp1)

    # Process each z-level
    @threads for kt in 1:grid.nz
        # Local arrays for thread safety
        local_temp1 = similar(temp1)
        local_temp2 = similar(temp2)
        local_field = similar(workspace.temp_real3)
        local_Q₁ = similar(∂ˣb)
        local_Q₂ = similar(∂ˣb)
        
        fill!(local_Q₁, 0)
        fill!(local_Q₂, 0)
        
        # Compute velocity derivatives
        Φ_slice = @view sol_Φ3d[:,:,kt]
        
        # ∂ˣuᵍ = -∂ˣʸΦ
        @inbounds for j in 1:grid.nl, i in 1:grid.nkr
            local_temp1[i,j] = grid.kr[i,j] * grid.l[i,j] * Φ_slice[i,j]
        end
        ldiv!(local_field, grid.rfftplan, local_temp1)
        @. local_Q₁ += local_field * ∂ˣb
        
        # ∂ʸuᵍ = -∂ʸʸΦ  
        @inbounds for j in 1:grid.nl, i in 1:grid.nkr
            local_temp1[i,j] = grid.l[i,j]^2 * Φ_slice[i,j]
        end
        ldiv!(local_field, grid.rfftplan, local_temp1)
        @. local_Q₂ += local_field * ∂ˣb

        # ∂ˣvᵍ = ∂ˣˣΦ
        @inbounds for j in 1:grid.nl, i in 1:grid.nkr
            local_temp1[i,j] = -grid.kr[i,j]^2 * Φ_slice[i,j]
        end
        ldiv!(local_field, grid.rfftplan, local_temp1)
        @. local_Q₁ += local_field * ∂ʸb

        # ∂ʸvᵍ = ∂ˣʸΦ
        @inbounds for j in 1:grid.nl, i in 1:grid.nkr
            local_temp1[i,j] = -grid.kr[i,j] * grid.l[i,j] * Φ_slice[i,j]
        end
        ldiv!(local_field, grid.rfftplan, local_temp1)
        @. local_Q₂ += local_field * ∂ʸb

        # Transform Q components and compute divergence
        mul!(local_temp1, grid.rfftplan, local_Q₁)
        mul!(local_temp2, grid.rfftplan, local_Q₂)

        @inbounds for j in 1:grid.nl, i in 1:grid.nkr
            divQ_h[i,j,kt] = -2 * grid.ε * (1im * grid.kr[i,j] * local_temp1[i,j] + 
                                           1im * grid.l[i,j] * local_temp2[i,j])
        end
    end
    
    return nothing
end

"""
Optimized integral calculations with vectorized operations
"""
function compute_integral_1(divQ_h::Array{Complex{T}, 3}, 
                           workspace::ComputationWorkspace{T}, grid) where {T}
    
    Integral_1 = similar(divQ_h)
    K, invK = workspace.K, sqrt.(grid.invKrsq)

    @threads for j in 1:grid.nl
        for i in 1:grid.nkr
            k_val = K[i,j]
            invk_val = invK[i,j]
            factor = 0.5 * invk_val
            
            for kt in 1:grid.nz
                z_current = grid.z[kt]
                
                # Vectorized integration from kt to end
                if kt < grid.nz
                    z_vals = @view grid.z[kt:end]
                    divQ_vals = @view divQ_h[i,j,kt:end]
                    
                    # Compute integrand efficiently
                    integrand = similar(z_vals, Complex{T})
                    @. integrand = exp(-k_val * z_vals) * divQ_vals * factor
                    
                    integral_result = trapz(z_vals, integrand)
                    Integral_1[i,j,kt] = -integral_result * exp(k_val * z_current)
                else
                    Integral_1[i,j,kt] = 0
                end
            end
        end
    end
    
    return Integral_1
end

function compute_integral_2(divQ_h::Array{Complex{T}, 3}, 
                           workspace::ComputationWorkspace{T}, grid) where {T}
    
    Integral_2 = similar(divQ_h)
    K, invK = workspace.K, sqrt.(grid.invKrsq)

    @threads for j in 1:grid.nl
        for i in 1:grid.nkr
            k_val = K[i,j]
            invk_val = invK[i,j]
            factor = 0.5 * invk_val
            
            for kt in 1:grid.nz
                z_current = grid.z[kt]
                
                if kt < grid.nz
                    z_vals = @view grid.z[kt:end]
                    divQ_vals = @view divQ_h[i,j,kt:end]
                    
                    integrand = similar(z_vals, Complex{T})
                    @. integrand = exp(k_val * z_vals) * divQ_vals * factor
                    
                    integral_result = trapz(z_vals, integrand)
                    Integral_2[i,j,kt] = -integral_result * exp(-k_val * z_current)
                else
                    Integral_2[i,j,kt] = 0
                end
            end
        end
    end
    
    return Integral_2
end

function compute_integral_3(divQ_h::Array{Complex{T}, 3}, 
                           workspace::ComputationWorkspace{T}, grid) where {T}
    
    Integral_3 = similar(divQ_h)
    K, invK = workspace.K, sqrt.(grid.invKrsq)
    
    # Temporary arrays for integration
    t₃ = similar(divQ_h)
    t₄ = similar(divQ_h)

    # Precompute exponential factors
    @threads for kt in 1:grid.nz
        z_val = grid.z[kt]
        for j in 1:grid.nl
            for i in 1:grid.nkr
                k_val = K[i,j]
                invk_val = invK[i,j]
                factor = 0.5 * invk_val
                divQ_val = divQ_h[i,j,kt]
                
                @inbounds t₃[i,j,kt] = exp(-k_val * z_val) * divQ_val * factor
                @inbounds t₄[i,j,kt] = exp(k_val * z_val) * divQ_val * factor
            end
        end
    end

    # Compute integrals over entire domain
    I₂ = trapz(grid.z, t₃, dims=3)
    t₁ = trapz(grid.z, t₄, dims=3)
    
    # Final computation
    @threads for j in 1:grid.nl
        for i in 1:grid.nkr
            k_val = K[i,j]
            
            if k_val > 1e-14
                I₁_val = -2 * sinh(k_val)
                I₂_val = exp(-k_val) * I₂[i,j] - exp(k_val) * t₁[i,j]
                factor = 2 * I₂_val / I₁_val
                
                for kt in 1:grid.nz
                    @inbounds Integral_3[i,j,kt] = factor * sinh(k_val * grid.z[kt])
                end
            else
                # Handle k=0 case
                @. Integral_3[i,j,:] = 0
            end
        end
    end
    
    return Integral_3
end

"""
Main function for vertical velocity calculation with comprehensive optimization
"""
function compute_vertical_velocity(vars, grid; 
                                  data_file::String="ep_0.2/SurfaceSG_nx_512.jld2",
                                  snapshot::Int=2401,
                                  phi_tolerance::Real=1e-5,
                                  phi_max_iter::Int=15)
    
    println("Starting vertical velocity computation...")
    
    # Initialize workspace
    workspace = ComputationWorkspace(grid)
    
    # Load buoyancy data efficiently
    @time "Data loading" begin
        b₁ = jldopen(data_file, "r") do file
            file["/snapshots/b/$snapshot"]
        end
    end
    
    # Transform to spectral space
    sol_b = similar(workspace.temp_complex1)
    mul!(sol_b, grid.rfftplan, b₁)
    
    # Initialize arrays
    sol_Φ3d = workspace.temp_3d_complex
    w = workspace.temp_3d_real
    invJac = similar(w)
    
    # Solve for geostrophic streamfunction
    @time "Computing Φ" begin
        converged = compute_phi!(sol_Φ3d, sol_b, workspace, grid; 
                               max_iterations=phi_max_iter, tolerance=phi_tolerance)
        if !converged
            @warn "Φ computation did not fully converge"
        end
    end
    
    # Compute divergence of Q
    divQ_h = similar(sol_Φ3d)
    @time "Computing divergence Q" compute_divergence_Q!(divQ_h, sol_Φ3d, sol_b, workspace, grid)

    # Calculate Jacobian inverse (coordinate transformation factor)
    @time "Computing inverse Jacobian" begin
        @threads for kt in 1:grid.nz
            local_temp = similar(sol_b)
            local_field = similar(workspace.temp_real1)
            
            # Compute J⁻¹ = 1 - ε(∂ₓₓΦ + ∂ᵧᵧΦ)
            fill!(local_field, 1.0)  # Start with identity
            
            Φ_slice = @view sol_Φ3d[:,:,kt]
            
            # ∂ₓₓΦ contribution
            @inbounds for j in 1:grid.nl, i in 1:grid.nkr
                local_temp[i,j] = -grid.kr[i,j]^2 * Φ_slice[i,j]
            end
            ldiv!(workspace.temp_real2, grid.rfftplan, local_temp)
            @. local_field -= grid.ε * workspace.temp_real2
            
            # ∂ᵧᵧΦ contribution  
            @inbounds for j in 1:grid.nl, i in 1:grid.nkr
                local_temp[i,j] = -grid.l[i,j]^2 * Φ_slice[i,j]
            end
            ldiv!(workspace.temp_real2, grid.rfftplan, local_temp)
            @. local_field -= grid.ε * workspace.temp_real2
            
            # Store result
            invJac[:,:,kt] .= local_field
        end
    end

    # Compute vertical velocity integrals
    @time "Computing integrals" begin
        Integral_1 = compute_integral_1(divQ_h, workspace, grid)
        Integral_2 = compute_integral_2(divQ_h, workspace, grid)  
        Integral_3 = compute_integral_3(divQ_h, workspace, grid)
    end
    
    # Combine integrals for final w* solution
    w_spectral = Integral_1 .+ Integral_2 .+ Integral_3
    
    # Transform to physical space and apply Jacobian
    @time "Final transformation" begin
        @threads for kt in 1:grid.nz
            local_w = similar(workspace.temp_real1)
            ldiv!(local_w, grid.rfftplan, w_spectral[:,:,kt])
            
            # Apply coordinate transformation: w = w*/J
            @inbounds for j in 1:grid.ny, i in 1:grid.nx
                w[i,j,kt] = local_w[i,j] / invJac[i,j,kt]
            end
        end
    end

    println("Vertical velocity computation completed successfully!")
    return w
end


"""
Example grid initialization showing proper FFT plan setup
"""
function create_optimized_grid(nx, ny, nz, Lx, Ly, Lz, dev, T; 
                              nthreads=Threads.nthreads(), 
                              effort=FFTW.MEASURE)
    
    println("Setting up optimized grid with FFT plans...")
    
    # Create coordinate arrays
    x = range(0, Lx, length=nx+1)[1:end-1]  # Periodic, exclude endpoint
    y = range(0, Ly, length=ny+1)[1:end-1]
    z = range(-1, 0, length=nz)  # Vertical coordinate
    
    # Wavenumber arrays
    kx = fftfreq(nx, 2π*nx/Lx)
    ky = fftfreq(ny, 2π*ny/Ly)
    kr = rfftfreq(nx, 2π*nx/Lx)
    
    # Create 2D wavenumber grids
    Kx = reshape(kx, (nx, 1))
    Ky = reshape(ky, (1, ny))
    Kr = reshape(kr, (length(kr), 1))
    L = reshape(ky, (1, ny))
    
    # Squared wavenumbers
    Krsq = Kr.^2 .+ L.^2
    invKrsq = similar(Krsq)
    @. invKrsq = ifelse(Krsq == 0, 0, 1/Krsq)
    
    # Setup FFT plans with threading
    fftplan, rfftplan = setup_fft_plans(dev, T, nx, ny, nthreads; effort=effort)
    
    # Create grid structure (this would be your actual grid type)
    grid = (
        nx=nx, ny=ny, nz=nz, nkr=length(kr), nl=ny,
        Lx=Lx, Ly=Ly, Lz=Lz,
        x=x, y=y, z=z,
        kr=Kr, l=L, Krsq=Krsq, invKrsq=invKrsq,
        fftplan=fftplan, rfftplan=rfftplan,
        device=dev, ε=0.2  # Example epsilon value
    )
    
    println("Grid setup complete:")
    println("  Resolution: $(nx)×$(ny)×$(nz)")
    println("  Domain: $(Lx)×$(Ly)×$(Lz)")
    println("  FFT threads: $(FFTW.get_num_threads())")
    println("  Julia threads: $(Threads.nthreads())")
    
    return grid
end

"""
Example driver function showing proper usage with grid setup
"""
function demo_vertical_velocity(; 
                               nx=512, ny=512, nz=64,
                               Lx=2π, Ly=2π, Lz=1.0,
                               data_file="ep_0.2/SurfaceSG_nx_512.jld2",
                               snapshot=2401,
                               epsilon=0.2,
                               dev=CPU(),
                               precision=Float64)
    
    println("Demo: Computing vertical velocity in geostrophic coordinates")
    println("ε = $epsilon, snapshot = $snapshot")
    
    # Setup optimized grid with proper FFT plans
    grid = create_optimized_grid(nx, ny, nz, Lx, Ly, Lz, dev, precision)
    
    # Setup variables (this would be your variable initialization)
    vars = (u=zeros(precision, nx, ny), v=zeros(precision, nx, ny))
    
    # Run computation
    @time "Total computation" w = compute_vertical_velocity(vars, grid; 
                                                           data_file=data_file,
                                                           snapshot=snapshot)
    
    println("Computation complete. Results ready for analysis.")
    println("Vertical velocity field size: $(size(w))")
    println("Min/Max w: $(extrema(w))")
    
    return w, grid
end
