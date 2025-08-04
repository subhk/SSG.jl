"""
    run_initial_condition_example()

Complete example of setting up and analyzing initial conditions.
"""
function run_initial_condition_example()
    # Initialize MPI
    MPI.Initialized() || MPI.Init()
    
    try
        # Problem setup matching paper parameters
        Nx, Ny, Nz = 512, 512, 20
        Lx = Ly = 6.0  # L/L_D = 6 from paper
        Lz = 1.0
        
        # Create domain with exponential vertical spacing (surface concentrated)
        domain = make_domain(Nx, Ny, Nz; 
                           Lx=Lx, Ly=Ly, Lz=Lz,
                           z_boundary=:dirichlet,
                           z_grid=:stretched,
                           stretch_params=(type=:exponential, β=2.0, surface_concentration=true),
                           comm=MPI.COMM_WORLD)
        
        # Allocate fields structure
        fields = allocate_fields(domain)
        
        # Set initial conditions following equation (20)
        amplitude = 1.0  # Will be adjusted automatically
        k₀ = 14.0       # Peak wavenumber (non-dimensional)
        m = 20          # Spectral slope parameter
        target_rms = 1.0 # Target RMS velocity in m/s
        
        if MPI.Comm_rank(domain.pc.comm) == 0
            println("🌊 Setting up initial conditions from spectral perturbation")
            println("Parameters:")
            println("  Domain: $(Nx)×$(Ny)×$(Nz), L/L_D = $(Lx)")
            println("  k₀ = $(k₀) (peak wavenumber)")
            println("  m = $(m) (spectral slope)")
            println("  Target RMS velocity = $(target_rms) m/s")
            println()
        end
        
        # Initialize with spectral perturbation
        initialize_spectral_buoyancy!(fields, domain, amplitude, k₀, m;
                                     target_rms_velocity=target_rms,
                                     seed=12345)
        
        # Analyze initial state
        if MPI.Comm_rank(domain.pc.comm) == 0
            println("✓ Initial conditions set successfully")
            
            # Compute diagnostics using functions from transforms.jl
            ke = compute_energy(fields.u, fields.v, domain)
            b_variance = compute_buoyancy_variance(fields.bₛ, domain)
            
            # Local field statistics
            max_b = maximum(abs.(fields.bₛ.data))
            max_u = maximum(abs.(fields.u.data))
            max_v = maximum(abs.(fields.v.data))
            
            # Global reductions
            max_b_global = MPI.Allreduce(max_b, MPI.MAX, domain.pc.comm)
            max_u_global = MPI.Allreduce(max_u, MPI.MAX, domain.pc.comm)
            max_v_global = MPI.Allreduce(max_v, MPI.MAX, domain.pc.comm)
            
            println("\nInitial state diagnostics:")
            println("  Kinetic energy: $(ke)")
            println("  Buoyancy variance: $(b_variance)")
            println("  Max |buoyancy|: $(max_b_global)")
            println("  Max |u|: $(max_u_global) m/s")
            println("  Max |v|: $(max_v_global) m/s")
            
            # Verify RMS velocity
            final_rms = compute_rms_velocity(fields, domain)
            println("  Final RMS velocity: $(final_rms) m/s")
            println()
        end
        
        # Plot initial spectrum
        plot_initial_spectrum(fields, domain; k₀=k₀, m=m)
        
        # Save initial state (if output functions are available)
        if MPI.Comm_rank(domain.pc.comm) == 0
            try
                # Save basic field data using JLD2 (simpler than full state save)
                using JLD2
                jldopen("initial_state_spectral.jld2", "w") do file
                    file["buoyancy"] = fields.bₛ.data
                    file["streamfunction"] = fields.φ.data
                    file["u_velocity"] = fields.u.data
                    file["v_velocity"] = fields.v.data
                    file["parameters/k0"] = k₀
                    file["parameters/m"] = m
                    file["parameters/amplitude"] = amplitude
                    file["parameters/target_rms"] = target_rms
                    file["grid/Nx"] = Nx
                    file["grid/Ny"] = Ny
                    file["grid/Nz"] = Nz
                    file["grid/Lx"] = Lx
                    file["grid/Ly"] = Ly
                    file["grid/Lz"] = Lz
                end
                println("✓ Initial state saved to initial_state_spectral.jld2")
            catch e
                println("⚠️  Could not save initial state: $e")
            end
        end
        
        if MPI.Comm_rank(domain.pc.comm) == 0
            println("\nReady for time integration!")
            println("Next steps:")
            println("  1. Create SemiGeostrophicProblem or use fields directly")
            println("  2. Set up time stepping with timestep.jl")
            println("  3. Use: step_until!(prob, target_time) to advance simulation")
        end
        
        return fields, domain
        
    catch e
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println("❌ Error in initial condition setup: $e")
            println(stacktrace())
        end
        rethrow(e)
    finally
        # Don't finalize MPI here as it might be used elsewhere
        # MPI.Finalize()
    end
end

# ============================================================================
# ALTERNATIVE INITIALIZATION METHODS
# ============================================================================

"""
    initialize_taylor_green_perturbed!(fields::Fields{T}, domain::Domain,
                                      tg_amplitude::T, spectral_amplitude::T,
                                      k₀::T, m::Int) where T

Initialize with Taylor-Green vortex plus spectral perturbation.
Useful for studying transition to turbulence.
"""
function initialize_taylor_green_perturbed!(fields::Fields{T}, domain::Domain,
                                          tg_amplitude::T, 
                                          spectral_amplitude::T,
                                          k₀::T, m::Int) where T
    
    # First set Taylor-Green base flow
    initialize_taylor_green_base!(fields, domain, tg_amplitude)
    
    # Store base state
    b_base = copy(fields.bₛ.data)
    
    # Add spectral perturbation
    initialize_spectral_buoyancy!(fields, domain, spectral_amplitude, k₀, m; 
                                 target_rms_velocity=T(0.1))
    
    # Combine base flow and perturbation
    fields.bₛ.data .+= b_base
    
    # Re-solve for consistent state
    solve_monge_ampere_wrapper!(fields, domain)
    
    return nothing
end

"""
    initialize_taylor_green_base!(fields::Fields{T}, domain::Domain, amplitude::T) where T

Initialize Taylor-Green vortex base flow.
"""
function initialize_taylor_green_base!(fields::Fields{T}, domain::Domain, amplitude::T) where T
    local_ranges = local_range(fields.bₛ.pencil)
    b_local = fields.bₛ.data
    
    @inbounds for (j_local, j_global) in enumerate(local_ranges[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(local_ranges[1])
            x = domain.x[i_global]
            
            # Taylor-Green pattern: sin(x)cos(y)
            b_local[i_local, j_local] = amplitude * sin(2π*x/domain.Lx) * cos(2π*y/domain.Ly)
        end
    end
    
    return nothing
end

"""
    initialize_realistic_ocean_front!(fields::Fields{T}, domain::Domain;
                                     front_width::T=T(0.1),
                                     front_strength::T=T(1.0),
                                     noise_level::T=T(0.05)) where T

Initialize with realistic ocean front plus small-scale perturbations.
"""
function initialize_realistic_ocean_front!(fields::Fields{T}, domain::Domain;
                                          front_width::T=T(0.1),
                                          front_strength::T=T(1.0),
                                          noise_level::T=T(0.05)) where T
    
    # Get local grid points
    local_ranges = local_range(fields.bₛ.pencil)
    b_local = fields.bₛ.data
    
    # Create front structure
    @inbounds for (j_local, j_global) in enumerate(local_ranges[2])
        y = domain.y[j_global]
        y_normalized = 2 * (y / domain.Ly) - 1  # y ∈ [-1, 1]
        
        for (i_local, i_global) in enumerate(local_ranges[1])
            x = domain.x[i_global]
            
            # Tanh front profile
            front_profile = front_strength * tanh(y_normalized / front_width)
            
            # Add small-scale noise
            noise = noise_level * front_strength * (2 * rand(T) - 1)
            
            b_local[i_local, j_local] = front_profile + noise
        end
    end
    
    # Remove mean and solve for consistent state
    remove_mean!(fields.bₛ)
    solve_monge_ampere_wrapper!(fields, domain)
    
    return nothing
end

# ============================================================================
# INTEGRATION WITH EXISTING SSG STRUCTURE  
# ============================================================================

"""
    create_problem_with_spectral_ics(Nx::Int, Ny::Int, Nz::Int;
                                    k₀::Real=14.0, m::Int=20,
                                    target_rms::Real=1.0, kwargs...)

Create a complete SemiGeostrophicProblem with spectral initial conditions.
"""
function create_problem_with_spectral_ics(Nx::Int, Ny::Int, Nz::Int;
                                         Lx::Real=6.0, Ly::Real=6.0, Lz::Real=1.0,
                                         k₀::Real=14.0, m::Int=20,
                                         target_rms::Real=1.0,
                                         scheme=RK3,
                                         dt::Real=0.001,
                                         kwargs...)
    
    # Create domain
    domain = make_domain(Nx, Ny, Nz; Lx=Lx, Ly=Ly, Lz=Lz, kwargs...)
    
    # Create fields
    fields = allocate_fields(domain)
    
    # Set spectral initial conditions
    initialize_spectral_buoyancy!(fields, domain, 1.0, k₀, m; 
                                 target_rms_velocity=target_rms)
    
    # Create timestepper
    timestepper = TimeParams{Float64}(dt; scheme=scheme)
    clock = TimeState{Float64, typeof(fields.bₛ)}(0.0, fields)
    
    # Create problem structure
    prob = SemiGeostrophicProblem{Float64}(fields, domain, timestepper, clock;
                                         enable_diagnostics=true)
    
    return prob
end

"""
    demo_spectral_initialization()

Demonstrate different initialization methods.
"""
function demo_spectral_initialization()
    println("🌊 SSG Spectral Initialization Demo")
    println("=" ^ 40)
    
    # Test different initialization methods
    methods = [
        ("spectral_only", "Pure spectral perturbation (equation 20)"),
        ("taylor_green_perturbed", "Taylor-Green + spectral perturbations"),
        ("ocean_front", "Realistic ocean front with noise")
    ]
    
    for (method, description) in methods
        println("\n$(uppercase(method)): $description")
        println("-" ^ 50)
        
        try
            # Smaller domain for demo
            domain = make_domain(64, 64, 8; Lx=2π, Ly=2π, Lz=1.0)
            fields = allocate_fields(domain)
            
            if method == "spectral_only"
                initialize_spectral_buoyancy!(fields, domain, 1.0, 14.0, 20)
                
            elseif method == "taylor_green_perturbed"
                initialize_taylor_green_perturbed!(fields, domain, 1.0, 0.1, 14.0, 20)
                
            elseif method == "ocean_front"
                initialize_realistic_ocean_front!(fields, domain; 
                                                front_width=0.1, 
                                                front_strength=1.0,
                                                noise_level=0.05)
            end
            
            # Basic diagnostics
            rms_vel = compute_rms_velocity(fields, domain)
            ke = compute_energy(fields.u, fields.v, domain)
            
            if MPI.Comm_rank(domain.pc.comm) == 0
                println("  ✓ Initialization successful")
                println("    RMS velocity: $(rms_vel) m/s")
                println("    Kinetic energy: $(ke)")
            end
            
        catch e
            if MPI.Comm_rank(MPI.COMM_WORLD) == 0
                println("  ❌ Error: $e")
            end
        end
    end
    
    println("\n🎯 All initialization methods demonstrated!")
    return true
end

# ============================================================================
# EXAMPLE USAGE AND DOCUMENTATION 
# ============================================================================

"""
# Example Usage

## Basic Spectral Initialization
```julia
using SSG

# Create domain and fields
domain = make_domain(512, 512, 20; Lx=6.0, Ly=6.0, Lz=1.0)
fields = allocate_fields(domain)

# Set spectral initial conditions following equation (20)
initialize_spectral_buoyancy!(fields, domain, 1.0, 14.0, 20; target_rms_velocity=1.0)

# Now ready for time integration
prob = SemiGeostrophicProblem(fields, domain, ...)
step_until!(prob, 10.0)
```

## Complete Problem Setup
```julia
# One-line setup with spectral initial conditions
prob = create_problem_with_spectral_ics(512, 512, 20; 
                                       k₀=14.0, m=20, target_rms=1.0)

# Run simulation
step_until!(prob, 25.0)
```

## Analysis and Diagnostics
```julia
# Analyze initial spectrum
plot_initial_spectrum(fields, domain; k₀=14.0, m=20)

# Compute various diagnostics
rms_velocity = compute_rms_velocity(fields, domain)
kinetic_energy = compute_energy(fields.u, fields.v, domain)
buoyancy_variance = compute_buoyancy_variance(fields.bₛ, domain)
```

## Parameters from Paper
- **Grid**: 512×512×20 points
- **Domain**: L/L_D = 6 (non-dimensional)
- **Vertical**: 20 levels, exponentially spaced near surface
- **Spectrum**: k₀ = 14, m = 20 
- **Initial RMS velocity**: 1 m/s (typical oceanic front velocity)
"""

# Run example if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_initial_condition_example()
end# examples/initial_conditions_example.jl
# Example implementation of initial surface buoyancy perturbation
# Based on equation (20) from the paper:
#
#   b̃ₛ(k, t=0) = A * k^(m/4) / (k + k₀)^(m/2)
#
# Parameters from paper:
# - m = 20 (spectral peak sharpness)
# - k₀ = 14 (peak wavenumber, non-dimensional)
# - A tuned so initial RMS velocity = 1 m/s
# - Random phase assigned to each mode

using SSG
using Random
using Printf
using LinearAlgebra
using MPI
using PencilArrays
using PencilFFTs

"""
    initialize_spectral_buoyancy!(fields::Fields{T}, domain::Domain, 
                                 amplitude::T, k₀::T, m::Int; 
                                 target_rms_velocity::T=T(1.0),
                                 seed::Int=12345) where T

Initialize surface buoyancy with spectral perturbation following equation (20).

# Arguments
- `fields`: Fields structure containing buoyancy and other variables
- `domain`: Domain structure with grid and FFT information
- `amplitude`: Initial amplitude A (will be adjusted to match target RMS velocity)
- `k₀`: Peak wavenumber (non-dimensional, typically 14)
- `m`: Spectral slope parameter (typically 20)
- `target_rms_velocity`: Target RMS surface velocity in m/s (typically 1.0)
- `seed`: Random seed for reproducible phases

# Physics
The initial spectrum follows: b̃ₛ(k) = A * k^(m/4) / (k + k₀)^(m/2)
This creates a spectrum that peaks around k₀ and falls off at high wavenumbers.
"""
function initialize_spectral_buoyancy!(fields::Fields{T}, domain::Domain, 
                                      amplitude::T, k₀::T, m::Int;
                                      target_rms_velocity::T=T(1.0),
                                      seed::Int=12345) where T
    
    # Set random seed for reproducible results
    Random.seed!(seed)
    
    # Zero the buoyancy field initially
    zero_field!(fields.bₛ)
    
    # Transform to spectral space for initialization
    rfft!(domain, fields.bₛ, fields.bhat)
    
    # Get local spectral array and ranges
    bhat_local = fields.bhat.data
    local_ranges = local_range(fields.bhat.pencil)
    
    # Apply spectral initialization
    @inbounds for k_z in axes(bhat_local, 3)
        for (j_local, j_global) in enumerate(local_ranges[2])
            if j_global <= length(domain.ky)
                ky = domain.ky[j_global]
                for (i_local, i_global) in enumerate(local_ranges[1])
                    if i_global <= length(domain.kx)
                        kx = domain.kx[i_global]
                        
                        # Compute wavenumber magnitude
                        k_mag = sqrt(kx^2 + ky^2)
                        
                        # Skip k=0 mode (mean is zero)
                        if k_mag < 1e-14
                            bhat_local[i_local, j_local, k_z] = Complex{T}(0)
                            continue
                        end
                        
                        # Apply spectral shape from equation (20)
                        # b̃ₛ(k) = A * k^(m/4) / (k + k₀)^(m/2)
                        spectral_amplitude = amplitude * (k_mag^(m/4)) / ((k_mag + k₀)^(m/2))
                        
                        # Assign random phase (uniform in [0, 2π])
                        random_phase = 2π * rand(T)
                        
                        # Set complex amplitude with random phase
                        bhat_local[i_local, j_local, k_z] = Complex{T}(
                            spectral_amplitude * cos(random_phase),
                            spectral_amplitude * sin(random_phase)
                        )
                    else
                        bhat_local[i_local, j_local, k_z] = Complex{T}(0)
                    end
                end
            else
                @views bhat_local[:, j_local, k_z] .= Complex{T}(0)
            end
        end
    end
    
    # Apply dealiasing to prevent high-frequency contamination
    dealias!(domain, fields.bhat)
    
    # Transform back to physical space
    irfft!(domain, fields.bhat, fields.bₛ)
    
    # Ensure zero mean (required for periodic domains)
    remove_mean!(fields.bₛ)
    
    # Solve for initial streamfunction from buoyancy
    solve_monge_ampere_wrapper!(fields, domain)
    
    # Calculate actual RMS velocity
    actual_rms = compute_rms_velocity(fields, domain)
    
    # Adjust amplitude to match target RMS velocity
    if actual_rms > 1e-12
        scaling_factor = target_rms_velocity / actual_rms
        
        # Scale buoyancy field
        fields.bₛ.data .*= scaling_factor
        
        # Re-solve for consistent streamfunction and velocities
        solve_monge_ampere_wrapper!(fields, domain)
        
        # Verify final RMS velocity
        final_rms = compute_rms_velocity(fields, domain)
        
        if MPI.Comm_rank(domain.pc.comm) == 0
            @printf("Initial condition scaling:\n")
            @printf("  Target RMS velocity: %.3f m/s\n", target_rms_velocity)
            @printf("  Achieved RMS velocity: %.3f m/s\n", final_rms)
            @printf("  Scaling factor applied: %.6f\n", scaling_factor)
        end
    end
    
    return nothing
end

"""
    remove_mean!(field::PencilArray)

Remove the domain mean from a field (ensures zero mean for periodic domains).
"""
function remove_mean!(field::PencilArray{T, N}) where {T, N}
    field_local = field.data
    
    # Compute local sum
    local_sum = sum(field_local)
    local_count = length(field_local)
    
    # Global reduction
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, field.pencil.comm)
    global_count = MPI.Allreduce(local_count, MPI.SUM, field.pencil.comm)
    
    # Remove mean
    mean_value = global_sum / global_count
    field_local .-= mean_value
    
    return nothing
end

"""
    compute_rms_velocity(fields::Fields, domain::Domain) -> Real

Compute RMS velocity: sqrt(⟨u² + v²⟩) where ⟨⟩ denotes spatial average.
"""
function compute_rms_velocity(fields::Fields{T}, domain::Domain) where T
    u_local = fields.u.data
    v_local = fields.v.data
    
    # Compute local contribution to velocity squared
    local_vel_sq = zero(T)
    local_count = length(u_local)
    
    @inbounds for i in eachindex(u_local)
        local_vel_sq += u_local[i]^2 + v_local[i]^2
    end
    
    # Global reduction
    global_vel_sq = MPI.Allreduce(local_vel_sq, MPI.SUM, fields.u.pencil.comm)
    global_count = MPI.Allreduce(local_count, MPI.SUM, fields.u.pencil.comm)
    
    # RMS velocity
    mean_vel_sq = global_vel_sq / global_count
    rms_velocity = sqrt(mean_vel_sq)
    
    return rms_velocity
end

"""
    solve_monge_ampere_wrapper!(fields::Fields{T}, domain::Domain) where T

Wrapper to solve Monge-Ampère equation and compute velocities using existing functions.
"""
function solve_monge_ampere_wrapper!(fields::Fields{T}, domain::Domain) where T
    # Solve Monge-Ampère equation: det(D²φ) = b
    # This uses the existing implementation from poisson.jl
    converged = solve_monge_ampere_fields!(fields, domain; 
                                         tol=T(1e-10), 
                                         maxiter=20, 
                                         verbose=false)
    
    if !converged
        @warn "Monge-Ampère solver did not converge during initialization"
    end
    
    # Compute geostrophic velocities: u = -∂φ/∂y, v = ∂φ/∂x
    compute_geostrophic_velocities!(fields, domain)
    
    return nothing
end

"""
    plot_initial_spectrum(fields::Fields, domain::Domain; 
                         k₀::T=T(14.0), m::Int=20,
                         save_file="initial_spectrum.png") where T

Plot the initial buoyancy spectrum and compare with theoretical shape.
"""
function plot_initial_spectrum(fields::Fields{T}, domain::Domain; 
                              k₀::T=T(14.0), m::Int=20,
                              save_file="initial_spectrum.png") where T
    
    # Only root process creates the plot
    if MPI.Comm_rank(domain.pc.comm) != 0
        return nothing
    end
    
    # Transform to spectral space
    rfft!(domain, fields.bₛ, fields.bhat)
    
    # Gather spectral data to root (simplified version for surface field)
    bhat_gathered = gather_spectral_to_root_2d(fields.bhat)
    
    if bhat_gathered !== nothing
        # Compute radial spectrum
        k_bins, spectrum = compute_radial_spectrum_2d(bhat_gathered, domain)
        
        # Theoretical spectrum shape
        k_theory = k_bins
        A_theory = maximum(spectrum) / maximum(k_theory.^(m/4) ./ (k_theory .+ k₀).^(m/2))
        spectrum_theory = A_theory .* k_theory.^(m/4) ./ (k_theory .+ k₀).^(m/2)
        
        # Create plot (pseudo-code - requires plotting package)
        println("Initial spectrum analysis:")
        println("k₀ = $k₀, m = $m")
        println("Peak wavenumber: $(k_bins[argmax(spectrum)])")
        println("Peak theoretical: $(k₀)")
        
        # Save spectrum data
        open("initial_spectrum_data.txt", "w") do f
            println(f, "# k_magnitude  spectrum_actual  spectrum_theoretical")
            for i in eachindex(k_bins)
                println(f, "$(k_bins[i])  $(spectrum[i])  $(spectrum_theory[i])")
            end
        end
        
        println("Spectrum data saved to initial_spectrum_data.txt")
    end
    
    return nothing
end

"""
    gather_spectral_to_root_2d(field::PencilArray{Complex{T}, N}) -> Array{Complex{T}, 2}

Simplified gather function for 2D surface spectral data.
"""
function gather_spectral_to_root_2d(field::PencilArray{Complex{T}, N}) where {T, N}
    comm = field.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        # For simplicity, just use local data on root process
        # In practice, you'd implement proper MPI gathering
        local_data = field.data
        if N == 3
            return local_data[:, :, end]  # Use surface level
        else
            return local_data[:, :]
        end
    else
        return nothing
    end
end

"""
    compute_radial_spectrum_2d(bhat_2d, domain::Domain) -> (k_bins, spectrum)

Compute radially averaged spectrum from 2D spectral field.
"""
function compute_radial_spectrum_2d(bhat_2d::Array{Complex{T}, 2}, domain::Domain) where T
    nx, ny = size(bhat_2d)
    
    # Create wavenumber arrays (adjust for local vs global indexing)
    kx_local = length(domain.kx) >= nx ? domain.kx[1:nx] : domain.kx
    ky_local = length(domain.ky) >= ny ? domain.ky[1:ny] : domain.ky
    
    # Maximum wavenumber for binning
    k_max = min(length(kx_local)-1, length(ky_local)-1)
    k_bins = collect(1:k_max)
    spectrum = zeros(T, k_max)
    
    # Compute radial average
    for (j, ky_val) in enumerate(ky_local)
        for (i, kx_val) in enumerate(kx_local) 
            k_mag = sqrt(kx_val^2 + ky_val^2)
            k_bin = round(Int, k_mag)
            
            if 1 <= k_bin <= k_max
                # Power spectral density
                power = abs2(bhat_2d[i, j])
                spectrum[k_bin] += power
            end
        end
    end
    
    return k_bins, spectrum
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
    run_initial_condition_example()

Complete example of setting up and analyzing initial conditions.
"""
function run_initial_condition_example()
    # Initialize MPI
    MPI.Init()
    
    try
        # Problem setup matching paper parameters
        Nx, Ny, Nz = 512, 512, 20
        Lx = Ly = 6.0  # L/L_D = 6 from paper
        Lz = 1.0
        
        # Create domain with exponential vertical spacing
        domain = make_domain(Nx, Ny, Nz; Lx=Lx, Ly=Ly, Lz=Lz,
                           z_boundary=:dirichlet,
                           z_grid=:stretched,
                           stretch_params=(type=:exponential, β=2.0, surface_concentration=true))
        
        # Create problem
        prob = SemiGeostrophicProblem(domain; 
                                    scheme=RK3, 
                                    dt=0.001,
                                    adaptive_dt=true,
                                    enable_diagnostics=true)
        
        # Set initial conditions following equation (20)
        amplitude = 1.0  # Will be adjusted automatically
        k₀ = 14.0       # Peak wavenumber (non-dimensional)
        m = 20          # Spectral slope parameter
        target_rms = 1.0 # Target RMS velocity in m/s
        
        if MPI.Comm_rank(domain.pc.comm) == 0
            println("🌊 Setting up initial conditions from spectral perturbation")
            println("Parameters:")
            println("  Domain: $(Nx)×$(Ny)×$(Nz), L/L_D = $(Lx)")
            println("  k₀ = $(k₀) (peak wavenumber)")
            println("  m = $(m) (spectral slope)")
            println("  Target RMS velocity = $(target_rms) m/s")
            println()
        end
        
        # Initialize with spectral perturbation
        initialize_spectral_buoyancy!(prob, amplitude, k₀, m;
                                     target_rms_velocity=target_rms,
                                     seed=12345)
        
        # Analyze initial state
        if MPI.Comm_rank(domain.pc.comm) == 0
            println("✓ Initial conditions set successfully")
            
            # Compute diagnostics
            ke = compute_kinetic_energy(prob.fields, domain)
            b_variance = compute_buoyancy_variance(prob.fields.bₛ, domain)
            max_b = maximum(abs.(prob.fields.bₛ.data))
            max_u = maximum(abs.(prob.fields.u.data))
            max_v = maximum(abs.(prob.fields.v.data))
            
            println("\nInitial state diagnostics:")
            println("  Kinetic energy: $(ke)")
            println("  Buoyancy variance: $(b_variance)")
            println("  Max |buoyancy|: $(max_b)")
            println("  Max |u|: $(max_u) m/s")
            println("  Max |v|: $(max_v) m/s")
            
            # Verify RMS velocity
            final_rms = compute_rms_velocity(prob.fields, domain)
            println("  Final RMS velocity: $(final_rms) m/s")
            println()
        end
        
        # Plot initial spectrum
        plot_initial_spectrum(prob; k₀=k₀, m=m)
        
        # Save initial state
        save_simulation_state_full("initial_state_spectral.jld2", prob;
                                  save_spectral=true,
                                  save_metadata=true,
                                  compress=true)
        
        if MPI.Comm_rank(domain.pc.comm) == 0
            println("✓ Initial state saved to initial_state_spectral.jld2")
            println("\nReady for time integration!")
            println("Use: step_until!(prob, target_time) to advance simulation")
        end
        
        return prob
        
    finally
        MPI.Finalize()
    end
end

# ============================================================================
# ALTERNATIVE INITIALIZATION METHODS
# ============================================================================

"""
    initialize_taylor_green_perturbed!(prob::SemiGeostrophicProblem{T},
                                      tg_amplitude::T, spectral_amplitude::T,
                                      k₀::T, m::Int) where T

Initialize with Taylor-Green vortex plus spectral perturbation.
Useful for studying transition to turbulence.
"""
function initialize_taylor_green_perturbed!(prob::SemiGeostrophicProblem{T},
                                          tg_amplitude::T, 
                                          spectral_amplitude::T,
                                          k₀::T, m::Int) where T
    
    # First set Taylor-Green base flow
    set_initial_conditions!(prob, initialize_taylor_green!; amplitude=tg_amplitude)
    
    # Store base state
    b_base = copy(prob.fields.bₛ.data)
    
    # Add spectral perturbation
    initialize_spectral_buoyancy!(prob, spectral_amplitude, k₀, m; target_rms_velocity=T(0.1))
    
    # Combine base flow and perturbation
    prob.fields.bₛ.data .+= b_base
    
    # Re-solve for consistent state
    solve_monge_ampere_fields!(prob.fields, prob.domain; tol=T(1e-10))
    compute_geostrophic_velocities!(prob.fields, prob.domain)
    
    return nothing
end

"""
    initialize_realistic_ocean_front!(prob::SemiGeostrophicProblem{T};
                                     front_width::T=T(0.1),
                                     front_strength::T=T(1.0),
                                     noise_level::T=T(0.05)) where T

Initialize with realistic ocean front plus small-scale perturbations.
"""
function initialize_realistic_ocean_front!(prob::SemiGeostrophicProblem{T};
                                          front_width::T=T(0.1),
                                          front_strength::T=T(1.0),
                                          noise_level::T=T(0.05)) where T
    
    domain = prob.domain
    fields = prob.fields
    
    # Get local grid points
    local_ranges = local_range(fields.bₛ.pencil)
    b_local = fields.bₛ.data
    
    # Create front structure
    @inbounds for (j_local, j_global) in enumerate(local_ranges[2])
        y = domain.y[j_global]
        y_normalized = 2 * (y / domain.Ly) - 1  # y ∈ [-1, 1]
        
        for (i_local, i_global) in enumerate(local_ranges[1])
            x = domain.x[i_global]
            
            # Tanh front profile
            front_profile = front_strength * tanh(y_normalized / front_width)
            
            # Add small-scale noise
            noise = noise_level * front_strength * (2 * rand(T) - 1)
            
            b_local[i_local, j_local] = front_profile + noise
        end
    end
    
    # Remove mean and solve for consistent state
    remove_mean!(fields.bₛ)
    solve_monge_ampere_fields!(fields, domain; tol=T(1e-10))
    compute_geostrophic_velocities!(fields, domain)
    
    return nothing
end

# Run example if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_initial_condition_example()
end