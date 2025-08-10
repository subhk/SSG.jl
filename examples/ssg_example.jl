# Test example based on the SSG JFM paper
# Surface Semi-Geostrophic equations simulation with realistic oceanographic parameters
# 
# This test reproduces key elements from the JFM paper:
# - Spectral initial conditions: b̃ₛ(k, t=0) = A * k^(m/4) / (k + k₀)^(m/2)
# - Ocean-like domain with vertical stretching
# - Realistic time integration and filtering
# - Energy and enstrophy diagnostics matching the paper
#
# Usage: 
#   julia ssg_example.jl                    # Single process
#   mpirun -np 4 julia ssg_example.jl       # Multi-process

using MPI
using PencilArrays
using PencilFFTs
using Random
using Printf
using LinearAlgebra

# Add the source directory to the load path
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))

using SSG

"""
    create_jfm_domain(Nx, Ny, Nz; Lx=100e3, Ly=100e3, Lz=1000.0, comm=MPI.COMM_WORLD)

Create domain matching JFM paper specifications:
- Horizontal domain: 100 km × 100 km (typical mesoscale)  
- Vertical domain: 1000 m depth with surface concentration
- Exponential grid stretching for surface boundary layer resolution
"""
function create_jfm_domain(Nx::Int, Ny::Int, Nz::Int; 
                          Lx=100e3, Ly=100e3, Lz=1000.0, 
                          comm=MPI.COMM_WORLD)
    
    rank = MPI.Comm_rank(comm)
    
    # Ocean-like domain with exponential surface stretching
    domain = make_domain(Nx, Ny, Nz; 
                        Lx=Lx, Ly=Ly, Lz=Lz,
                        z_boundary=:dirichlet,
                        z_grid=:stretched,
                        stretch_params=(type=:exponential, β=2.5, surface_concentration=true),
                        comm=comm)
    
    if rank == 0
        println("🌊 Domain Configuration:")
        println("  Horizontal: $(Lx/1e3) × $(Ly/1e3) km")
        println("  Vertical: $(Lz) m with surface stretching")
        println("  Resolution: $(Nx)×$(Ny)×$(Nz)")
        println("  Grid spacing: Δx=$(Lx/Nx/1e3) km, surface Δz≈$(domain.dz[end]) m")
    end
    
    return domain
end

"""
    initialize_jfm_spectral_buoyancy!(fields, domain; A=1e-6, k₀=14.0, m=20, seed=12345)

Initialize surface buoyancy following JFM paper equation:
b̃ₛ(k, t=0) = A * k^(m/4) / (k + k₀)^(m/2)

This creates a realistic mesoscale turbulent initial condition with:
- Power law spectrum at low wavenumbers
- Exponential decay at high wavenumbers  
- Realistic amplitude scaling for ocean conditions
"""
function initialize_jfm_spectral_buoyancy!(fields::Fields{T}, domain::Domain;
                                          A::T=T(1e-6),      # Amplitude (m/s²)
                                          k₀::T=T(14.0),     # Transition wavenumber
                                          m::Int=20,         # Spectral slope parameter
                                          seed::Int=12345) where T
    
    comm = fields.bₛ.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Transform to spectral space
    zero_field!(fields.bₛ)
    rfft_2d!(domain, fields.bₛ, fields.bshat)
    
    # Initialize spectral field with synchronized random phases
    bhat_local = fields.bshat.data
    range_locals = range_local(fields.bshat.pencil)
    
    total_energy = zero(T)
    n_modes = 0
    
    @inbounds for (j_local, j_global) in enumerate(range_locals[2])
        j_global > length(domain.ky) && continue
        ky = domain.ky[j_global]
        
        for (i_local, i_global) in enumerate(range_locals[1])
            i_global > length(domain.kx) && continue
            kx = domain.kx[i_global]
            
            k_mag = sqrt(kx^2 + ky^2)
            k_mag < 1e-14 && continue  # Skip k=0 mode
            
            # JFM paper spectrum: A * k^(m/4) / (k + k₀)^(m/2)
            spec_amplitude = A * (k_mag^(m/4)) / ((k_mag + k₀)^(m/2))
            
            # Synchronized random phase across all MPI processes
            Random.seed!(seed + 1000*i_global + j_global)
            phase = 2π * rand(T)
            
            # Set spectral coefficient
            bhat_local[i_local, j_local] = Complex{T}(
                spec_amplitude * cos(phase), 
                spec_amplitude * sin(phase)
            )
            
            # Track energy for diagnostics
            total_energy += abs2(bhat_local[i_local, j_local])
            n_modes += 1
        end
    end
    
    # Apply dealiasing and transform back
    dealias_2d!(domain, fields.bshat)
    irfft_2d!(domain, fields.bshat, fields.bₛ)
    
    # Remove mean (critical for periodic domains)
    remove_mean_2d!(fields.bₛ)
    
    # Global energy statistics
    total_energy_global = MPI.Allreduce(total_energy, MPI.SUM, comm)
    n_modes_global = MPI.Allreduce(n_modes, MPI.SUM, comm)
    
    if rank == 0
        avg_mode_energy = n_modes_global > 0 ? total_energy_global / n_modes_global : zero(T)
        println("  Spectral initialization complete:")
        println("    Total spectral energy: $(round(total_energy_global, digits=8))")
        println("    Active modes: $n_modes_global")
        println("    Average mode energy: $(round(avg_mode_energy, digits=10))")
    end
    
    return nothing
end

"""
    remove_mean_2d!(field::PencilArray{T, 2}) where T

Remove spatial mean from 2D field with proper MPI handling.
"""
function remove_mean_2d!(field::PencilArray{T, 2}) where T
    local_sum = sum(field.data)
    local_count = length(field.data)
    
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, field.pencil.comm)
    global_count = MPI.Allreduce(local_count, MPI.SUM, field.pencil.comm)
    
    mean_value = global_sum / global_count
    field.data .-= mean_value
    
    return nothing
end

"""
    solve_for_streamfunction_and_velocities!(fields, domain; verbose=false)

Solve Monge-Ampère equation and compute geostrophic velocities.
"""
function solve_for_streamfunction_and_velocities!(fields::Fields{T}, domain::Domain; 
                                                 verbose::Bool=false) where T
    comm = fields.bₛ.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Solve surface semi-geostrophic equation: det(D²φ) = b
    start_time = MPI.Wtime()
    
    converged = solve_monge_ampere_multigrid!(fields.φₛ, fields.bₛ, domain;
                                            tol=T(1e-10), maxiter=50, 
                                            verbose=(rank == 0 && verbose))
    
    solve_time = MPI.Wtime() - start_time
    
    if rank == 0 && verbose
        println("  Monge-Ampère solver: $(converged ? "✓" : "❌") converged ($(round(solve_time, digits=2))s)")
    end
    
    # Compute surface geostrophic velocities: u = -∂φ/∂y, v = ∂φ/∂x
    compute_surface_geostrophic_velocities!(fields, domain)
    
    return converged
end

"""
    compute_jfm_diagnostics(fields, domain) -> NamedTuple

Compute energy and enstrophy diagnostics following JFM paper methodology.
"""
function compute_jfm_diagnostics(fields::Fields{T}, domain::Domain) where T
    comm = fields.bₛ.pencil.comm
    
    # Surface kinetic energy: KE = (1/2) ∫(u² + v²) dA
    local_ke = 0.5 * sum(fields.u.data[:,:,end].^2 + fields.v.data[:,:,end].^2)
    global_ke = MPI.Allreduce(local_ke, MPI.SUM, comm)
    
    # Surface potential energy: PE = (1/2) ∫b² dA  
    local_pe = 0.5 * sum(fields.bₛ.data.^2)
    global_pe = MPI.Allreduce(local_pe, MPI.SUM, comm)
    
    # Grid cell area for proper normalization
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    cell_area = dx * dy
    
    # Normalize by total domain area
    total_area = domain.Lx * domain.Ly
    ke_density = global_ke * cell_area / total_area  
    pe_density = global_pe * cell_area / total_area
    
    # Surface enstrophy: Z = (1/2) ∫ζ² dA where ζ = ∂v/∂x - ∂u/∂y
    # Computed spectrally for accuracy
    rfft_2d!(domain, fields.u[:,:,end], fields.tmpc_2d)
    ddx_2d!(domain, fields.tmpc_2d, fields.tmpc2_2d)
    irfft_2d!(domain, fields.tmpc2_2d, fields.tmp)
    dudy = copy(fields.tmp.data)
    
    rfft_2d!(domain, fields.v[:,:,end], fields.tmpc_2d)  
    ddy_2d!(domain, fields.tmpc_2d, fields.tmpc2_2d)
    irfft_2d!(domain, fields.tmpc2_2d, fields.tmp)
    dvdx = copy(fields.tmp.data)
    
    vorticity = dvdx - dudy
    local_enstrophy = 0.5 * sum(vorticity.^2)
    global_enstrophy = MPI.Allreduce(local_enstrophy, MPI.SUM, comm)
    enstrophy_density = global_enstrophy * cell_area / total_area
    
    # Field statistics
    b_min = MPI.Allreduce(minimum(fields.bₛ.data), MPI.MIN, comm)
    b_max = MPI.Allreduce(maximum(fields.bₛ.data), MPI.MAX, comm)
    u_max = MPI.Allreduce(maximum(abs, fields.u.data), MPI.MAX, comm)
    v_max = MPI.Allreduce(maximum(abs, fields.v.data), MPI.MAX, comm)
    
    return (
        kinetic_energy = ke_density,
        potential_energy = pe_density, 
        total_energy = ke_density + pe_density,
        enstrophy = enstrophy_density,
        buoyancy_range = (b_min, b_max),
        velocity_max = (u_max, v_max),
        rossby_number = max(u_max, v_max) * domain.Lx / (2π * 1e-4 * domain.Lx)  # Rough estimate
    )
end

"""
    run_jfm_test_simulation(Nx=64, Ny=64, Nz=16; t_end=10.0, dt=0.1)

Run complete test simulation matching JFM paper methodology.
"""
function run_jfm_test_simulation(Nx::Int=64, Ny::Int=64, Nz::Int=16; 
                                t_end::Real=10.0, dt::Real=0.1)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    try
        if rank == 0
            println("="^60)
            println("SSG JFM Paper Test Simulation")
            println("="^60)
            println("Grid: $(Nx)×$(Ny)×$(Nz) on $nprocs processes")
            println("Domain: 100×100 km, 1000m depth")
            println("Time integration: t=0 → $t_end, dt=$dt")
        end
        
        # Create oceanographic domain
        domain = create_jfm_domain(Nx, Ny, Nz)
        fields = allocate_fields(domain)
        
        # Initialize with JFM spectral buoyancy  
        initialize_jfm_spectral_buoyancy!(fields, domain; 
                                        A=1e-6, k₀=14.0, m=20, seed=12345)
        
        # Solve for initial streamfunction and velocities
        converged = solve_for_streamfunction_and_velocities!(fields, domain; verbose=true)
        
        if !converged && rank == 0
            @warn "Initial Monge-Ampère solver did not converge"
        end
        
        # Initial diagnostics
        initial_diag = compute_jfm_diagnostics(fields, domain)
        
        if rank == 0
            println("\n📊 Initial Conditions:")
            println("  Surface KE: $(round(initial_diag.kinetic_energy, digits=8)) m²/s²")
            println("  Surface PE: $(round(initial_diag.potential_energy, digits=8)) m²/s⁴") 
            println("  Total Energy: $(round(initial_diag.total_energy, digits=8))")
            println("  Enstrophy: $(round(initial_diag.enstrophy, digits=8)) s⁻²")
            println("  Buoyancy range: [$(round(initial_diag.buoyancy_range[1], digits=8)), $(round(initial_diag.buoyancy_range[2], digits=8))]")
            println("  Max velocities: u=$(round(initial_diag.velocity_max[1], digits=4)), v=$(round(initial_diag.velocity_max[2], digits=4)) m/s")
        end
        
        # Set up time integration
        params = TimeParams{Float64}(dt; 
                                   scheme=:RK3, 
                                   filter_freq=10, 
                                   filter_strength=0.05,
                                   adaptive_dt=false)
        
        state = TimeState{Float64, typeof(fields.bₛ)}(0.0, fields)
        
        # Time integration loop
        n_steps = Int(ceil(t_end / dt))
        save_freq = max(1, n_steps ÷ 10)  # Save 10 times during simulation
        
        if rank == 0
            println("\n  Starting time integration ($n_steps steps)...")
        end
        
        start_time = MPI.Wtime()
        
        for step = 1:n_steps
            # Take time step
            timestep_rk3!(fields, domain, params, state)
            state.t += dt
            
            # Periodic diagnostics and output
            if step % save_freq == 0 || step == n_steps
                current_diag = compute_jfm_diagnostics(fields, domain)
                elapsed = MPI.Wtime() - start_time
                
                if rank == 0
                    energy_conservation = abs(current_diag.total_energy - initial_diag.total_energy) / initial_diag.total_energy
                    
                    println("Step $step/$(n_steps), t=$(round(state.t, digits=2)):")
                    println("  KE: $(round(current_diag.kinetic_energy, digits=8))")
                    println("  PE: $(round(current_diag.potential_energy, digits=8))")  
                    println("  Enstrophy: $(round(current_diag.enstrophy, digits=8))")
                    println("  Energy conservation: $(round(energy_conservation*100, digits=4))%")
                    println("  Elapsed: $(round(elapsed, digits=1))s")
                    
                    # Check for numerical instability
                    if !isfinite(current_diag.total_energy) || current_diag.total_energy > 100 * initial_diag.total_energy
                        println("❌ Numerical instability detected - stopping simulation")
                        break
                    end
                end
            end
        end
        
        # Final diagnostics
        final_diag = compute_jfm_diagnostics(fields, domain)
        total_time = MPI.Wtime() - start_time
        
        if rank == 0
            println("\n"*"="^60)
            println("SIMULATION COMPLETE")
            println("="^60)
            println("Runtime: $(round(total_time, digits=1))s ($(round(total_time/n_steps, digits=3))s/step)")
            
            # Energy and enstrophy evolution
            ke_change = (final_diag.kinetic_energy - initial_diag.kinetic_energy) / initial_diag.kinetic_energy
            pe_change = (final_diag.potential_energy - initial_diag.potential_energy) / initial_diag.potential_energy
            enstrophy_change = (final_diag.enstrophy - initial_diag.enstrophy) / initial_diag.enstrophy
            energy_conservation = abs(final_diag.total_energy - initial_diag.total_energy) / initial_diag.total_energy
            
            println("\n Evolution Summary:")
            println("  KE change: $(round(ke_change*100, digits=2))%")
            println("  PE change: $(round(pe_change*100, digits=2))%") 
            println("  Enstrophy change: $(round(enstrophy_change*100, digits=2))%")
            println("  Energy conservation error: $(round(energy_conservation*100, digits=6))%")
            
            # Success criteria
            success = energy_conservation < 0.01 && isfinite(final_diag.total_energy)
            
            if success
                println("\n✅ Test PASSED - Energy conserved and simulation stable")
            else
                println("\n❌ Test FAILED - Energy conservation or stability issues")
            end
            
            return success
        end
        
    catch e
        if rank == 0
            println("❌ Error during simulation: $e")
            rethrow(e)
        end
        return false
    finally
        MPI.Barrier(comm)
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    end
    
    return true
end

"""
    test_jfm_small_scale()

Quick test with small grid for development and CI.
"""
function test_jfm_small_scale()
    return run_jfm_test_simulation(32, 32, 8; t_end=2.0, dt=0.05)
end

"""
    test_jfm_medium_scale() 

Medium resolution test for validation.
"""
function test_jfm_medium_scale()
    return run_jfm_test_simulation(64, 64, 16; t_end=5.0, dt=0.1)  
end

"""
    test_jfm_production_scale()

High resolution test matching JFM paper parameters.
"""
function test_jfm_production_scale()
    return run_jfm_test_simulation(128, 128, 32; t_end=25.0, dt=0.01)
end

# Run tests based on command line arguments or direct execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments for different test scales
    test_scale = length(ARGS) > 0 ? ARGS[1] : "small"
    
    println("Running JFM SSG test at '$test_scale' scale...")
    
    success = if test_scale == "small"
        test_jfm_small_scale()
    elseif test_scale == "medium"  
        test_jfm_medium_scale()
    elseif test_scale == "production" || test_scale == "full"
        test_jfm_production_scale()
    else
        println("Unknown test scale: $test_scale. Use 'small', 'medium', or 'production'")
        false
    end
    
    exit(success ? 0 : 1)
end