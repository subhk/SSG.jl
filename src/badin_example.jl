# examples/single_process_example.jl
# Clean single-process SSG simulation without MPI complications
# Usage: julia single_process_example.jl

using Random
using Printf
using LinearAlgebra
using Statistics
using JLD2

# Import packages individually to avoid conflicts
import PencilArrays: PencilArray, size_local, range_local
import PencilFFTs
import MPI

# # Only load SSG module once
# if !isdefined(Main, :SSG)
#     include("../src/SSG.jl")
#     using .SSG
# end

include("SSG.jl")
using .SSG

"""
Simple single-process simulation wrapper
"""
struct SingleProcessSimulation{T}
    domain::Domain
    fields::Fields{T}
    
    # Simulation parameters
    dt::T
    current_time::T
    step_count::Int
    
    # Statistics tracking
    energy_history::Vector{T}
    time_history::Vector{T}
    
    function SingleProcessSimulation{T}(Nx, Ny, Nz; Lx=2π, Ly=2π, Lz=1.0) where T
        # Create domain with single-process communicator
        domain = Domain(Nx, Ny, Nz; Lx=Lx, Ly=Ly, Lz=Lz, comm=MPI.COMM_SELF)
        fields = allocate_fields(domain)
        
        new{T}(domain, fields, T(0.001), T(0.0), 0, T[], T[])
    end
end

"""
Initialize buoyancy field with simple pattern
"""
function initialize_buoyancy!(sim::SingleProcessSimulation{T}, 
                             amplitude::T=T(0.1)) where T
    
    println("🌊 Initializing buoyancy field...")
    
    # Get direct access to field data (single process)
    b_data = sim.fields.bₛ.data
    
    # Initialize with Taylor-Green-like pattern
    for k in axes(b_data, 3)
        z = sim.domain.z[k]
        for j in axes(b_data, 2)
            y = sim.domain.y[j]
            for i in axes(b_data, 1)
                x = sim.domain.x[i]
                
                # Multi-scale pattern
                val = amplitude * (
                    sin(2π*x/sim.domain.Lx) * cos(2π*y/sim.domain.Ly) +
                    0.5 * sin(4π*x/sim.domain.Lx) * sin(4π*y/sim.domain.Ly) +
                    0.25 * cos(6π*x/sim.domain.Lx) * cos(6π*y/sim.domain.Ly)
                ) * exp(-2*abs(z - sim.domain.Lz/2))
                
                b_data[i, j, k] = val
            end
        end
    end
    
    # Remove mean
    mean_b = mean(b_data)
    b_data .-= mean_b
    
    println("✓ Buoyancy initialized (mean removed: $(round(mean_b, digits=6)))")
    return nothing
end

"""
Solve for streamfunction using simple Poisson equation: ∇²φ = b
"""
function solve_streamfunction!(sim::SingleProcessSimulation{T}) where T
    
    # Transform buoyancy to spectral space
    rfft!(sim.domain, sim.fields.bₛ, sim.fields.bshat)
    
    # Solve Poisson equation in spectral space
    bshat_data = sim.fields.bshat.data
    φhat_data = sim.fields.φhat.data
    
    for k in axes(bshat_data, 3)
        for j in axes(bshat_data, 2)
            ky = j <= length(sim.domain.ky) ? sim.domain.ky[j] : 0.0
            for i in axes(bshat_data, 1)
                kx = i <= length(sim.domain.kx) ? sim.domain.kx[i] : 0.0
                
                k2 = kx^2 + ky^2
                if k2 > 1e-14
                    φhat_data[i, j, k] = -bshat_data[i, j, k] / k2
                else
                    φhat_data[i, j, k] = 0.0
                end
            end
        end
    end
    
    # Transform back to physical space
    irfft!(sim.domain, sim.fields.φhat, sim.fields.φ)
    
    return nothing
end

"""
Compute geostrophic velocities: u = -∂φ/∂y, v = ∂φ/∂x
"""
function compute_velocities!(sim::SingleProcessSimulation{T}) where T
    
    # Transform streamfunction to spectral space
    rfft!(sim.domain, sim.fields.φ, sim.fields.φhat)
    
    φhat_data = sim.fields.φhat.data
    tmpc_data = sim.fields.tmpc.data
    
    # Compute u = -∂φ/∂y
    for k in axes(φhat_data, 3)
        for j in axes(φhat_data, 2)
            ky = j <= length(sim.domain.ky) ? sim.domain.ky[j] : 0.0
            for i in axes(φhat_data, 1)
                tmpc_data[i, j, k] = im * ky * φhat_data[i, j, k]
            end
        end
    end
    
    irfft!(sim.domain, sim.fields.tmpc, sim.fields.u)
    sim.fields.u.data .*= -1  # Apply negative sign
    
    # Compute v = ∂φ/∂x
    for k in axes(φhat_data, 3)
        for j in axes(φhat_data, 2)
            for i in axes(φhat_data, 1)
                kx = i <= length(sim.domain.kx) ? sim.domain.kx[i] : 0.0
                tmpc_data[i, j, k] = im * kx * φhat_data[i, j, k]
            end
        end
    end
    
    irfft!(sim.domain, sim.fields.tmpc, sim.fields.v)
    
    return nothing
end

"""
Compute Jacobian: J(φ, b) = ∂φ/∂x ∂b/∂y - ∂φ/∂y ∂b/∂x
"""
function compute_jacobian!(result, φ, b, sim::SingleProcessSimulation{T}) where T
    
    # Transform fields to spectral space
    rfft!(sim.domain, φ, sim.fields.φhat)
    rfft!(sim.domain, b, sim.fields.bshat)
    
    φhat_data = sim.fields.φhat.data
    bshat_data = sim.fields.bshat.data
    tmpc_data = sim.fields.tmpc.data
    tmpc2_data = sim.fields.tmpc2.data
    
    # Compute ∂φ/∂x
    for k in axes(φhat_data, 3)
        for j in axes(φhat_data, 2)
            for i in axes(φhat_data, 1)
                kx = i <= length(sim.domain.kx) ? sim.domain.kx[i] : 0.0
                tmpc_data[i, j, k] = im * kx * φhat_data[i, j, k]
            end
        end
    end
    irfft!(sim.domain, sim.fields.tmpc, sim.fields.tmp)  # ∂φ/∂x in physical space
    
    # Compute ∂φ/∂y
    for k in axes(φhat_data, 3)
        for j in axes(φhat_data, 2)
            ky = j <= length(sim.domain.ky) ? sim.domain.ky[j] : 0.0
            for i in axes(φhat_data, 1)
                tmpc_data[i, j, k] = im * ky * φhat_data[i, j, k]
            end
        end
    end
    irfft!(sim.domain, sim.fields.tmpc, sim.fields.tmp2)  # ∂φ/∂y in physical space
    
    # Compute ∂b/∂x
    for k in axes(bshat_data, 3)
        for j in axes(bshat_data, 2)
            for i in axes(bshat_data, 1)
                kx = i <= length(sim.domain.kx) ? sim.domain.kx[i] : 0.0
                tmpc_data[i, j, k] = im * kx * bshat_data[i, j, k]
            end
        end
    end
    irfft!(sim.domain, sim.fields.tmpc, sim.fields.tmp3)  # ∂b/∂x in physical space
    
    # Compute ∂b/∂y
    for k in axes(bshat_data, 3)
        for j in axes(bshat_data, 2)
            ky = j <= length(sim.domain.ky) ? sim.domain.ky[j] : 0.0
            for i in axes(bshat_data, 1)
                tmpc2_data[i, j, k] = im * ky * bshat_data[i, j, k]
            end
        end
    end
    irfft!(sim.domain, sim.fields.tmpc2, result)  # ∂b/∂y in physical space
    
    # Compute Jacobian: J = ∂φ/∂x * ∂b/∂y - ∂φ/∂y * ∂b/∂x
    result_data = result.data
    φx_data = sim.fields.tmp.data    # ∂φ/∂x
    φy_data = sim.fields.tmp2.data   # ∂φ/∂y
    bx_data = sim.fields.tmp3.data   # ∂b/∂x
    
    for i in eachindex(result_data)
        result_data[i] = φx_data[i] * result_data[i] - φy_data[i] * bx_data[i]
    end
    
    return nothing
end

"""
Take one time step using forward Euler
"""
function time_step!(sim::SingleProcessSimulation{T}) where T
    
    # Compute tendency: ∂b/∂t = -J(φ, b)
    compute_jacobian!(sim.fields.R, sim.fields.φ, sim.fields.bₛ, sim)
    
    # Forward Euler step: b^{n+1} = b^n - dt * J(φ, b)
    sim.fields.bₛ.data .-= sim.dt .* sim.fields.R.data
    
    # Solve for new streamfunction and velocities
    solve_streamfunction!(sim)
    compute_velocities!(sim)
    
    # Update time
    sim.current_time += sim.dt
    sim.step_count += 1
    
    return nothing
end

"""
Compute kinetic energy
"""
function compute_energy(sim::SingleProcessSimulation{T}) where T
    u_data = sim.fields.u.data
    v_data = sim.fields.v.data
    return 0.5 * mean(u_data.^2 + v_data.^2)
end

"""
Run simulation
"""
function run_simulation!(sim::SingleProcessSimulation{T}, 
                        final_time::T; 
                        save_interval::T=T(0.5),
                        output_interval::Int=50) where T
    
    println("🚀 Starting simulation...")
    println("   Final time: $final_time")
    println("   Time step: $(sim.dt)")
    println("   Save interval: $save_interval")
    
    next_save_time = save_interval
    save_counter = 0
    
    # Save initial state
    save_state(sim, "initial_state.jld2")
    
    while sim.current_time < final_time
        # Take time step
        time_step!(sim)
        
        # Record diagnostics
        if sim.step_count % 10 == 0
            energy = compute_energy(sim)
            push!(sim.energy_history, energy)
            push!(sim.time_history, sim.current_time)
        end
        
        # Output progress
        if sim.step_count % output_interval == 0
            energy = compute_energy(sim)
            b_range = extrema(sim.fields.bₛ.data)
            u_max = maximum(abs.(sim.fields.u.data))
            v_max = maximum(abs.(sim.fields.v.data))
            
            @printf("Step %4d: t=%.3f, E=%.2e, b∈[%.2e,%.2e], |u|_max=%.2e, |v|_max=%.2e\n",
                   sim.step_count, sim.current_time, energy, b_range..., u_max, v_max)
        end
        
        # Save state
        if sim.current_time >= next_save_time - sim.dt/2
            save_counter += 1
            filename = "state_$(lpad(save_counter, 3, '0')).jld2"
            save_state(sim, filename)
            println(" Saved: $filename (t=$(round(sim.current_time, digits=3)))")
            next_save_time += save_interval
        end
        
        # Check for instabilities
        if !all(isfinite.(sim.fields.bₛ.data))
            println("❌ NaN/Inf detected! Stopping simulation.")
            break
        end
        
        max_vel = max(maximum(abs.(sim.fields.u.data)), maximum(abs.(sim.fields.v.data)))
        if max_vel > 10.0
            println("⚠️  Large velocities detected ($(max_vel)). Consider reducing time step.")
        end
    end
    
    println("✅ Simulation completed!")
    println("   Final time: $(round(sim.current_time, digits=3))")
    println("   Total steps: $(sim.step_count)")
    
    return sim
end

"""
Save simulation state
"""
function save_state(sim::SingleProcessSimulation, filename::String)
    try
        jldopen(filename, "w") do file
            file["time"] = sim.current_time
            file["step"] = sim.step_count
            file["dt"] = sim.dt
            
            # Fields
            file["buoyancy"] = Array(sim.fields.bₛ.data)
            file["streamfunction"] = Array(sim.fields.φ.data)
            file["u_velocity"] = Array(sim.fields.u.data)
            file["v_velocity"] = Array(sim.fields.v.data)
            
            # Grid info
            file["grid"] = (
                Nx=sim.domain.Nx, Ny=sim.domain.Ny, Nz=sim.domain.Nz,
                Lx=sim.domain.Lx, Ly=sim.domain.Ly, Lz=sim.domain.Lz,
                x=sim.domain.x, y=sim.domain.y, z=sim.domain.z
            )
            
            # Diagnostics
            if !isempty(sim.energy_history)
                file["energy_history"] = sim.energy_history
                file["time_history"] = sim.time_history
            end
            
            # Statistics
            energy = compute_energy(sim)
            b_stats = extrema(sim.fields.bₛ.data)
            file["diagnostics"] = (
                energy=energy,
                buoyancy_min=b_stats[1],
                buoyancy_max=b_stats[2],
                rms_velocity=sqrt(mean(sim.fields.u.data.^2 + sim.fields.v.data.^2))
            )
        end
    catch e
        println("❌ Failed to save $filename: $e")
    end
end

"""
Main function
"""
function main()
    println("🌊 SSG Single-Process Simulation")
    println("=" ^ 40)
    
    # Problem setup
    T = Float64
    Nx, Ny, Nz = 128, 128, 8
    Lx, Ly = 4.0, 4.0
    
    println("Grid: $(Nx)×$(Ny)×$(Nz)")
    println("Domain: [0,$(Lx)] × [0,$(Ly)] × [0,1]")
    
    # Create simulation
    sim = SingleProcessSimulation{T}(Nx, Ny, Nz; Lx=Lx, Ly=Ly, Lz=1.0)
    
    # Initialize
    initialize_buoyancy!(sim, T(0.1))
    solve_streamfunction!(sim)
    compute_velocities!(sim)
    
    # Initial diagnostics
    energy = compute_energy(sim)
    b_range = extrema(sim.fields.bₛ.data)
    println("✓ Initial energy: $(round(energy, digits=6))")
    println("✓ Buoyancy range: [$(round(b_range[1], digits=4)), $(round(b_range[2], digits=4))]")
    
    # Run simulation
    run_simulation!(sim, T(5.0); save_interval=T(1.0), output_interval=100)
    
    return sim
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Make sure MPI is initialized for single process
    if !MPI.Initialized()
        MPI.Init()
        atexit(() -> MPI.Finalized() || MPI.Finalize())
    end
    
    main()
end