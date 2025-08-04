# examples/initial_conditions_example.jl
# Surface Semi-Geostrophic initial conditions from paper equation (20):
#   b̃ₛ(k, t=0) = A * k^(m/4) / (k + k₀)^(m/2)
# 
# Usage: mpirun -np 4 julia initial_conditions_example.jl

using Random
using Printf
using LinearAlgebra
using MPI
using PencilArrays
using PencilFFTs

include("../src/SSG.jl")
using ..SSG

"""
    initialize_spectral_buoyancy!(fields, domain, amplitude, k₀, m; 
                                target_rms_velocity=1.0, seed=12345)

Initialize surface buoyancy with spectral perturbation. MPI-parallel 
with synchronized random phases.
"""
function initialize_spectral_buoyancy!(fields::Fields{T}, domain::Domain, 
                                      amplitude::T, k₀::T, m::Int;
                                      target_rms_velocity::T=T(1.0),
                                      seed::Int=12345) where T
    
    comm = fields.bₛ.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Zero field and transform to spectral space
    zero_field!(fields.bₛ)
    rfft!(domain, fields.bₛ, fields.bhat)
    
    # Initialize spectral field on each process's local domain
    bhat_local = fields.bhat.data
    local_ranges = local_range(fields.bhat.pencil)
    
    @inbounds for k_z in axes(bhat_local, 3)
        for (j_local, j_global) in enumerate(local_ranges[2])
            j_global > length(domain.ky) && continue
            ky = domain.ky[j_global]
            
            for (i_local, i_global) in enumerate(local_ranges[1])
                i_global > length(domain.kx) && continue
                kx = domain.kx[i_global]
                
                k_mag = sqrt(kx^2 + ky^2)
                k_mag < 1e-14 && continue  # Skip k=0
                
                # Equation (20): spectral amplitude
                spec_amp = amplitude * (k_mag^(m/4)) / ((k_mag + k₀)^(m/2))
                
                # Synchronized random phase across processes
                Random.seed!(seed + 1000*i_global + j_global)
                phase = 2π * rand(T)
                
                bhat_local[i_local, j_local, k_z] = Complex{T}(
                    spec_amp * cos(phase), spec_amp * sin(phase))
            end
        end
    end
    
    # Transform back and solve for streamfunction
    dealias!(domain, fields.bhat)
    irfft!(domain, fields.bhat, fields.bₛ)
    remove_mean!(fields.bₛ)
    
    solve_monge_ampere_fields!(fields, domain; tol=T(1e-10), maxiter=20, verbose=false)
    compute_geostrophic_velocities!(fields, domain)
    
    # Scale to target RMS velocity
    actual_rms = compute_rms_velocity(fields, domain)
    if actual_rms > 1e-12
        scaling = target_rms_velocity / actual_rms
        fields.bₛ.data .*= scaling
        solve_monge_ampere_fields!(fields, domain; tol=T(1e-10), maxiter=20, verbose=false) 
        compute_geostrophic_velocities!(fields, domain)
        
        rank == 0 && @printf("Scaled RMS velocity: %.3f → %.3f m/s (factor: %.3f)\n", 
                             actual_rms, compute_rms_velocity(fields, domain), scaling)
    end
    
    MPI.Barrier(comm)
    return nothing
end

"""Remove domain mean from field"""
function remove_mean!(field::PencilArray{T, N}) where {T, N}
    local_sum = sum(field.data)
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, field.pencil.comm)
    global_count = MPI.Allreduce(length(field.data), MPI.SUM, field.pencil.comm)
    field.data .-= global_sum / global_count
    return nothing
end

"""Compute RMS velocity with MPI reduction"""
function compute_rms_velocity(fields::Fields{T}, domain::Domain) where T
    local_vel_sq = sum(fields.u.data.^2 + fields.v.data.^2)
    global_vel_sq = MPI.Allreduce(local_vel_sq, MPI.SUM, fields.u.pencil.comm)
    global_count = MPI.Allreduce(length(fields.u.data), MPI.SUM, fields.u.pencil.comm)
    return sqrt(global_vel_sq / global_count)
end

"""Gather distributed field to root process for saving"""
function gather_to_root(field::PencilArray{T,N}) where {T,N}
    comm = field.pencil.comm
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    nprocs == 1 && return Array(field.data)
    
    global_dims = size_global(field.pencil)
    
    if rank == 0
        global_array = zeros(T, global_dims...)
        local_ranges = range_local(field.pencil)
        global_array[local_ranges...] = field.data
        
        for src = 1:nprocs-1
            # Receive range info and data
            range_info = Vector{Int}(undef, 2*N)
            MPI.Recv!(range_info, src, 100, comm)
            
            ranges = ntuple(N) do i
                range_info[2*i-1]:range_info[2*i]
            end
            
            remote_size = prod(length.(ranges))
            remote_data = Vector{T}(undef, remote_size)
            MPI.Recv!(remote_data, src, 101, comm)
            
            global_array[ranges...] = reshape(remote_data, length.(ranges)...)
        end
        
        return global_array
    else
        # Send range info and data
        local_ranges = range_local(field.pencil)
        range_info = Int[]
        for r in local_ranges
            push!(range_info, first(r), last(r))
        end
        MPI.Send(range_info, 0, 100, comm)
        MPI.Send(vec(field.data), 0, 101, comm)
        return nothing
    end
end

"""Save simulation state with MPI gathering at specified time intervals"""
function save_initial_state(filename::String, fields::Fields, domain::Domain)
    comm = fields.bₛ.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Gather data to root
    b_global = gather_to_root(fields.bₛ)
    φ_global = gather_to_root(fields.φ)
    u_global = gather_to_root(fields.u)
    v_global = gather_to_root(fields.v)
    
    if rank == 0
        try
            using JLD2
            jldopen(filename, "w") do file
                file["buoyancy"] = b_global
                file["streamfunction"] = φ_global
                file["u_velocity"] = u_global
                file["v_velocity"] = v_global
                file["grid"] = (Nx=domain.Nx, Ny=domain.Ny, Nz=domain.Nz,
                               Lx=domain.Lx, Ly=domain.Ly, Lz=domain.Lz)
                file["stats"] = (
                    b_min=minimum(b_global), b_max=maximum(b_global),
                    u_max=maximum(abs.(u_global)), v_max=maximum(abs.(v_global)),
                    rms_vel=sqrt(mean(u_global.^2 + v_global.^2))
                )
            end
            println("✓ Saved: $filename")
        catch e
            println("❌ Save failed: $e")
        end
    end
    MPI.Barrier(comm)
end

"""
    run_with_time_based_output(prob, final_time; save_interval=1.0)

Run simulation with time-based output (saves every save_interval time units).
"""
function run_with_time_based_output(prob, final_time::Real; save_interval::Real=1.0)
    comm = prob.fields.bₛ.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    current_time = prob.clock.t
    next_save_time = save_interval * ceil(current_time / save_interval)
    save_counter = 0
    
    rank == 0 && println("Starting simulation with time-based saves every $(save_interval) time units")
    
    while current_time < final_time
        # Determine next target time
        target_time = min(next_save_time, final_time)
        
        # Step to target time
        step_until!(prob, target_time)
        current_time = prob.clock.t
        
        # Save if we've reached a save time
        if abs(current_time - next_save_time) < 1e-10
            save_counter += 1
            filename = "state_$(save_counter).jld2"
            
            if rank == 0
                println("Saving at t=$(round(current_time, digits=2)) → $(filename)")
            end
            
            save_initial_state(filename, prob.fields, prob.domain)
            next_save_time += save_interval
        end
        
        # Progress update
        if rank == 0 && prob.diagnostics !== nothing
            rms_vel = compute_rms_velocity(prob.fields, prob.domain)
            println("  t=$(round(current_time, digits=2)), step=$(prob.clock.step), RMS_vel=$(round(rms_vel, digits=3))")
        end
    end
    
    rank == 0 && println("✅ Simulation complete: saved $save_counter files")
    return prob
end

"""Main example function"""
function run_initial_condition_example()
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    try
        # Setup
        Nx, Ny, Nz = 512, 512, 20
        Lx = Ly = 6.0  # L/L_D = 6 from paper
        k₀, m = 14.0, 20  # Paper parameters
        
        rank == 0 && println("🌊 SSG Initial Conditions: $(Nx)×$(Ny)×$(Nz) on $nprocs processes")
        
        # RUNTIME ESTIMATES (before starting)
        if rank == 0
            base_step_time = 0.8  # seconds per step for 512² grid, single process
            parallel_efficiency = min(0.8, 4.0/sqrt(nprocs))
            step_time = base_step_time * parallel_efficiency / max(1, nprocs/4)
            
            # Full simulation estimate
            t_end = 25.0  # Full paper simulation
            n_steps = Int(round(t_end / 0.001))  # dt ≈ 0.001
            total_hours = n_steps * step_time / 3600
            n_saves = Int(round(t_end / 2.0))  # Save every 2.0 time units
            
            println("\n FULL SIMULATION RUNTIME:")
            println("   Target: t=0 → $(t_end) ($(n_saves) saves every 2.0 time units)")
            println("   Expected time: $(round(total_hours, digits=1)) hours")
            println("   Memory per process: $(round(8 * Nx * Ny * Nz * 10 / nprocs / 1e9, digits=1)) GB")
            println()
        end
        
        # Create domain and fields
        domain = make_domain(Nx, Ny, Nz; Lx=Lx, Ly=Ly, Lz=1.0,
                           z_boundary=:dirichlet, z_grid=:stretched,
                           stretch_params=(type=:exponential, β=2.0, surface_concentration=true),
                           comm=comm)
        fields = allocate_fields(domain)
        
        # Initialize spectral buoyancy
        start_time = MPI.Wtime()
        initialize_spectral_buoyancy!(fields, domain, 1.0, k₀, m; target_rms_velocity=1.0)
        init_time = MPI.Wtime() - start_time
        
        # Diagnostics (shorter, focused on key results)
        if rank == 0
            rms_vel = compute_rms_velocity(fields, domain)
            println("✓ Initialization complete ($(round(init_time, digits=1))s)")
            println("  RMS velocity: $(round(rms_vel, digits=3)) m/s ✓")
        end
        
        # Save state
        save_initial_state("initial_state.jld2", fields, domain)
        
        # Create problem for time integration
        timestepper = TimeParams{Float64}(0.001; scheme=RK3, adaptive_dt=true)
        clock = TimeState{Float64, typeof(fields.bₛ)}(0.0, fields)
        
        # Set up time-based output frequency
        save_interval = 1.0  # Save every 1.0 time units
        output_manager = OutputManager{Float64}("output";
                                               snapshot_time_freq=save_interval,
                                               full_state_time_freq=5.0,  # Full state every 5.0 time units
                                               diagnostics_time_freq=0.1, # Diagnostics every 0.1 time units
                                               save_spectral_data=true,
                                               verbose_output=true)
        
        prob = SemiGeostrophicProblem{Float64}(fields, domain, timestepper, clock;
                                             diagnostics=DiagnosticTimeSeries{Float64}(),
                                             output_settings=(manager=output_manager, 
                                                            save_interval=save_interval))
        
        rank == 0 && println(" Starting full simulation...")
        rank == 0 && println("   run_with_time_based_output(prob, 25.0; save_interval=2.0)")
        
        # Auto-start full simulation
        run_with_time_based_output(prob, 25.0; save_interval=2.0)
        
        return prob, fields, domain
        
    catch e
        rank == 0 && println("❌ Error: $e")
        MPI.Abort(comm, 1)
    finally
        MPI.Barrier(comm)
        if MPI.Initialized() && !MPI.Finalized() && abspath(PROGRAM_FILE) == @__FILE__
            MPI.Finalize()
        end
    end
end

# # Execute if run directly
# if abspath(PROGRAM_FILE) == @__FILE__
#     run_initial_condition_example()
# end