# src/output.jl
# Comprehensive JLD2 output system for semi-geostrophic simulations
# Supports time-based and step-based output frequencies
# Includes physical and spectral field output with MPI support

using PencilArrays: range_local, size_global 

#####################################################################
# JLD2 OUTPUT SYSTEM FOR SEMI-GEOSTROPHIC SIMULATIONS
#####################################################################
#
# FEATURES:
# - Time-based and step-based output frequencies
# - Physical and spectral field storage
# - Complete simulation state preservation
# - MPI-aware data gathering from distributed arrays
# - Flexible output types (snapshots, full state, diagnostics)
# - Automatic checkpoint/restart capability
# - Cross-platform compatibility
# - Comprehensive metadata preservation
#
# OUTPUT TYPES:
# 1. Snapshots: Essential fields for analysis
# 2. Full State: Complete simulation state for restart
# 3. Spectral Data: Fourier coefficients for spectral analysis
# 4. Diagnostics: Time series of integrated quantities
#
# spectral data is now organized as:
# fields/spectral/
# ├── 2d/
# │   ├── buoyancy_real, buoyancy_imag
# │   ├── streamfunction_real, streamfunction_imag  
# │   ├── tmp_complex_real, tmp_complex_imag
# │   └── tmp1_complex_real, tmp1_complex_imag
# └── 3d/
#     ├── streamfunction_real, streamfunction_imag
#     ├── tmp_complex_real, tmp_complex_imag
#     └── tmp1_complex_real, tmp1_complex_imag
#
# spectra/
# ├── 2d/
# │   ├── energy_spectrum
# │   └── enstrophy_spectrum
# └── 3d/
#     ├── energy_spectrum
#     └── enstrophy_spectrum
################################################################

# ===============================
# OUTPUT FREQUENCY MANAGEMENT
# ===============================
"""
Output frequency specification (time-based or step-based)
"""
struct OutputFrequency{T<:AbstractFloat}
    # Time-based frequency
    time_interval::Union{T, Nothing}        # Save every Δt simulation time
    
    # Step-based frequency  
    step_interval::Union{Int, Nothing}      # Save every N steps
    
    # Control flags
    use_time_based::Bool                    # Primary frequency type
    min_time_between_saves::T               # Minimum time between saves
    max_time_between_saves::T               # Maximum time between saves
    
    function OutputFrequency{T}(;
                               time_interval::Union{T, Nothing}=nothing,
                               step_interval::Union{Int, Nothing}=nothing,
                               min_time_between_saves::T=T(0.0),
                               max_time_between_saves::T=T(Inf)) where T
        
        # Validation
        if time_interval === nothing && step_interval === nothing
            error("Must specify either time_interval or step_interval")
        end
        
        if time_interval !== nothing && time_interval <= 0
            error("time_interval must be positive")
        end
        
        if step_interval !== nothing && step_interval <= 0
            error("step_interval must be positive")
        end
        
        use_time_based = time_interval !== nothing
        
        new{T}(time_interval, step_interval, use_time_based, 
               min_time_between_saves, max_time_between_saves)
    end
end

"""
Convenience constructors for output frequencies
"""
OutputFrequency(time_interval::T) where T = OutputFrequency{T}(time_interval=time_interval)
OutputFrequency(step_interval::Int, ::Type{T}=Float64) where T = OutputFrequency{T}(step_interval=step_interval)

"""
Check if output should occur based on frequency specification
"""
function should_output(freq::OutputFrequency{T}, 
                      current_time::T, current_step::Int,
                      last_output_time::T, last_output_step::Int) where T
    
    if freq.use_time_based
        # Time-based output
        time_since_last = current_time - last_output_time
        
        # Check minimum time constraint
        if time_since_last < freq.min_time_between_saves
            return false
        end
        
        # Check if we've reached the time interval
        if time_since_last >= freq.time_interval
            return true
        end
        
        # Check maximum time constraint
        if time_since_last >= freq.max_time_between_saves
            return true
        end
        
        return false
    else
        # Step-based output
        steps_since_last = current_step - last_output_step
        return steps_since_last >= freq.step_interval
    end
end

# ===========================
# SPECTRAL DATA HANDLING
# ===========================
"""
Gather distributed spectral data to root process
"""
function gather_spectral_to_root(field::PencilArray{Complex{T}, 2}) where T
    comm = field.pencil.comm
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if rank == 0
        # Root process: collect all spectral data
        nx_spec, ny_spec = size_global(field.pencil)
        global_data = zeros(Complex{T}, nx_spec, ny_spec)
        
        # Copy local data
        range_locals = range_local(field.pencil)
        global_data[range_locals[1], range_locals[2]] = field.data
        
        # Receive from other processes
        for src_rank = 1:size-1
            # Receive range information
            ranges_info = Vector{Int}(undef, 4)
            MPI.Recv!(ranges_info, src_rank, 100, comm)
            i_start, i_end, j_start, j_end = ranges_info
            
            # Receive complex data
            local_size = (i_end - i_start + 1) * (j_end - j_start + 1)
            recv_data = Vector{Complex{T}}(undef, local_size)
            MPI.Recv!(recv_data, src_rank, 101, comm)
            
            # Place in global array
            recv_matrix = reshape(recv_data, (i_end - i_start + 1, j_end - j_start + 1))
            global_data[i_start:i_end, j_start:j_end] = recv_matrix
        end
        
        return global_data
    else
        # Non-root processes: send spectral data
        range_locals = range_local(field.pencil)
        ranges_info = [range_locals[1].start, range_locals[1].stop,
                      range_locals[2].start, range_locals[2].stop]
        
        # Send range information
        MPI.Send(ranges_info, 0, 100, comm)
        
        # Send complex data
        MPI.Send(vec(field.data), 0, 101, comm)
        
        return nothing
    end
end

"""
Distribute spectral data from root to all processes
"""
function distribute_spectral_from_root!(field::PencilArray{Complex{T}, 2}, 
                                       global_data::Union{Array{Complex{T}, 2}, Nothing}) where T
    comm = field.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Broadcast approach for simplicity
    if rank == 0 && global_data !== nothing
        MPI.Bcast!(global_data, 0, comm)
    else
        nx_spec, ny_spec = size_global(field.pencil)
        global_data = zeros(Complex{T}, nx_spec, ny_spec)
        MPI.Bcast!(global_data, 0, comm)
    end
    
    # Extract local portion on all processes
    range_locals = range_local(field.pencil)
    field.data .= global_data[range_locals[1], range_locals[2]]
    
    return nothing
end

# ===================================
# COMPREHENSIVE OUTPUT FUNCTIONS
# ===================================
"""
Save complete simulation state including spectral fields (now supports 3D)
"""
function save_simulation_state_full(filename::String, 
                                   prob::SemiGeostrophicProblem{T};
                                   save_spectral::Bool=true,
                                   save_diagnostics::Bool=true,
                                   save_metadata::Bool=true,
                                   compress::Bool=true) where T
    
    # Only root process writes the file
    if MPI.Comm_rank(prob.domain.pr3d.comm) != 0
        return filename
    end
    
    # Ensure directory exists
    mkpath(dirname(filename))
    
    # Gather distributed 2D surface fields
    bₛ_global = gather_to_root(prob.fields.bₛ)
    φₛ_global = gather_to_root(prob.fields.φₛ)
    
    # Gather distributed 3D fields
    φ_global = gather_to_root(prob.fields.φ)
    u_global = gather_to_root(prob.fields.u)
    v_global = gather_to_root(prob.fields.v)
    
    # Gather multigrid workspace if needed
    φ_mg_global = gather_to_root(prob.fields.φ_mg)
    b_mg_global = gather_to_root(prob.fields.b_mg)
    
    # Gather 2D scratch arrays
    R_global = gather_to_root(prob.fields.R)
    tmp_global = gather_to_root(prob.fields.tmp)
    tmp2_global = gather_to_root(prob.fields.tmp2)
    tmp3_global = gather_to_root(prob.fields.tmp3)
    
    # Gather spectral data if requested
    local bshat_global, φshat_global, φhat_global, tmpc_2d_global, tmpc1_2d_global, tmpc_3d_global, tmpc1_3d_global
    if save_spectral
        # 2D spectral fields
        bshat_global = gather_spectral_to_root(prob.fields.bshat)
        φshat_global = gather_spectral_to_root(prob.fields.φshat)
        tmpc_2d_global = gather_spectral_to_root(prob.fields.tmpc_2d)
        tmpc1_2d_global = gather_spectral_to_root(prob.fields.tmpc1_2d)
        
        # 3D spectral fields  
        φhat_global = gather_spectral_to_root(prob.fields.φhat)
        tmpc_3d_global = gather_spectral_to_root(prob.fields.tmpc_3d)
        tmpc1_3d_global = gather_spectral_to_root(prob.fields.tmpc1_3d)
    end
    
    # Create JLD2 file with comprehensive data
    jldopen(filename, "w"; compress=compress) do file
        # Time and integration state
        file["time"] = prob.clock.t
        file["step"] = prob.clock.step
        file["dt"]   = prob.clock.dt_actual > 0 ? prob.clock.dt_actual : prob.timestepper.dt
        
        # Grid information
        file["grid/Nx"] = prob.domain.Nx
        file["grid/Ny"] = prob.domain.Ny
        file["grid/Nz"] = prob.domain.Nz

        file["grid/Lx"] = prob.domain.Lx
        file["grid/Ly"] = prob.domain.Ly
        file["grid/Lz"] = prob.domain.Lz

        file["grid/x"]  = prob.domain.x  
        file["grid/y"]  = prob.domain.y  
        file["grid/z"]  = prob.domain.z
        file["grid/dz"] = prob.domain.dz
        
        # Wavenumber arrays
        if save_spectral
            file["grid/kx"] = collect(prob.domain.kx)
            file["grid/ky"] = collect(prob.domain.ky)
        end
        
        # 2D Surface fields
        file["fields/surface/buoyancy"]         = bₛ_global
        file["fields/surface/streamfunction"]   = φₛ_global
        file["fields/surface/residual"]         = R_global
        file["fields/surface/tmp"]              = tmp_global
        file["fields/surface/tmp2"]             = tmp2_global  
        file["fields/surface/tmp3"]             = tmp3_global
        
        # 3D Physical fields
        file["fields/3d/streamfunction"]    = φ_global
        file["fields/3d/u_velocity"]        = u_global
        file["fields/3d/v_velocity"]        = v_global
        file["fields/3d/phi_mg"]            = φ_mg_global
        file["fields/3d/b_mg"]              = b_mg_global
        
        # Spectral fields
        if save_spectral
            # 2D spectral fields
            file["fields/spectral/2d/buoyancy_real"]        = real(bshat_global)
            file["fields/spectral/2d/buoyancy_imag"]        = imag(bshat_global)
            file["fields/spectral/2d/streamfunction_real"]  = real(φshat_global)
            file["fields/spectral/2d/streamfunction_imag"]  = imag(φshat_global)
            file["fields/spectral/2d/tmp_complex_real"]     = real(tmpc_2d_global)
            file["fields/spectral/2d/tmp_complex_imag"]     = imag(tmpc_2d_global)
            file["fields/spectral/2d/tmp1_complex_real"]    = real(tmpc1_2d_global)
            file["fields/spectral/2d/tmp1_complex_imag"]    = imag(tmpc1_2d_global)
            
            # 3D spectral fields
            file["fields/spectral/3d/streamfunction_real"]  = real(φhat_global)
            file["fields/spectral/3d/streamfunction_imag"]  = imag(φhat_global)
            file["fields/spectral/3d/tmp_complex_real"]     = real(tmpc_3d_global)
            file["fields/spectral/3d/tmp_complex_imag"]     = imag(tmpc_3d_global)
            file["fields/spectral/3d/tmp1_complex_real"]    = real(tmpc1_3d_global)
            file["fields/spectral/3d/tmp1_complex_imag"]    = imag(tmpc1_3d_global)
        end
        
        # Time integration parameters
        file["timestepper/scheme"]          = string(prob.timestepper.scheme)
        file["timestepper/dt"]              = prob.timestepper.dt
        file["timestepper/adaptive_dt"]     = prob.timestepper.adaptive_dt
        file["timestepper/filter_freq"]     = prob.timestepper.filter_freq
        file["timestepper/filter_strength"] = prob.timestepper.filter_strength
        file["timestepper/cfl_safety"]      = prob.timestepper.cfl_safety
        file["timestepper/max_dt"]          = prob.timestepper.max_dt
        file["timestepper/min_dt"]          = prob.timestepper.min_dt
        
        # Save diagnostics if available and requested
        if save_diagnostics && prob.diagnostics !== nothing && length(prob.diagnostics.times) > 0
            file["diagnostics/times"]           = prob.diagnostics.times
            file["diagnostics/kinetic_energy"]  = prob.diagnostics.kinetic_energy
            file["diagnostics/enstrophy"]       = prob.diagnostics.enstrophy
            file["diagnostics/total_buoyancy"]  = prob.diagnostics.total_buoyancy
            file["diagnostics/max_divergence"]  = prob.diagnostics.max_divergence
            file["diagnostics/max_cfl"]         = prob.diagnostics.max_cfl
        end
        
        # Comprehensive metadata
        if save_metadata
            file["metadata/creation_time"]      = string(now())
            file["metadata/julia_version"]      = string(VERSION)
            file["metadata/hostname"]           = gethostname()
            file["metadata/mpi_size"]           = MPI.Comm_size(prob.domain.pr3d.comm)
            file["metadata/precision"]          = string(T)
            file["metadata/equation_type"]      = "surface_semi_geostrophic"
            file["metadata/description"]        = "Surface buoyancy evolution with Monge-Ampère constraint"
            file["metadata/file_type"]          = "complete_simulation_state"
            file["metadata/has_spectral_data"]  = save_spectral
            file["metadata/has_diagnostics"]    = save_diagnostics && prob.diagnostics !== nothing
            
            # Physics parameters
            file["metadata/physics/domain_aspect_ratio"]    = prob.domain.Lx / prob.domain.Ly
            file["metadata/physics/grid_spacing_x"]         = prob.domain.Lx / prob.domain.Nx
            file["metadata/physics/grid_spacing_y"]         = prob.domain.Ly / prob.domain.Ny
            file["metadata/physics/grid_spacing_z_min"]     = minimum(prob.domain.dz)
            file["metadata/physics/grid_spacing_z_max"]     = maximum(prob.domain.dz)
            file["metadata/physics/z_boundary"]             = string(prob.domain.z_boundary)
            file["metadata/physics/z_grid"]                 = string(prob.domain.z_grid)
        end
    end
    
    return filename
end


"""
Save spectral-only snapshot for frequency domain analysis (now supports 3D)
"""
function save_spectral_snapshot(filename::String, prob::SemiGeostrophicProblem{T};
                               include_derived_spectra::Bool=true) where T
    
    if MPI.Comm_rank(prob.domain.pc.comm) != 0
        return filename
    end
    
    mkpath(dirname(filename))
    
    # Ensure spectral fields are up to date
    rfft_2d!(prob.domain, prob.fields.bₛ, prob.fields.bshat)
    rfft!(   prob.domain, prob.fields.φ,  prob.fields.φhat)
    
    # Gather spectral data
    bshat_global = gather_spectral_to_root(prob.fields.bshat)
    φhat_global  = gather_spectral_to_root(prob.fields.φhat)
    
    # Compute derived spectral quantities if requested
    local energy_spectrum, enstrophy_spectrum
    if include_derived_spectra
        energy_spectrum     = compute_energy_spectrum(φhat_global, prob.domain)
        enstrophy_spectrum  = compute_enstrophy_spectrum(φhat_global, prob.domain)
    end
    
    jldopen(filename, "w") do file
        # Basic info
        file["time"] = prob.clock.t
        file["step"] = prob.clock.step
        
        # Grid info
        file["Nx"] = prob.domain.Nx
        file["Ny"] = prob.domain.Ny
        file["Nz"] = prob.domain.Nz

        file["Lx"] = prob.domain.Lx
        file["Ly"] = prob.domain.Ly
        file["Lz"] = prob.domain.Lz
        
        # Wavenumbers
        file["kx"] = collect(prob.domain.kx)
        file["ky"] = collect(prob.domain.ky)
        file["kr"] = collect(rfftfreq(prob.domain.Nx, 2π*prob.domain.Nx/prob.domain.Lx))
        
        # Spectral fields (as complex numbers)
        file["buoyancy_hat"]        = bshat_global
        file["streamfunction_hat"]  = φhat_global
        
        # Derived spectra
        if include_derived_spectra
            file["energy_spectrum"]     = energy_spectrum
            file["enstrophy_spectrum"]  = enstrophy_spectrum
        end
        
        # Metadata
        file["file_type"]       = "spectral_snapshot"
        file["creation_time"]   = string(now())
    end
    
    return filename
end


"""
Compute radially averaged energy spectrum for 2D or 3D spectral data
"""
function compute_energy_spectrum(φhat::Array{Complex{T}, N}, 
                            domain::Domain; 
                            is_2d::Bool=false) where {T, N}
    if is_2d
        nx, ny = size(φhat)
        nz = 1
    else
        nx, ny, nz = size(φhat)
    end
    
    # Create wavenumber arrays
    kx = rfftfreq(domain.Nx, 2π*domain.Nx/domain.Lx)
    ky = fftfreq(domain.Ny,  2π*domain.Ny/domain.Ly)
    
    # Maximum wavenumber for binning
    k_max = min(length(kx)-1, domain.Ny÷2)
    k_bins = 0:k_max
    energy_spectrum = zeros(T, length(k_bins))
    
    # Compute energy spectrum
    if is_2d
        # 2D case
        for (j, ky_val) in enumerate(ky)
            for (i, kx_val) in enumerate(kx)
                k_mag = sqrt(kx_val^2 + ky_val^2)
                k_bin = round(Int, k_mag)
                
                if k_bin <= k_max
                    # Energy density: 0.5 * |∇φ|² = 0.5 * k² * |φ̂|²
                    energy_density = 0.5 * k_mag^2 * abs2(φhat[i,j])
                    energy_spectrum[k_bin+1] += energy_density
                end
            end
        end
    else
        # 3D case - sum over all z levels
        for k in 1:nz
            for (j, ky_val) in enumerate(ky)
                for (i, kx_val) in enumerate(kx)
                    k_mag = sqrt(kx_val^2 + ky_val^2)
                    k_bin = round(Int, k_mag)
                    
                    if k_bin <= k_max
                        # Energy density: 0.5 * |∇φ|² = 0.5 * k² * |φ̂|²
                        energy_density = 0.5 * k_mag^2 * abs2(φhat[i,j,k])
                        energy_spectrum[k_bin+1] += energy_density
                    end
                end
            end
        end
    end
    
    return energy_spectrum
end


"""
Compute radially averaged enstrophy spectrum for 2D or 3D spectral data
"""
function compute_enstrophy_spectrum(φhat::Array{Complex{T}, N}, 
                                domain::Domain; 
                                is_2d::Bool=false) where {T, N}
    if is_2d
        nx, ny = size(φhat)
        nz = 1
    else
        nx, ny, nz = size(φhat)
    end
    
    # Create wavenumber arrays
    kx = rfftfreq(domain.Nx, 2π*domain.Nx/domain.Lx)
    ky = fftfreq(domain.Ny,  2π*domain.Ny/domain.Ly)
    
    # Maximum wavenumber for binning
    k_max = min(length(kx)-1, domain.Ny÷2)
    k_bins = 0:k_max
    enstrophy_spectrum = zeros(T, length(k_bins))
    
    # Compute enstrophy spectrum
    if is_2d
        # 2D case
        for (j, ky_val) in enumerate(ky)
            for (i, kx_val) in enumerate(kx)
                k_mag = sqrt(kx_val^2 + ky_val^2)
                k_bin = round(Int, k_mag)
                
                if k_bin <= k_max
                    # Enstrophy density: 0.5 * |ω|² = 0.5 * k⁴ * |φ̂|²
                    enstrophy_density = 0.5 * k_mag^4 * abs2(φhat[i,j])
                    enstrophy_spectrum[k_bin+1] += enstrophy_density
                end
            end
        end
    else
        # 3D case - sum over all z levels
        for k in 1:nz
            for (j, ky_val) in enumerate(ky)
                for (i, kx_val) in enumerate(kx)
                    k_mag = sqrt(kx_val^2 + ky_val^2)
                    k_bin = round(Int, k_mag)
                    
                    if k_bin <= k_max
                        # Enstrophy density: 0.5 * |ω|² = 0.5 * k⁴ * |φ̂|²
                        enstrophy_density = 0.5 * k_mag^4 * abs2(φhat[i,j,k])
                        enstrophy_spectrum[k_bin+1] += enstrophy_density
                    end
                end
            end
        end
    end
    
    return enstrophy_spectrum
end


# ============================================================================
# TIME-BASED OUTPUT MANAGER
# ============================================================================

"""
Advanced output manager with time-based and step-based frequencies
"""
mutable struct OutputManager{T<:AbstractFloat}
    base_dir::String
    
    # Output frequencies
    snapshot_freq::OutputFrequency{T}
    full_state_freq::OutputFrequency{T}
    spectral_freq::OutputFrequency{T}
    diagnostics_freq::OutputFrequency{T}
    
    # Last output tracking
    last_snapshot_time::T
    last_snapshot_step::Int
    last_full_state_time::T
    last_full_state_step::Int
    last_spectral_time::T
    last_spectral_step::Int
    last_diagnostics_time::T
    last_diagnostics_step::Int
    
    # Output counters
    snapshot_counter::Int
    full_state_counter::Int
    spectral_counter::Int
    diagnostics_counter::Int
    
    # Options
    save_spectral_data::Bool
    compress_files::Bool
    verbose_output::Bool
    
    function OutputManager{T}(base_dir::String;
                                    # Time-based frequencies (in simulation time units)
                                    snapshot_time_freq::Union{T, Nothing}=nothing,
                                    full_state_time_freq::Union{T, Nothing}=nothing,
                                    spectral_time_freq::Union{T, Nothing}=nothing,
                                    diagnostics_time_freq::Union{T, Nothing}=nothing,
                                    
                                    # Step-based frequencies (fallbacks)
                                    snapshot_step_freq::Union{Int, Nothing}=100,
                                    full_state_step_freq::Union{Int, Nothing}=1000,
                                    spectral_step_freq::Union{Int, Nothing}=500,
                                    diagnostics_step_freq::Union{Int, Nothing}=50,
                                    
                                    # Options
                                    save_spectral_data::Bool=true,
                                    compress_files::Bool=true,
                                    verbose_output::Bool=true) where T
        
        # Create directory structure
        mkpath(base_dir)
        mkpath(joinpath(base_dir, "snapshots"))
        mkpath(joinpath(base_dir, "full_states"))
        mkpath(joinpath(base_dir, "spectral"))
        mkpath(joinpath(base_dir, "diagnostics"))
        
        # Set up frequencies
        snapshot_freq = snapshot_time_freq !== nothing ? 
                       OutputFrequency{T}(time_interval=snapshot_time_freq) :
                       OutputFrequency{T}(step_interval=snapshot_step_freq)
                       
        full_state_freq = full_state_time_freq !== nothing ?
                        OutputFrequency{T}(time_interval=full_state_time_freq) :
                        OutputFrequency{T}(step_interval=full_state_step_freq)
                         
        spectral_freq = spectral_time_freq !== nothing ?
                       OutputFrequency{T}(time_interval=spectral_time_freq) :
                       OutputFrequency{T}(step_interval=spectral_step_freq)
                       
        diagnostics_freq = diagnostics_time_freq !== nothing ?
                        OutputFrequency{T}(time_interval=diagnostics_time_freq) :
                        OutputFrequency{T}(step_interval=diagnostics_step_freq)
        
        new{T}(base_dir, snapshot_freq, full_state_freq, spectral_freq, diagnostics_freq,
               T(0), 0, T(0), 0, T(0), 0, T(0), 0,
               0, 0, 0, 0,
               save_spectral_data, compress_files, verbose_output)
    end
end

"""
Process all outputs based on current simulation state
"""
function process_all_outputs!(manager::OutputManager{T}, 
                             prob::SemiGeostrophicProblem{T}) where T
    
    current_time = prob.clock.t
    current_step = prob.clock.step
    is_root = MPI.Comm_rank(prob.domain.pc.comm) == 0
    
    # Process snapshots
    if should_output(manager.snapshot_freq, current_time, current_step,
                    manager.last_snapshot_time, manager.last_snapshot_step)
        
        manager.snapshot_counter += 1
        filename = joinpath(manager.base_dir, "snapshots",
                          @sprintf("snapshot_%04d_t%.4f.jld2", 
                                  manager.snapshot_counter, current_time))
        
        save_snapshot(filename, prob; include_velocities=true, include_residual=false)
        
        manager.last_snapshot_time = current_time
        manager.last_snapshot_step = current_step
        
        if manager.verbose_output && is_root
            @info "Saved snapshot: $(basename(filename))"
        end
    end
    
    # Process full states
    if should_output(manager.full_state_freq, current_time, current_step,
                    manager.last_full_state_time, manager.last_full_state_step)
        
        manager.full_state_counter += 1
        filename = joinpath(manager.base_dir, "full_states",
                          @sprintf("state_%04d_t%.4f.jld2", 
                                  manager.full_state_counter, current_time))
        
        save_simulation_state_full(filename, prob; 
                                  save_spectral=manager.save_spectral_data,
                                  compress=manager.compress_files)
        
        manager.last_full_state_time = current_time
        manager.last_full_state_step = current_step
        
        if manager.verbose_output && is_root
            @info "Saved full state: $(basename(filename))"
        end
    end
    
    # Process spectral data
    if (manager.save_spectral_data && 
        should_output(manager.spectral_freq, current_time, current_step,
                     manager.last_spectral_time, manager.last_spectral_step))
        
        manager.spectral_counter += 1
        filename = joinpath(manager.base_dir, "spectral",
                          @sprintf("spectral_%04d_t%.4f.jld2", 
                                  manager.spectral_counter, current_time))
        
        save_spectral_snapshot(filename, prob; include_derived_spectra=true)
        
        manager.last_spectral_time = current_time
        manager.last_spectral_step = current_step
        
        if manager.verbose_output && is_root
            @info "Saved spectral data: $(basename(filename))"
        end
    end
    
    # Process diagnostics
    if (prob.diagnostics !== nothing &&
        should_output(manager.diagnostics_freq, current_time, current_step,
                     manager.last_diagnostics_time, manager.last_diagnostics_step))
        
        manager.diagnostics_counter += 1
        filename = joinpath(manager.base_dir, "diagnostics",
                          @sprintf("diagnostics_%04d_t%.4f.jld2", 
                                  manager.diagnostics_counter, current_time))
        
        metadata = Dict("step" => current_step, "time" => current_time,
                       "scheme" => string(prob.timestepper.scheme))
        save_diagnostics(filename, prob.diagnostics; metadata=metadata)
        
        manager.last_diagnostics_time = current_time
        manager.last_diagnostics_step = current_step
        
        if manager.verbose_output && is_root
            @info "Saved diagnostics: $(basename(filename))"
        end
    end
    
    return nothing
end


"""
Example: Mixed time and step-based frequencies
"""
function demo_mixed_frequencies()
    println(" Mixed Frequency Output Demo")
    println("=" ^ 35)
    
    # Example: Fast diagnostics, slower full states
    # output_manager = AdvancedOutputManager{Float64}("mixed_output";
    #     diagnostics_time_freq=0.05,    # Fast: every 0.05 time units
    #     snapshot_time_freq=0.5,        # Medium: every 0.5 time units
    #     spectral_step_freq=1000,       # Step-based: every 1000 steps
    #     full_state_time_freq=5.0,      # Slow: every 5.0 time units
    #     save_spectral_data=true)
    
    println("Mixed frequency capabilities:")
    println("  Different frequencies for different output types")
    println("  Time-based for physics (diagnostics, snapshots)")  
    println("  Step-based for numerical analysis (spectral)")
    println("  Flexible configuration per output type")
    println("  Independent timing for each output stream")
    
    return true
end


# ================================
# COMPREHENSIVE DATA GATHERING 
# ================================
"""
Gather distributed 2D data to root process
"""
function gather_to_root(field::PencilArray{T, 2}) where T
    comm   = field.pencil.comm
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        # Root process: collect all data
        nx_global, ny_global = size_global(field.pencil)
        global_data = zeros(T, nx_global, ny_global)
        
        # Copy local data
        range_locals = range_local(field.pencil)
        global_data[range_locals[1], range_locals[2]] = field.data
        
        # Receive from other processes
        for src_rank = 1:nprocs-1
            # Receive range information first
            range_buffer = Vector{Int}(undef, 4)
            MPI.Recv!(range_buffer, src_rank, 200, comm)
            i_start, i_end, j_start, j_end = range_buffer
            
            # Calculate expected data size
            ni_local = i_end - i_start + 1
            nj_local = j_end - j_start + 1
            expected_size = ni_local * nj_local
            
            # Receive the actual data
            data_buffer = Vector{T}(undef, expected_size)
            MPI.Recv!(data_buffer, src_rank, 201, comm)
            
            # Reshape and place in global array
            local_data = reshape(data_buffer, (ni_local, nj_local))
            global_data[i_start:i_end, j_start:j_end] = local_data
        end
        
        return global_data
    else
        # Non-root processes: send data to root
        range_locals = range_local(field.pencil)
        range_info = [range_locals[1].start, range_locals[1].stop,
                     range_locals[2].start, range_locals[2].stop]
        
        # Send range information
        MPI.Send(range_info, 0, 200, comm)
        
        # Send data
        MPI.Send(vec(field.data), 0, 201, comm)
        
        return nothing
    end
end

"""
Gather distributed 3D data to root process
"""
function gather_to_root(field::PencilArray{T, 3}) where T
    comm   = field.pencil.comm
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        # Root process: collect all data
        nx_global, ny_global, nz_global = size_global(field.pencil)
        global_data = zeros(T, nx_global, ny_global, nz_global)
        
        # Copy local data
        range_locals = range_local(field.pencil)
        global_data[range_locals[1], range_locals[2], range_locals[3]] = field.data
        
        # Receive from other processes
        for src_rank = 1:nprocs-1
            # Receive range information first (6 integers for 3D)
            range_buffer = Vector{Int}(undef, 6)
            MPI.Recv!(range_buffer, src_rank, 300, comm)
            i_start, i_end, j_start, j_end, k_start, k_end = range_buffer
            
            # Calculate expected data size
            ni_local = i_end - i_start + 1
            nj_local = j_end - j_start + 1
            nk_local = k_end - k_start + 1
            expected_size = ni_local * nj_local * nk_local
            
            # Receive the actual data
            data_buffer = Vector{T}(undef, expected_size)
            MPI.Recv!(data_buffer, src_rank, 301, comm)
            
            # Reshape and place in global array
            local_data = reshape(data_buffer, (ni_local, nj_local, nk_local))
            global_data[i_start:i_end, j_start:j_end, k_start:k_end] = local_data
        end
        
        return global_data
    else
        # Non-root processes: send data to root
        range_locals = range_local(field.pencil)
        range_info = [range_locals[1].start, range_locals[1].stop,
                     range_locals[2].start, range_locals[2].stop,
                     range_locals[3].start, range_locals[3].stop]
        
        # Send range information
        MPI.Send(range_info, 0, 300, comm)
        
        # Send data
        MPI.Send(vec(field.data), 0, 301, comm)
        
        return nothing
    end
end

"""
Distribute 2D data from root to all processes
"""
function distribute_from_root!(field::PencilArray{T, 2}, 
                            global_data::Union{Array{T, 2}, Nothing}) where T

    comm = field.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Get expected global dimensions
    nx_global, ny_global = size_global(field.pencil)
    
    if rank == 0
        # Validate input on root
        if global_data === nothing
            error("Root process must provide global_data")
        end
        if size(global_data) != (nx_global, ny_global)
            error("Global data size $(size(global_data)) doesn't match expected ($nx_global, $ny_global)")
        end
        
        # Broadcast the global data
        MPI.Bcast!(global_data, 0, comm)
    else
        # Receive broadcast
        global_data = zeros(T, nx_global, ny_global)
        MPI.Bcast!(global_data, 0, comm)
    end
    
    # Extract local portion on all processes
    range_locals = range_local(field.pencil)
    field.data .= global_data[range_locals[1], range_locals[2]]
    
    return nothing
end


"""
Distribute 3D data from root to all processes
"""
function distribute_from_root!(field::PencilArray{T, 3}, 
                               global_data::Union{Array{T, 3}, Nothing}) where T
    comm = field.pencil.comm
    rank = MPI.Comm_rank(comm)
    
    # Get expected global dimensions
    nx_global, ny_global, nz_global = size_global(field.pencil)
    
    if rank == 0
        # Validate input on root
        if global_data === nothing
            error("Root process must provide global_data")
        end
        if size(global_data) != (nx_global, ny_global, nz_global)
            error("Global data size $(size(global_data)) doesn't match expected ($nx_global, $ny_global, $nz_global)")
        end
        
        # Broadcast the global data
        MPI.Bcast!(global_data, 0, comm)
    else
        # Receive broadcast
        global_data = zeros(T, nx_global, ny_global, nz_global)
        MPI.Bcast!(global_data, 0, comm)
    end
    
    # Extract local portion on all processes
    range_locals = range_local(field.pencil)
    field.data .= global_data[range_locals[1], range_locals[2], range_locals[3]]
    
    return nothing
end


# =========================
# RESTART FUNCTIONALITY
# =========================
"""
Load complete simulation state with full validation (supports 3D)
"""
function load_simulation_state_full(filename::String, domain::Domain{T};
                                   load_spectral::Bool=true,
                                   load_diagnostics::Bool=true,
                                   validate_grid::Bool=true) where T
    
    # Check file existence
    isfile(filename) || error("File $filename does not exist")
    
    local prob
    
    jldopen(filename, "r") do file
        # Load basic information
        time = file["time"]
        step = file["step"]
        dt = file["dt"]
        
        # Grid validation
        if validate_grid
            file_Nx = file["grid/Nx"] 
            file_Ny = file["grid/Ny"]
            file_Nz = file["grid/Nz"]

            file_Lx = file["grid/Lx"]
            file_Ly = file["grid/Ly"]
            file_Lz = file["grid/Lz"]
            
            grid_compatible = (file_Nx == domain.Nx && file_Ny == domain.Ny && file_Nz == domain.Nz &&
                             isapprox(file_Lx, domain.Lx, rtol=1e-10) &&
                             isapprox(file_Ly, domain.Ly, rtol=1e-10) &&
                             isapprox(file_Lz, domain.Lz, rtol=1e-10))
            
            if !grid_compatible
                error("Grid incompatibility detected!\n" *
                      "File: $(file_Nx)×$(file_Ny)×$(file_Nz), Lx=$(file_Lx), Ly=$(file_Ly), Lz=$(file_Lz)\n" *
                      "Domain: $(domain.Nx)×$(domain.Ny)×$(domain.Nz), Lx=$(domain.Lx), Ly=$(domain.Ly), Lz=$(domain.Lz)")
            end
        end
        
        # Load timestepper parameters
        scheme_str      = file["timestepper/scheme"]
        scheme          = eval(Symbol(scheme_str))
        timestepper_dt  = file["timestepper/dt"]
        adaptive_dt     = file["timestepper/adaptive_dt"]
        filter_freq     = file["timestepper/filter_freq"]
        filter_strength = file["timestepper/filter_strength"]
        cfl_safety      = file["timestepper/cfl_safety"]
        
        # Create problem with loaded parameters
        prob = SemiGeostrophicProblem(domain;
            scheme=scheme,
            dt=timestepper_dt,
            initial_time=time,
            adaptive_dt=adaptive_dt,
            filter_freq=filter_freq,
            filter_strength=filter_strength,
            enable_diagnostics=load_diagnostics)
        
        # Load 2D surface fields
        bₛ_global = file["fields/surface/buoyancy"]
        φₛ_global = file["fields/surface/streamfunction"]
        
        distribute_from_root!(prob.fields.bₛ, bₛ_global)
        distribute_from_root!(prob.fields.φₛ, φₛ_global)
        
        # Load 3D fields
        φ_global = file["fields/3d/streamfunction"]
        u_global = file["fields/3d/u_velocity"]
        v_global = file["fields/3d/v_velocity"]
        
        distribute_from_root!(prob.fields.φ, φ_global)
        distribute_from_root!(prob.fields.u, u_global)
        distribute_from_root!(prob.fields.v, v_global)
        
        # Load spectral fields if available and requested
        if load_spectral && haskey(file, "fields/spectral/2d/buoyancy_real")
            # 2D spectral fields
            bshat_real      = file["fields/spectral/2d/buoyancy_real"]
            bshat_imag      = file["fields/spectral/2d/buoyancy_imag"]
            bshat_global    = complex.(bshat_real, bshat_imag)
            
            φshat_real      = file["fields/spectral/2d/streamfunction_real"]
            φshat_imag      = file["fields/spectral/2d/streamfunction_imag"]
            φshat_global    = complex.(φshat_real, φshat_imag)
            
            distribute_spectral_from_root!(prob.fields.bshat, bshat_global)
            distribute_spectral_from_root!(prob.fields.φshat, φshat_global)
            
            # Load 2D spectral scratch arrays if available
            if haskey(file, "fields/spectral/2d/tmp_complex_real")
                tmpc_2d_real    = file["fields/spectral/2d/tmp_complex_real"]
                tmpc_2d_imag    = file["fields/spectral/2d/tmp_complex_imag"]
                tmpc_2d_global  = complex.(tmpc_2d_real, tmpc_2d_imag)
                distribute_spectral_from_root!(prob.fields.tmpc_2d, tmpc_2d_global)
            end
            
            if haskey(file, "fields/spectral/2d/tmp1_complex_real")
                tmpc1_2d_real   = file["fields/spectral/2d/tmp1_complex_real"]
                tmpc1_2d_imag   = file["fields/spectral/2d/tmp1_complex_imag"]
                tmpc1_2d_global = complex.(tmpc1_2d_real, tmpc1_2d_imag)
                distribute_spectral_from_root!(prob.fields.tmpc1_2d, tmpc1_2d_global)
            end
            
            # 3D spectral fields
            if haskey(file, "fields/spectral/3d/streamfunction_real")
                φhat_real   = file["fields/spectral/3d/streamfunction_real"]
                φhat_imag   = file["fields/spectral/3d/streamfunction_imag"]
                φhat_global = complex.(φhat_real, φhat_imag)
                distribute_spectral_from_root!(prob.fields.φhat, φhat_global)
            end
            
            # Load 3D spectral scratch arrays if available
            if haskey(file, "fields/spectral/3d/tmp_complex_real")
                tmpc_3d_real    = file["fields/spectral/3d/tmp_complex_real"]
                tmpc_3d_imag    = file["fields/spectral/3d/tmp_complex_imag"]
                tmpc_3d_global  = complex.(tmpc_3d_real, tmpc_3d_imag)
                distribute_spectral_from_root!(prob.fields.tmpc_3d, tmpc_3d_global)
            end
            
            if haskey(file, "fields/spectral/3d/tmp1_complex_real")
                tmpc1_3d_real   = file["fields/spectral/3d/tmp1_complex_real"]
                tmpc1_3d_imag   = file["fields/spectral/3d/tmp1_complex_imag"]
                tmpc1_3d_global = complex.(tmpc1_3d_real, tmpc1_3d_imag)
                distribute_spectral_from_root!(prob.fields.tmpc1_3d, tmpc1_3d_global)
            end
        end
        
        # Update clock state
        prob.clock.t            = time
        prob.clock.step         = step
        prob.clock.dt_actual    = dt
        
        # Load diagnostics if available and requested
        if load_diagnostics && prob.diagnostics !== nothing && haskey(file, "diagnostics/times")
            prob.diagnostics.times          = file["diagnostics/times"]
            prob.diagnostics.kinetic_energy = file["diagnostics/kinetic_energy"]
            prob.diagnostics.enstrophy      = file["diagnostics/enstrophy"]
            prob.diagnostics.total_buoyancy = file["diagnostics/total_buoyancy"]
            prob.diagnostics.max_divergence = file["diagnostics/max_divergence"]
            prob.diagnostics.max_cfl        = file["diagnostics/max_cfl"]
        end
    end
    
    return prob
end


"""
Validate loaded simulation state
"""
function validate_simulation_state(prob::SemiGeostrophicProblem{T}) where T
    validation_results = Dict{String, Any}()
    
    # Check field statistics
    b_stats = enhanced_field_stats(prob.fields)[:b]
    validation_results["buoyancy_finite"] = isfinite(b_stats.max) && isfinite(b_stats.min)
    validation_results["buoyancy_range"]  = (b_stats.min, b_stats.max)
    
    # Check conservation if diagnostics available
    if prob.diagnostics !== nothing && length(prob.diagnostics.total_buoyancy) > 1
        b_drift = abs(prob.diagnostics.total_buoyancy[end] - prob.diagnostics.total_buoyancy[1])
        b_initial = abs(prob.diagnostics.total_buoyancy[1])
        validation_results["buoyancy_conservation"] = b_drift / (b_initial + eps(T))
    end
    
    # Check time consistency
    validation_results["time_positive"] = prob.clock.t >= 0
    validation_results["step_positive"] = prob.clock.step >= 0
    validation_results["dt_positive"]   = prob.timestepper.dt > 0
    
    # Overall validation
    validation_results["valid"] = all(values(validation_results))
    
    return validation_results
end

# ==================================
# BATCH PROCESSING AND UTILITIES
# ==================================
"""
Process multiple output files for batch analysis
"""
function process_output_series(output_dir::String, pattern::String="*.jld2";
                              output_summary::Bool=true)
    
    files = filter(f -> occursin(Regex(replace(pattern, "*" => ".*")), f), 
                  readdir(output_dir, join=true))
    sort!(files)
    
    if isempty(files)
        @warn "No files found matching pattern $pattern in $output_dir"
        return nothing
    end
    
    summary_data = []
    
    for file in files
        try
            jldopen(file, "r") do f
                # Extract basic info
                time = haskey(f, "time") ? f["time"] : NaN
                step = haskey(f, "step") ? f["step"] : -1
                file_type = haskey(f, "file_type") ? f["file_type"] : "unknown"
                
                # File size
                file_size = filesize(file)
                
                push!(summary_data, (
                    filename = basename(file),
                    time = time,
                    step = step,
                    type = file_type,
                    size_mb = file_size / (1024^2)
                ))
            end
        catch e
            @warn "Failed to process file $file: $e"
        end
    end
    
    if output_summary
        println("Output File Summary:")
        println("=" ^ 50)
        @printf("%-25s %10s %8s %12s %8s\n", "Filename", "Time", "Step", "Type", "Size(MB)")
        println("-" ^ 70)
        
        for data in summary_data
            @printf("%-25s %10.4f %8d %12s %8.2f\n", 
                   data.filename, data.time, data.step, data.type, data.size_mb)
        end
        
        total_size = sum(d.size_mb for d in summary_data)
        println("-" ^ 70)
        @printf("Total: %d files, %.2f MB\n", length(summary_data), total_size)
    end
    
    return summary_data
end

"""
Create output manifest for reproducibility
"""
function create_output_manifest(output_dir::String)
    manifest = Dict{String, Any}()
    
    # Basic info
    manifest["creation_time"] = string(now())
    manifest["output_directory"] = output_dir
    manifest["julia_version"] = string(VERSION)
    manifest["hostname"] = gethostname()
    
    # Scan for different output types
    for subdir in ["snapshots", "full_states", "spectral", "diagnostics"]
        subdir_path = joinpath(output_dir, subdir)
        if isdir(subdir_path)
            files = readdir(subdir_path)
            manifest[subdir] = Dict(
                "count" => length(files),
                "files" => files,
                "total_size_mb" => sum(filesize(joinpath(subdir_path, f)) for f in files) / (1024^2)
            )
        end
    end
    
    # Save manifest
    manifest_file = joinpath(output_dir, "output_manifest.jld2")
    jldopen(manifest_file, "w") do file
        for (key, value) in manifest
            file[key] = value
        end
    end
    
    return manifest_file
end



# ===================================
# COMPREHENSIVE EXAMPLE USAGE
# ===================================
# """
# Complete example showing all output capabilities
# """
# function demo_complete_jld2_workflow()
#     println("  Complete JLD2 Output Workflow Demo")
#     println("=" ^ 45)
    
#     # This would be your actual simulation setup
#     # dom = Domain(512, 512, 2π, 2π, MPI.COMM_WORLD)
    
#     # Create advanced output manager
#     # output_manager = AdvancedOutputManager{Float64}("production_run";
#     #     # Time-based frequencies
#     #     snapshot_time_freq=0.1,      # Every 0.1 time units
#     #     spectral_time_freq=0.5,      # Every 0.5 time units  
#     #     diagnostics_time_freq=0.05,  # Every 0.05 time units
#     #     full_state_time_freq=2.0,    # Every 2.0 time units
#     #     
#     #     # Options
#     #     save_spectral_data=true,
#     #     compress_files=true,
#     #     verbose_output=true)
    
#     # Initialize problem
#     # prob = SemiGeostrophicProblem(dom; scheme=RK3, dt=0.005, adaptive_dt=true)
#     # set_initial_conditions!(prob, initialize_taylor_green!; amplitude=2.0)
#     # set_initial_conditions!(prob, add_random_noise!; noise_amplitude=0.2)
    
#     # Save initial state
#     # save_simulation_state_full("initial_state.jld2", prob; 
#     #     save_spectral=true, save_metadata=true, compress=true)
    
#     # Main simulation loop with output
#     # target_times = 0.1:0.1:10.0
#     # for t_target in target_times
#     #     step_until!(prob, t_target)
#     #     process_all_outputs!(output_manager, prob)
#     #     
#     #     # Optional: save checkpoint every 5 time units
#     #     if t_target % 5.0 ≈ 0.0
#     #         checkpoint_file = "checkpoint_t$(t_target).jld2"
#     #         save_simulation_state_full(checkpoint_file, prob)
#     #     end
#     # end
    
#     # Create output summary
#     # create_output_manifest("production_run")
#     # process_output_series("production_run")
    
#     println("\nComplete workflow features:")
#     println("   Time-based output frequencies")
#     println("   Physical and spectral field storage")
#     println("   Automatic checkpoint/restart")
#     println("   Comprehensive metadata preservation")
#     println("   MPI-aware distributed data gathering")
#     println("   Compressed storage for efficiency")
#     println("   Cross-platform data compatibility")
#     println("   Batch processing and analysis tools")
#     println("   Output manifest for reproducibility")
#     println("   Complete validation and error checking")
    
#     return true
# end



# """
# Example: Loading and analyzing saved spectral data
# """
# function demo_spectral_analysis()
#     println(" Spectral Data Analysis Demo")
#     println("=" ^ 35)
    
#     # Example: Load spectral snapshot
#     # jldopen("spectral_0001_t5.0000.jld2", "r") do file
#     #     bhat = file["buoyancy_hat"]
#     #     φhat = file["streamfunction_hat"]
#     #     energy_spec = file["energy_spectrum"]
#     #     enstrophy_spec = file["enstrophy_spectrum"]
#     #     kx = file["kx"]
#     #     ky = file["ky"]
#     #     
#     #     # Analysis
#     #     total_energy = sum(energy_spec)
#     #     total_enstrophy = sum(enstrophy_spec)
#     #     peak_energy_wavenumber = argmax(energy_spec) - 1
#     #     
#     #     println("Total energy: $total_energy")
#     #     println("Total enstrophy: $total_enstrophy")
#     #     println("Peak energy at k = $peak_energy_wavenumber")
#     #     
#     #     # Spectral analysis
#     #     k_cutoff = length(energy_spec) ÷ 3
#     #     large_scale_energy = sum(energy_spec[1:k_cutoff])
#     #     small_scale_energy = sum(energy_spec[k_cutoff+1:end])
#     #     
#     #     println("Large scale energy fraction: $(large_scale_energy/total_energy)")
#     # end
    
#     println("Spectral analysis capabilities:")
#     println("   Complete Fourier coefficients (real + imaginary)")
#     println("   Pre-computed energy and enstrophy spectra")
#     println("   Wavenumber arrays for proper scaling")
#     println("   Cross-platform compatibility (Julia, Python, MATLAB)")
#     println("   Compressed storage for large datasets")
#     println("   Time series of spectral evolution")
    
#     return true
# end


# """
# Example: Set up time-based output every 0.5 simulation time units
# """
# function demo_time_based_output()
#     println(" Time-Based Output Demo")
#     println("=" ^ 30)
    
#     # Example domain setup
#     # dom = Domain(256, 256, 2π, 2π, MPI.COMM_WORLD)
    
#     # Create output manager with time-based frequencies
#     # output_manager = OutputManager{Float64}("simulation_run";
#     #     snapshot_time_freq=0.5,      # Every 0.5 time units
#     #     full_state_time_freq=2.0,    # Every 2.0 time units  
#     #     spectral_time_freq=1.0,      # Every 1.0 time units
#     #     diagnostics_time_freq=0.1,   # Every 0.1 time units
#     #     save_spectral_data=true,
#     #     verbose_output=true)
    
#     # Example integration with time-based output
#     # prob = SemiGeostrophicProblem(dom; scheme=RK3, dt=0.01)
#     # set_initial_conditions!(prob, initialize_taylor_green!)
    
#     # for t_target in 0.5:0.5:10.0
#     #     step_until!(prob, t_target)
#     #     process_all_outputs!(output_manager, prob)
#     # end
    
#     println("Time-based output features:")
#     println("  Output every Δt simulation time (independent of time step)")
#     println("  Adaptive time stepping compatible")
#     println("  Configurable minimum/maximum time between saves")
#     println("  Automatic file naming with time stamps")
#     println("  Complete spectral field preservation")
#     println("  Derived spectral quantities (energy/enstrophy spectra)")
    
#     return true
# end
