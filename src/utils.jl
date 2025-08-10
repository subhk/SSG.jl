# src/utils.jl
# Utility functions and macros for SSG solver

using PencilArrays: PencilArray, range_local

"""
    @ensuresamegrid(a, b)
Macro to check that two arrays have the same size.
"""
macro ensuresamegrid(a, b)
    :( @assert size($a) == size($b) "Grid mismatch: $(size($a)) vs $(size($b))" )
end

# twothirds_mask is defined in domain.jl

"""
    periodic_index(i::Int, n::Int) -> Int
Return periodic index: wraps i to be in range [1, n].
"""
periodic_index(i::Int, n::Int) = i < 1 ? n : (i > n ? 1 : i)

"""
    safe_divide(num, den; eps=1e-12)
Safe division that avoids division by zero.
"""
function safe_divide(num, den; eps=1e-12)
    if abs(den) < eps
        return num / (sign(den) * eps)
    else
        return num / den
    end
end

"""
    create_real_field(domain::Domain, ::Type{T}=FT) where T

Create a PencilArray for real-space fields.
"""
function create_real_field(domain::Domain, ::Type{T}=FT) where T
    return PencilArray{T}(undef, domain.pr3d)
end

"""
    create_spectral_field(domain::Domain, ::Type{T}=FT) where T

Create a PencilArray for spectral-space fields.
"""
function create_spectral_field(domain::Domain, ::Type{T}=FT) where T
    return PencilArray{Complex{T}}(undef, domain.pc3d)
end

"""
    copy_field!(dest, src)

Copy one PencilArray to another.
"""
function copy_field!(dest, src)
    dest_local = dest.data
    src_local = src.data
    @. dest_local = src_local
    #return dest
end

"""
    zero_field!(field)

Set all values in a PencilArray to zero.
"""
function zero_field!(field)
    fill!(field.data, zero(eltype(field)))
    #return field
end

"""
    norm_field(field; p=2)

Compute the norm of a PencilArray field across all MPI processes.
"""
function norm_field(field; p=2)
    field_local = field.data
    local_norm = norm(field_local, p)
    
    # MPI reduction to get global norm
    if p == 2
        global_norm_sq = MPI.Allreduce(local_norm^2, MPI.SUM, field.pencil.comm)
        return sqrt(global_norm_sq)
    elseif p == Inf
        return MPI.Allreduce(local_norm, MPI.MAX, field.pencil.comm)
    elseif p == 1
        return MPI.Allreduce(local_norm, MPI.SUM, field.pencil.comm)
    else
        global_norm_p = MPI.Allreduce(local_norm^p, MPI.SUM, field.pencil.comm)
        return global_norm_p^(1/p)
    end
end


"""
    inner_product(field1, field2)

Compute the inner product of two PencilArray fields across all MPI processes.
"""
function inner_product(field1, field2)
    local_dot = dot(field1.data, field2.data)
    return MPI.Allreduce(local_dot, MPI.SUM, field1.pencil.comm)
end

# =============================================================================
# GRID UTILITY FUNCTIONS
# =============================================================================

"""
    gridpoints(domain::Domain) -> (X, Y, Z)

Return 3D coordinate arrays for the domain.
"""
function gridpoints(domain::Domain)
    X = [domain.x[i] for i=1:domain.Nx, j=1:domain.Ny, k=1:domain.Nz]
    Y = [domain.y[j] for i=1:domain.Nx, j=1:domain.Ny, k=1:domain.Nz]
    Z = [domain.z[k] for i=1:domain.Nx, j=1:domain.Ny, k=1:domain.Nz]
    return X, Y, Z
end

"""
    gridpoints_2d(domain::Domain) -> (X, Y)

Return 2D horizontal coordinate arrays for the domain.
"""
function gridpoints_2d(domain::Domain)
    X = [domain.x[i] for i=1:domain.Nx, j=1:domain.Ny]
    Y = [domain.y[j] for i=1:domain.Nx, j=1:domain.Ny] 
    return X, Y
end


# =============================================================================
# SPECTRAL ANALYSIS FUNCTIONS
# =============================================================================

"""
    parsevalsum2(uh, domain::Domain)

Return the sum of `|uh|²` on the domain, which equals the domain integral of `u²`.
For a 3D domain with horizontal spectral representation:

```math
\\sum_{𝐤} |û_{𝐤}|² L_x L_y L_z = \\int u² 𝖽x 𝖽y 𝖽z
```

When the input `uh` comes from a real-FFT transform, `parsevalsum2` takes care to
count the contribution from certain k-modes twice to account for conjugate symmetry.
"""
function parsevalsum2(uh, domain::Domain)
    # Get local array from PencilArray
    uh_local = uh.data
    range_locals = range_local(uh.pencil)
    
    # Initialize local sum
    local_sum = 0.0
    
    # Handle real FFT conjugate symmetry
    if size(uh_local, 2) == length(domain.ky)  # Real FFT case
        
        # Sum over all z levels
        for k in axes(uh_local, 3)
            # k = 0 modes (count once)
            if 1 in range_locals[1]
                i_local = findfirst(x -> x == 1, range_locals[1])
                local_sum += sum(abs2, @view uh_local[i_local, :, k])
            end
            
            # k = nx/2 modes (count once)
            if domain.Nx÷2 + 1 in range_locals[1]
                i_local = findfirst(x -> x == domain.Nx÷2 + 1, range_locals[1])
                local_sum += sum(abs2, @view uh_local[i_local, :, k])
            end
            
            # 0 < k < nx/2 modes (count twice for conjugate symmetry)
            for (i_local, i_global) in enumerate(range_locals[1])
                if 1 < i_global < domain.Nx÷2 + 1
                    local_sum += 2 * sum(abs2, @view uh_local[i_local, :, k])
                end
            end
        end
        
    else  # Full complex FFT case
        local_sum = sum(abs2, uh_local)
    end
    
    # MPI reduction to get global sum
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, uh.pencil.comm)
    
    # Normalization for DFT
    normalization = (domain.Lx * domain.Ly * domain.Lz) / (domain.Nx^2 * domain.Ny^2 * domain.Nz)
    
    return global_sum * normalization
end

"""
    parsevalsum(uh, domain::Domain)

Return the real part of the sum of `uh` on the domain. For a 3D domain:

```math
ℜ [ \\sum_{𝐤} û_{𝐤} L_x L_y L_z ]
```

When the input `uh` comes from a real-FFT transform, `parsevalsum` accounts for
conjugate symmetry by counting certain k-modes twice.
"""
function parsevalsum(uh, domain::Domain)
    # Get local array from PencilArray
    uh_local = uh.data
    range_locals = range_local(uh.pencil)
    
    # Initialize local sum
    local_sum = 0.0 + 0.0im
    
    # Handle real FFT conjugate symmetry
    if size(uh_local, 2) == length(domain.ky)  # Real FFT case
        
        # Sum over all z levels
        for k in axes(uh_local, 3)
            # k = 0 modes (count once)
            if 1 in range_locals[1]
                i_local = findfirst(x -> x == 1, range_locals[1])
                local_sum += sum(@view uh_local[i_local, :, k])
            end
            
            # k = nx/2 modes (count once)
            if domain.Nx÷2 + 1 in range_locals[1]
                i_local = findfirst(x -> x == domain.Nx÷2 + 1, range_locals[1])
                local_sum += sum(@view uh_local[i_local, :, k])
            end
            
            # 0 < k < nx/2 modes (count twice for conjugate symmetry)
            for (i_local, i_global) in enumerate(range_locals[1])
                if 1 < i_global < domain.Nx÷2 + 1
                    local_sum += 2 * sum(@view uh_local[i_local, :, k])
                end
            end
        end
        
    else  # Full complex FFT case
        local_sum = sum(uh_local)
    end
    
    # MPI reduction to get global sum
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, uh.pencil.comm)
    
    # Normalization for DFT
    normalization = (domain.Lx * domain.Ly * domain.Lz) / (domain.Nx^2 * domain.Ny^2 * domain.Nz)
    
    return real(global_sum * normalization)
end

# =============================================================================
# JACOBIAN FUNCTIONS FOR NONLINEAR TERMS
# =============================================================================

"""
    jacobianh(a, b, domain::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)

Return the Fourier transform of the horizontal Jacobian of `a` and `b`:

```math
J(a, b) = \\frac{∂a}{∂x} \\frac{∂b}{∂y} - \\frac{∂a}{∂y} \\frac{∂b}{∂x}
```

This is computed in spectral space for efficiency. The function uses scratch arrays
to avoid allocation and is compatible with PencilArrays.
"""
function jacobianh(a, b, domain::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)
    # Transform b to spectral space using PencilFFTs
    rfft!(domain, b, tmp_spec1)  # tmp_spec1 = b̂
    
    # Compute ∂b/∂x using transforms module
    ddx!(domain, tmp_spec1, tmp_spec2)  # tmp_spec2 = ik_x * b̂
    irfft!(domain, tmp_spec2, tmp_real1)  # tmp_real1 = ∂b/∂x
    
    # Compute ∂b/∂y using transforms module
    ddy!(domain, tmp_spec1, tmp_spec2)  # tmp_spec2 = ik_y * b̂
    irfft!(domain, tmp_spec2, tmp_real2)  # tmp_real2 = ∂b/∂y
    
    # Compute a * ∂b/∂y and a * ∂b/∂x (work with local arrays)
    a_local = a.data
    bx_local = tmp_real1.data
    by_local = tmp_real2.data
    
    @. bx_local = a_local * bx_local  # a * ∂b/∂x
    @. by_local = a_local * by_local  # a * ∂b/∂y
    
    # Transform back to spectral space and take derivatives
    rfft!(domain, tmp_real2, tmp_spec1)  # tmp_spec1 = F[a * ∂b/∂y]
    ddx!(domain, tmp_spec1, tmp_spec2)   # tmp_spec2 = ik_x * F[a * ∂b/∂y]
    
    rfft!(domain, tmp_real1, tmp_spec1)  # tmp_spec1 = F[a * ∂b/∂x]
    ddy!(domain, tmp_spec1, tmp_spec1)   # tmp_spec1 = ik_y * F[a * ∂b/∂x]
    
    # Compute Jacobian: ∂(a∂b/∂y)/∂x - ∂(a∂b/∂x)/∂y
    jac_local = tmp_spec2.data
    term2_local = tmp_spec1.data
    @. jac_local = jac_local - term2_local
    
    return tmp_spec2
end

"""
    jacobian(a, b, domain::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2, output)

Compute the horizontal Jacobian of `a` and `b` in physical space:

```math
J(a, b) = \\frac{∂a}{∂x} \\frac{∂b}{∂y} - \\frac{∂a}{∂y} \\frac{∂b}{∂x}
```

The result is stored in `output`. Compatible with PencilArrays.
"""
function jacobian(a, b, domain::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2, output)
    # Compute Jacobian in spectral space
    jac_spec = jacobianh(a, b, domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)
    
    # Transform back to physical space using PencilFFTs
    irfft!(domain, jac_spec, output)
    
    return output
end

# advection_term! and vorticity_advection! functions removed

# =============================================================================
# ENERGY AND ENSTROPHY CALCULATIONS
# =============================================================================

"""
    compute_enstrophy(ω, domain::Domain) -> Float64

Compute the total enstrophy (0.5 * ∫ω² dA) of the flow.
This is a conserved quantity for inviscid 2D flows.
Uses PencilFFTs for spectral transform.
"""
function compute_enstrophy(ω, domain::Domain)
    # Create temporary spectral field
    ω_spec = create_spectral_field(domain)
    
    # Transform to spectral space using PencilFFTs
    rfft!(domain, ω, ω_spec)
    
    # Compute enstrophy using Parseval's theorem
    enstrophy = 0.5 * parsevalsum2(ω_spec, domain)
    
    return enstrophy
end

"""
    compute_energy(u, v, domain::Domain) -> Float64

Compute the total kinetic energy (0.5 * ∫(u² + v²) dA) of the flow.
Uses PencilFFTs for spectral transforms.
"""
function compute_energy(u, v, domain::Domain)
    # Create temporary spectral fields
    u_spec = create_spectral_field(domain)
    v_spec = create_spectral_field(domain)
    
    # Transform to spectral space using PencilFFTs
    rfft!(domain, u, u_spec)
    rfft!(domain, v, v_spec)
    
    # Compute energy using Parseval's theorem
    energy = 0.5 * (parsevalsum2(u_spec, domain) + parsevalsum2(v_spec, domain))
    
    return energy
end

"""
    compute_total_buoyancy(b, domain::Domain) -> Float64

Compute the total buoyancy integral ∫b dA.
This should be conserved in the absence of diabatic forcing.
Uses PencilFFTs for spectral transform.
"""
function compute_total_buoyancy(b, domain::Domain)
    # Create temporary spectral field
    b_spec = create_spectral_field(domain)
    
    # Transform to spectral space using PencilFFTs
    rfft!(domain, b, b_spec)
    
    # Compute integral using Parseval's theorem
    total_b = parsevalsum(b_spec, domain)
    
    return total_b
end

"""
    compute_buoyancy_variance(b, domain::Domain) -> Float64

Compute the buoyancy variance ∫b² dA.
This measures the strength of buoyancy gradients.
Uses PencilFFTs for spectral transform.
"""
function compute_buoyancy_variance(b, domain::Domain)
    # Create temporary spectral field
    b_spec = create_spectral_field(domain)
    
    # Transform to spectral space using PencilFFTs
    rfft!(domain, b, b_spec)
    
    # Compute variance using Parseval's theorem
    variance = parsevalsum2(b_spec, domain)
    
    return variance
end

# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

"""
    compute_cfl_number(u, v, domain::Domain, dt::Real) -> Float64

Compute the maximum CFL number for the current velocity field.
"""
function compute_cfl_number(u, v, domain::Domain, dt::Real)
    u_local = u.data
    v_local = v.data
    
    # Local maximum velocity
    u_max_local = maximum(abs.(u_local))
    v_max_local = maximum(abs.(v_local))
    vel_max_local = max(u_max_local, v_max_local)
    
    # Global maximum across all processes
    vel_max_global = MPI.Allreduce(vel_max_local, MPI.MAX, u.pencil.comm)
    
    # Grid spacing
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    h_min = min(dx, dy)
    
    # CFL number
    cfl = vel_max_global * dt / h_min
    
    return cfl
end

"""
    compute_spectral_diagnostics(field_spec, domain::Domain) -> NamedTuple

Compute spectral diagnostics including energy in different wavenumber bands.
"""
function compute_spectral_diagnostics(field_spec, domain::Domain)
    field_local = field_spec.data
    range_locals = range_local(field_spec.pencil)
    
    # Initialize energy counters for different scales
    large_scale_energy = 0.0   # Low wavenumbers
    small_scale_energy = 0.0   # High wavenumbers
    total_energy = 0.0
    
    # Define scale separation (can be adjusted)
    k_cutoff = min(domain.Nx, length(domain.ky)) ÷ 3
    
    for k in axes(field_local, 3)
        for (j_local, j_global) in enumerate(range_locals[2])
            for (i_local, i_global) in enumerate(range_locals[1])
                energy_density = abs2(field_local[i_local, j_local, k])
                
                # Apply conjugate symmetry factor for real FFT
                if size(field_local, 1) == domain.Nx && 1 < i_global < domain.Nx÷2 + 1
                    energy_density *= 2
                end
                
                total_energy += energy_density
                
                # Classify by scale
                k_mag = sqrt(domain.kx[i_global]^2 + domain.ky[j_global]^2)
                if k_mag < k_cutoff * 2π / max(domain.Lx, domain.Ly)
                    large_scale_energy += energy_density
                else
                    small_scale_energy += energy_density
                end
            end
        end
    end
    
    # MPI reductions
    total_energy = MPI.Allreduce(total_energy, MPI.SUM, field_spec.pencil.comm)
    large_scale_energy = MPI.Allreduce(large_scale_energy, MPI.SUM, field_spec.pencil.comm)
    small_scale_energy = MPI.Allreduce(small_scale_energy, MPI.SUM, field_spec.pencil.comm)
    
    # Normalization
    norm_factor = (domain.Lx * domain.Ly * domain.Lz) / (domain.Nx^2 * domain.Ny^2 * domain.Nz)
    
    return (
        total_energy = total_energy * norm_factor,
        large_scale_energy = large_scale_energy * norm_factor,
        small_scale_energy = small_scale_energy * norm_factor,
        scale_ratio = small_scale_energy / max(large_scale_energy, 1e-16)
    )
end

"""
    print_conservation_summary(domain::Domain, fields::Fields; step::Int=0, time::Real=0.0)

Print a summary of conserved quantities for monitoring simulation health.
"""
function print_conservation_summary(domain::Domain, fields::Fields; step::Int=0, time::Real=0.0)
    if MPI.Comm_rank(domain.pr3d.comm) == 0
        # Compute conserved quantities
        energy = compute_energy(fields.u, fields.v, domain)
        # Compute enstrophy from velocity field (simplified placeholder)
        enstrophy = 0.0  # TODO: implement proper enstrophy calculation from velocities
        total_buoyancy = compute_total_buoyancy(fields.bₛ, domain)
        
        println("=" ^60)
        println("Conservation Summary - Step: $step, Time: $(round(time, digits=4))")
        println("  Kinetic Energy : $(round(energy, sigdigits=8))")
        println("  Enstrophy     : $(round(enstrophy, sigdigits=8))")
        println("  Total Buoyancy: $(round(total_buoyancy, sigdigits=8))")
        println("=" ^60)
    end
    
    return nothing
end