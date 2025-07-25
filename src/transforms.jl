# src/transforms.jl
# FFT operations and spectral derivatives for 3D domain
# Horizontal directions: spectral (FFT)
# Vertical direction: finite differences

"""
    rfft!(dom::Domain, realfield, specfield)

Forward real FFT: real space → spectral space (horizontal directions only).
"""
function rfft!(dom::Domain, realfield, specfield)
    mul!(specfield, dom.fplan, realfield)
    return specfield
end

"""
    irfft!(dom::Domain, specfield, realfield)

Inverse real FFT: spectral space → real space (horizontal directions only).
"""
function irfft!(dom::Domain, specfield, realfield)
    mul!(realfield, dom.iplan, specfield)
    return realfield
end

"""
    dealias!(dom::Domain, Â)

Apply two-thirds dealiasing rule to spectral field (horizontal directions only).
"""
function dealias!(dom::Domain, Â)
    local_ranges = local_range(dom.pc)
    mask_local = view(dom.mask, local_ranges[1], local_ranges[2])
    
    # Get local array from PencilArray
    Â_local = Â.data
    
    @inbounds for k in axes(Â_local, 3)
        @views @. Â_local[:, :, k] = ifelse(mask_local, Â_local[:, :, k], 0)
    end
    return Â
end

# =============================================================================
# HORIZONTAL SPECTRAL DERIVATIVES
# =============================================================================

"""
    ddx!(dom::Domain, Â, out̂)

Spectral derivative ∂/∂x: multiply by ik_x (all z levels).
"""
function ddx!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    kx_local = view(dom.kx, local_ranges[1])
    
    # Get local arrays from PencilArrays
    Â_local = Â.data
    out̂_local = out̂.data
    
    @inbounds for k in axes(Â_local, 3)
        for (i_local, i_global) in enumerate(local_ranges[1])
            kx = kx_local[i_local]
            @views out̂_local[i_local, :, k] = (im * kx) .* Â_local[i_local, :, k]
        end
    end
    return out̂
end

"""
    ddy!(dom::Domain, Â, out̂)

Spectral derivative ∂/∂y: multiply by ik_y (all z levels).
"""
function ddy!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    ky_local = view(dom.ky, local_ranges[2])
    
    # Get local arrays from PencilArrays
    Â_local = Â.data
    out̂_local = out̂.data
    
    @inbounds for k in axes(Â_local, 3)
        for (j_local, j_global) in enumerate(local_ranges[2])
            ky = ky_local[j_local]
            @views out̂_local[:, j_local, k] = (im * ky) .* Â_local[:, j_local, k]
        end
    end
    return out̂
end

"""
    laplacian_h!(dom::Domain, Â, out̂)

Horizontal Laplacian: multiply by -(k_x² + k_y²) (all z levels).
"""
function laplacian_h!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    kx_local = view(dom.kx, local_ranges[1])
    ky_local = view(dom.ky, local_ranges[2])
    
    @inbounds for k in axes(Â, 3)
        for (i_local, i_global) in enumerate(local_ranges[1])
            kx2 = kx_local[i_local]^2
            for (j_local, j_global) in enumerate(local_ranges[2])
                ky2 = ky_local[j_local]^2
                out̂[i_local, j_local, k] = -(kx2 + ky2) * Â[i_local, j_local, k]
            end
        end
    end
    return out̂
end

"""
    d2dxdy!(dom::Domain, Â, out̂)

Mixed horizontal derivative ∂²/∂x∂y: multiply by -k_x k_y (all z levels).
"""
function d2dxdy!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    kx_local = view(dom.kx, local_ranges[1])
    ky_local = view(dom.ky, local_ranges[2])
    
    @inbounds for k in axes(Â, 3)
        for (i_local, i_global) in enumerate(local_ranges[1])
            kx = kx_local[i_local]
            for (j_local, j_global) in enumerate(local_ranges[2])
                ky = ky_local[j_local]
                out̂[i_local, j_local, k] = (-kx * ky) * Â[i_local, j_local, k]
            end
        end
    end
    return out̂
end

# =============================================================================
# MISSING SPECTRAL DERIVATIVES (IMPLEMENTING FROM UTILS.JL)
# =============================================================================

"""
    parsevalsum2(uh, dom::Domain)

Return the sum of `|uh|²` on the domain, which equals the domain integral of `u²`.
For a 3D domain with horizontal spectral representation:

```math
\\sum_{𝐤} |û_{𝐤}|² L_x L_y L_z = \\int u² 𝖽x 𝖽y 𝖽z
```

When the input `uh` comes from a real-FFT transform, `parsevalsum2` takes care to
count the contribution from certain k-modes twice to account for conjugate symmetry.
"""
function parsevalsum2(uh, dom::Domain)
    # Get local array from PencilArray
    uh_local = uh.data
    local_ranges = local_range(uh.pencil)
    
    # Initialize local sum
    local_sum = 0.0
    
    # Handle real FFT conjugate symmetry
    if size(uh_local, 2) == length(dom.ky)  # Real FFT case
        
        # Sum over all z levels
        for k in axes(uh_local, 3)
            # k = 0 modes (count once)
            if 1 in local_ranges[1]
                i_local = findfirst(x -> x == 1, local_ranges[1])
                local_sum += sum(abs2, @view uh_local[i_local, :, k])
            end
            
            # k = nx/2 modes (count once)
            if dom.Nx÷2 + 1 in local_ranges[1]
                i_local = findfirst(x -> x == dom.Nx÷2 + 1, local_ranges[1])
                local_sum += sum(abs2, @view uh_local[i_local, :, k])
            end
            
            # 0 < k < nx/2 modes (count twice for conjugate symmetry)
            for (i_local, i_global) in enumerate(local_ranges[1])
                if 1 < i_global < dom.Nx÷2 + 1
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
    normalization = (dom.Lx * dom.Ly * dom.Lz) / (dom.Nx^2 * dom.Ny^2 * dom.Nz)
    
    return global_sum * normalization
end

"""
    parsevalsum(uh, dom::Domain)

Return the real part of the sum of `uh` on the domain. For a 3D domain:

```math
ℜ [ \\sum_{𝐤} û_{𝐤} L_x L_y L_z ]
```

When the input `uh` comes from a real-FFT transform, `parsevalsum` accounts for
conjugate symmetry by counting certain k-modes twice.
"""
function parsevalsum(uh, dom::Domain)
    # Get local array from PencilArray
    uh_local = uh.data
    local_ranges = local_range(uh.pencil)
    
    # Initialize local sum
    local_sum = 0.0 + 0.0im
    
    # Handle real FFT conjugate symmetry
    if size(uh_local, 2) == length(dom.ky)  # Real FFT case
        
        # Sum over all z levels
        for k in axes(uh_local, 3)
            # k = 0 modes (count once)
            if 1 in local_ranges[1]
                i_local = findfirst(x -> x == 1, local_ranges[1])
                local_sum += sum(@view uh_local[i_local, :, k])
            end
            
            # k = nx/2 modes (count once)
            if dom.Nx÷2 + 1 in local_ranges[1]
                i_local = findfirst(x -> x == dom.Nx÷2 + 1, local_ranges[1])
                local_sum += sum(@view uh_local[i_local, :, k])
            end
            
            # 0 < k < nx/2 modes (count twice for conjugate symmetry)
            for (i_local, i_global) in enumerate(local_ranges[1])
                if 1 < i_global < dom.Nx÷2 + 1
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
    normalization = (dom.Lx * dom.Ly * dom.Lz) / (dom.Nx^2 * dom.Ny^2 * dom.Nz)
    
    return real(global_sum * normalization)
end

"""
    jacobianh(a, b, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)

Return the Fourier transform of the horizontal Jacobian of `a` and `b`:

```math
J(a, b) = \\frac{∂a}{∂x} \\frac{∂b}{∂y} - \\frac{∂a}{∂y} \\frac{∂b}{∂x}
```

This is computed in spectral space for efficiency. The function uses scratch arrays
to avoid allocation and is compatible with PencilArrays.
"""
function jacobianh(a, b, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)
    # Transform b to spectral space using PencilFFTs
    rfft!(dom, b, tmp_spec1)  # tmp_spec1 = b̂
    
    # Compute ∂b/∂x using transforms module
    ddx!(dom, tmp_spec1, tmp_spec2)  # tmp_spec2 = ik_x * b̂
    irfft!(dom, tmp_spec2, tmp_real1)  # tmp_real1 = ∂b/∂x
    
    # Compute ∂b/∂y using transforms module
    ddy!(dom, tmp_spec1, tmp_spec2)  # tmp_spec2 = ik_y * b̂
    irfft!(dom, tmp_spec2, tmp_real2)  # tmp_real2 = ∂b/∂y
    
    # Compute a * ∂b/∂y and a * ∂b/∂x (work with local arrays)
    a_local = a.data
    bx_local = tmp_real1.data
    by_local = tmp_real2.data
    
    @. bx_local = a_local * bx_local  # a * ∂b/∂x
    @. by_local = a_local * by_local  # a * ∂b/∂y
    
    # Transform back to spectral space and take derivatives
    rfft!(dom, tmp_real2, tmp_spec1)  # tmp_spec1 = F[a * ∂b/∂y]
    ddx!(dom, tmp_spec1, tmp_spec2)   # tmp_spec2 = ik_x * F[a * ∂b/∂y]
    
    rfft!(dom, tmp_real1, tmp_spec1)  # tmp_spec1 = F[a * ∂b/∂x]
    ddy!(dom, tmp_spec1, tmp_spec1)   # tmp_spec1 = ik_y * F[a * ∂b/∂x]
    
    # Compute Jacobian: ∂(a∂b/∂y)/∂x - ∂(a∂b/∂x)/∂y
    jac_local = tmp_spec2.data
    term2_local = tmp_spec1.data
    @. jac_local = jac_local - term2_local
    
    return tmp_spec2
end

"""
    jacobian(a, b, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2, output)

Compute the horizontal Jacobian of `a` and `b` in physical space:

```math
J(a, b) = \\frac{∂a}{∂x} \\frac{∂b}{∂y} - \\frac{∂a}{∂y} \\frac{∂b}{∂x}
```

The result is stored in `output`. Compatible with PencilArrays.
"""
function jacobian(a, b, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2, output)
    # Compute Jacobian in spectral space
    jac_spec = jacobianh(a, b, dom, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)
    
    # Transform back to physical space using PencilFFTs
    irfft!(dom, jac_spec, output)
    
    return output
end

"""
    gradient_h!(dom::Domain, field, field_spec, ∂x, ∂y, tmp_spec1, tmp_spec2)

Compute horizontal gradient of a field in physical space.
"""
function gradient_h!(dom::Domain, field, field_spec, ∂x, ∂y, tmp_spec1, tmp_spec2)
    # Transform to spectral space
    rfft!(dom, field, field_spec)
    
    # Compute ∂field/∂x
    ddx!(dom, field_spec, tmp_spec1)
    irfft!(dom, tmp_spec1, ∂x)
    
    # Compute ∂field/∂y
    ddy!(dom, field_spec, tmp_spec2)
    irfft!(dom, tmp_spec2, ∂y)
    
    return ∂x, ∂y
end

"""
    advection_term!(result, u, v, field, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)

Compute the advection term -u·∇field = -(u∂field/∂x + v∂field/∂y) for surface flows.

This is a common operation in surface semigeostrophic equations for computing
the advection of buoyancy or other scalars.
"""
function advection_term!(result, u, v, field, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)
    # Compute field gradients using the transforms module functions
    gradient_h!(dom, field, tmp_spec1, tmp_real1, tmp_real2, tmp_spec2, tmp_spec1)
    
    # tmp_real1 = ∂field/∂x, tmp_real2 = ∂field/∂y
    result_local = result.data
    u_local = u.data
    v_local = v.data
    fx_local = tmp_real1.data
    fy_local = tmp_real2.data
    
    # Compute advection: -(u∂field/∂x + v∂field/∂y)
    @. result_local = -(u_local * fx_local + v_local * fy_local)
    
    return result
end

"""
    vorticity_advection!(result, u, v, ω, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)

Compute the vorticity advection term for 2D flows: -u·∇ω.
This is used in vorticity-based formulations of the surface equations.
"""
function vorticity_advection!(result, u, v, ω, dom::Domain, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)
    return advection_term!(result, u, v, ω, dom, tmp_spec1, tmp_spec2, tmp_real1, tmp_real2)
end

"""
    compute_enstrophy(ω, dom::Domain) -> Float64

Compute the total enstrophy (0.5 * ∫ω² dA) of the flow.
This is a conserved quantity for inviscid 2D flows.
Uses PencilFFTs for spectral transform.
"""
function compute_enstrophy(ω, dom::Domain)
    # Create temporary spectral field
    ω_spec = create_spectral_field(dom)
    
    # Transform to spectral space using PencilFFTs
    rfft!(dom, ω, ω_spec)
    
    # Compute enstrophy using Parseval's theorem
    enstrophy = 0.5 * parsevalsum2(ω_spec, dom)
    
    return enstrophy
end

"""
    compute_energy(u, v, dom::Domain) -> Float64

Compute the total kinetic energy (0.5 * ∫(u² + v²) dA) of the flow.
Uses PencilFFTs for spectral transforms.
"""
function compute_energy(u, v, dom::Domain)
    # Create temporary spectral fields
    u_spec = create_spectral_field(dom)
    v_spec = create_spectral_field(dom)
    
    # Transform to spectral space using PencilFFTs
    rfft!(dom, u, u_spec)
    rfft!(dom, v, v_spec)
    
    # Compute energy using Parseval's theorem
    energy = 0.5 * (parsevalsum2(u_spec, dom) + parsevalsum2(v_spec, dom))
    
    return energy
end

"""
    compute_total_buoyancy(b, dom::Domain) -> Float64

Compute the total buoyancy integral ∫b dA.
This should be conserved in the absence of diabatic forcing.
Uses PencilFFTs for spectral transform.
"""
function compute_total_buoyancy(b, dom::Domain)
    # Create temporary spectral field
    b_spec = create_spectral_field(dom)
    
    # Transform to spectral space using PencilFFTs
    rfft!(dom, b, b_spec)
    
    # Compute integral using Parseval's theorem
    total_b = parsevalsum(b_spec, dom)
    
    return total_b
end

"""
    compute_buoyancy_variance(b, dom::Domain) -> Float64

Compute the buoyancy variance ∫b² dA.
This measures the strength of buoyancy gradients.
Uses PencilFFTs for spectral transform.
"""
function compute_buoyancy_variance(b, dom::Domain)
    # Create temporary spectral field
    b_spec = create_spectral_field(dom)
    
    # Transform to spectral space using PencilFFTs
    rfft!(dom, b, b_spec)
    
    # Compute variance using Parseval's theorem
    variance = parsevalsum2(b_spec, dom)
    
    return variance
end

# """
#     compute_cfl_number(u, v, dom::Domain, dt::Real) -> Float64

# Compute the maximum CFL number for the current velocity field.
# """
# function compute_cfl_number(u, v, dom::Domain, dt::Real)
#     u_local = u.data
#     v_local = v.data
    
#     # Local maximum velocity
#     u_max_local = maximum(abs.(u_local))
#     v_max_local = maximum(abs.(v_local))
#     vel_max_local = max(u_max_local, v_max_local)
    
#     # Global maximum across all processes
#     vel_max_global = MPI.Allreduce(vel_max_local, MPI.MAX, u.pencil.comm)
    
#     # Grid spacing
#     dx = dom.Lx / dom.Nx
#     dy = dom.Ly / dom.Ny
#     h_min = min(dx, dy)
    
#     # CFL number
#     cfl = vel_max_global * dt / h_min
    
#     return cfl
# end

# =============================================================================
# VERTICAL FINITE DIFFERENCE DERIVATIVES
# =============================================================================

"""
    ddz!(dom::Domain, A, out; order=2, bc=:default)

Vertical derivative ∂/∂z using finite differences.

# Arguments
- `A`: Input field (real space)
- `out`: Output field (real space)
- `order`: Finite difference order (2, 4, 6)
- `bc`: Boundary condition treatment (:default, :periodic, :extrapolate)
"""
function ddz!(dom::Domain, A, out; order=2, bc=:default)
    if bc == :default
        bc = dom.z_boundary == :periodic ? :periodic : :extrapolate
    end
    
    if order == 2
        ddz_o2!(dom, A, out, bc)
    elseif order == 4
        ddz_o4!(dom, A, out, bc)
    elseif order == 6
        ddz_o6!(dom, A, out, bc)
    else
        error("Unsupported finite difference order: $order. Use 2, 4, or 6.")
    end
    
    return out
end

"""
    ddz_o2!(dom::Domain, A, out, bc)

Second-order central difference for ∂/∂z.
"""
function ddz_o2!(dom::Domain, A, out, bc)
    Nz = dom.Nz
    dz = dom.dz
    
    # Get local arrays from PencilArrays
    A_local = A.data
    out_local = out.data
    
    @inbounds for k in 2:Nz-1
        dzm = dz[k-1]  # spacing below
        dzp = dz[k]    # spacing above
        α = 2 / (dzm * (dzm + dzp))
        β = 2 / (dzm * dzp)
        γ = 2 / (dzp * (dzm + dzp))
        
        @views out_local[:, :, k] = -α .* A_local[:, :, k-1] + β .* A_local[:, :, k] - γ .* A_local[:, :, k+1]
    end
    
    # Boundary conditions
    # apply_vertical_bc_ddz!(dom, A, out, bc, 1)
end

"""
    ddz_o4!(dom::Domain, A, out, bc)

Fourth-order central difference for ∂/∂z (uniform grid approximation).
"""
function ddz_o4!(dom::Domain, A, out, bc)
    Nz = dom.Nz
    dz = dom.dz
    
    # Get local arrays from PencilArrays
    A_local = A.data
    out_local = out.data
    
    # Interior points (assuming approximately uniform spacing)
    @inbounds for k in 3:Nz-2
        h = 0.25 * (dz[k-1] + dz[k] + dz[k+1] + dz[k+2])  # Average spacing
        @views out_local[:, :, k] = (A_local[:, :, k-2] - 8*A_local[:, :, k-1] + 8*A_local[:, :, k+1] - A_local[:, :, k+2]) / (12*h)
    end
    
    # Near-boundary points (fall back to second order)
    ddz_o2_single!(dom, A, out, 2)
    ddz_o2_single!(dom, A, out, Nz-1)
    
    # Boundary conditions
    # apply_vertical_bc_ddz!(dom, A, out, bc, 2)
end

"""
    ddz_o6!(dom::Domain, A, out, bc)

Sixth-order central difference for ∂/∂z (uniform grid approximation).
"""
function ddz_o6!(dom::Domain, A, out, bc)
    Nz = dom.Nz
    dz = dom.dz
    
    # Get local arrays from PencilArrays
    A_local = A.data
    out_local = out.data
    
    # Interior points
    @inbounds for k in 4:Nz-3
        h = (1/6) * sum(dz[k-2:k+3])  # Average spacing
        @views out_local[:, :, k] = (-A_local[:, :, k-3] + 9*A_local[:, :, k-2] - 45*A_local[:, :, k-1] + 
                               45*A_local[:, :, k+1] - 9*A_local[:, :, k+2] + A_local[:, :, k+3]) / (60*h)
    end
    
    # Near-boundary points (fall back to lower order)
    ddz_o2_single!(dom, A, out, 2)
    ddz_o2_single!(dom, A, out, 3)
    ddz_o2_single!(dom, A, out, Nz-2)
    ddz_o2_single!(dom, A, out, Nz-1)
    
    # Boundary conditions
    # apply_vertical_bc_ddz!(dom, A, out, bc, 3)
end

"""
    ddz_o2_single!(dom::Domain, A, out, k)

Apply second-order difference at a single z level.
"""
function ddz_o2_single!(dom::Domain, A, out, k)
    dz = dom.dz
    dzm = k > 1 ? dz[k-1] : dz[k]
    dzp = k < dom.Nz ? dz[k] : dz[k-1]
    
    # Get local arrays from PencilArrays
    A_local = A.data
    out_local = out.data
    
    if k == 1
        # Forward difference
        @views out_local[:, :, k] = (A_local[:, :, k+1] - A_local[:, :, k]) / dzp
    elseif k == dom.Nz
        # Backward difference
        @views out_local[:, :, k] = (A_local[:, :, k] - A_local[:, :, k-1]) / dzm
    else
        # Central difference
        α = 2 / (dzm * (dzm + dzp))
        β = 2 / (dzm * dzp)
        γ = 2 / (dzp * (dzm + dzp))
        @views out_local[:, :, k] = -α .* A_local[:, :, k-1] + β .* A_local[:, :, k] - γ .* A_local[:, :, k+1]
    end
end

"""
    apply_vertical_bc_ddz!(dom::Domain, A, out, bc, stencil_width)

Apply boundary conditions for vertical derivatives.
"""
function apply_vertical_bc_ddz!(dom::Domain, A, out, bc, stencil_width)
    if bc == :periodic
        # Handle periodic boundaries (wrap around)
        apply_periodic_bc_ddz!(dom, A, out, stencil_width)
    elseif bc == :extrapolate
        # Extrapolate to boundaries
        if dom.z_boundary == :dirichlet
            # For Dirichlet BC: set derivative at boundary to zero
            out_local = out.data
            @views out_local[:, :, 1] .= 0
            @views out_local[:, :, dom.Nz] .= 0
        else
            # Use one-sided differences
            ddz_o2_single!(dom, A, out, 1)
            ddz_o2_single!(dom, A, out, dom.Nz)
        end
    end
end

"""
    apply_periodic_bc_ddz!(dom::Domain, A, out, stencil_width)

Apply periodic boundary conditions for vertical derivatives.
"""
function apply_periodic_bc_ddz!(dom::Domain, A, out, stencil_width)
    Nz = dom.Nz
    dz = dom.dz
    
    # Get local arrays from PencilArrays
    A_local = A.data
    out_local = out.data
    
    # Assume uniform spacing for periodic case
    h = dz[1]
    
    for k in 1:stencil_width
        # Bottom boundary
        if stencil_width == 1  # Second order
            @views out_local[:, :, k] = (A_local[:, :, k+1] - A_local[:, :, Nz+k-1]) / (2*h)
        elseif stencil_width == 2  # Fourth order
            k1 = k == 1 ? Nz : k-1
            k2 = k+1 > Nz ? k+1-Nz : k+1
            k3 = k+2 > Nz ? k+2-Nz : k+2
            @views out_local[:, :, k] = (A_local[:, :, k1] - 8*A_local[:, :, k] + 8*A_local[:, :, k2] - A_local[:, :, k3]) / (12*h)
        end
        
        # Top boundary
        kt = Nz + 1 - k
        if stencil_width == 1  # Second order
            k_m1 = kt-1 < 1 ? Nz+kt-1 : kt-1
            @views out_local[:, :, kt] = (A_local[:, :, 1] - A_local[:, :, k_m1]) / (2*h)
        end
    end
end

"""
    d2dz2!(dom::Domain, A, out; order=2, bc=:default)

Second vertical derivative ∂²/∂z² using finite differences.
"""
function d2dz2!(dom::Domain, A, out; order=2, bc=:default)
    if bc == :default
        bc = dom.z_boundary == :periodic ? :periodic : :extrapolate
    end
    
    Nz = dom.Nz
    dz = dom.dz
    
    # Get local arrays from PencilArrays
    A_local = A.data
    out_local = out.data
    
    if order == 2
        # Second-order central difference
        @inbounds for k in 2:Nz-1
            dzm = dz[k-1]
            dzp = dz[k]
            α = 2 / (dzm * (dzm + dzp))
            β = -2 / (dzm * dzp)
            γ = 2 / (dzp * (dzm + dzp))
            
            @views out_local[:, :, k] = α .* A_local[:, :, k-1] + β .* A_local[:, :, k] + γ .* A_local[:, :, k+1]
        end
    else
        error("Higher order second derivatives not implemented yet")
    end
    
    # # Boundary conditions
    # if bc == :periodic
    #     # Handle periodic boundaries
    #     k = 1
    #     dzm = dz[Nz]
    #     dzp = dz[1]
    #     α = 2 / (dzm * (dzm + dzp))
    #     β = -2 / (dzm * dzp)
    #     γ = 2 / (dzp * (dzm + dzp))
    #     @views out_local[:, :, 1] = α .* A_local[:, :, Nz] + β .* A_local[:, :, 1] + γ .* A_local[:, :, 2]
        
    #     k = Nz
    #     dzm = dz[Nz-1]
    #     dzp = dz[1]  # wrap around
    #     α = 2 / (dzm * (dzm + dzp))
    #     β = -2 / (dzm * dzp)
    #     γ = 2 / (dzp * (dzm + dzp))
    #     @views out_local[:, :, Nz] = α .* A_local[:, :, Nz-1] + β .* A_local[:, :, Nz] + γ .* A_local[:, :, 1]
    # else
    #     # Dirichlet or other BC: zero second derivative at boundaries
    #     @views out_local[:, :, 1] .= 0
    #     @views out_local[:, :, Nz] .= 0
    # end
    
    return out
end

"""
    laplacian_3d!(dom::Domain, A, Â, lap, tmp_spec, tmp_real; fd_order=2)

Compute full 3D Laplacian: ∂²/∂x² + ∂²/∂y² + ∂²/∂z².
"""
function laplacian_3d!(dom::Domain, A, Â, lap, tmp_spec, tmp_real; fd_order=2)
    # Horizontal Laplacian (spectral)
    rfft!(dom, A, Â)
    laplacian_h!(dom, Â, tmp_spec)
    irfft!(dom, tmp_spec, lap)
    
    # Vertical second derivative (finite difference)
    d2dz2!(dom, A, tmp_real; order=fd_order)
    
    # Add them together
    lap_local = lap.data
    tmp_real_local = tmp_real.data
    @. lap_local += tmp_real_local
    
    return lap
end

"""
    divergence_3d!(dom::Domain, u, v, w, û, div, tmp_spec; fd_order=2)

Compute 3D divergence: ∂u/∂x + ∂v/∂y + ∂w/∂z.
"""
function divergence_3d!(dom::Domain, u, v, w, û, div, tmp_spec; fd_order=2)
    # ∂u/∂x (spectral)
    rfft!(dom, u, û)
    ddx!(dom, û, tmp_spec)
    irfft!(dom, tmp_spec, div)
    
    # ∂v/∂y (spectral)
    rfft!(dom, v, û)
    ddy!(dom, û, tmp_spec)
    irfft!(dom, tmp_spec, tmp_spec)  # Reuse tmp_spec as temp real array
    
    # Add ∂v/∂y to ∂u/∂x
    div_local = div.data
    tmp_spec_local = tmp_spec.data
    @. div_local += tmp_spec_local
    
    # ∂w/∂z (finite difference)
    ddz!(dom, w, tmp_spec; order=fd_order)
    @. div_local += tmp_spec_local
    
    return div
end

# =============================================================================
# UTILITY FUNCTIONS FOR PENCIL ARRAYS
# =============================================================================

"""
    create_real_field(dom::Domain, ::Type{T}=FT) where T

Create a PencilArray for real-space fields.
"""
function create_real_field(dom::Domain, ::Type{T}=FT) where T
    return PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
end

"""
    create_spectral_field(dom::Domain, ::Type{T}=FT) where T

Create a PencilArray for spectral-space fields.
"""
function create_spectral_field(dom::Domain, ::Type{T}=FT) where T
    return PencilArray(dom.pc, zeros(Complex{T}, local_size(dom.pc)))
end

"""
    copy_field!(dest, src)

Copy one PencilArray to another.
"""
function copy_field!(dest, src)
    dest_local = dest.data
    src_local = src.data
    @. dest_local = src_local
    return dest
end

"""
    zero_field!(field)

Set all values in a PencilArray to zero.
"""
function zero_field!(field)
    field_local = field.data
    fill!(field_local, 0)
    return field
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
    field1_local = field1.data
    field2_local = field2.data
    
    local_dot = dot(field1_local, field2_local)
    return MPI.Allreduce(local_dot, MPI.SUM, field1.pencil.comm)
end

# =============================================================================
# CONVENIENCE WRAPPERS FOR COMMON OPERATIONS
# =============================================================================

"""
    apply_operator!(op!, dom::Domain, input, output, args...; kwargs...)

Generic wrapper for applying operators that modify PencilArrays in-place.
"""
function apply_operator!(op!, dom::Domain, input, output, args...; kwargs...)
    return op!(dom, input, output, args...; kwargs...)
end

"""
    compute_kinetic_energy(dom::Domain, u, v, w, tmp_real)

Compute kinetic energy: 0.5 * (u² + v² + w²).
"""
function compute_kinetic_energy(dom::Domain, u, v, w, tmp_real)
    u_local = u.data
    v_local = v.data
    w_local = w.data
    tmp_local = tmp_real.data
    
    @. tmp_local = 0.5 * (u_local^2 + v_local^2 + w_local^2)
    
    return tmp_real
end

"""
    compute_vorticity_z!(dom::Domain, u, v, û, ω_z, tmp_spec)

Compute vertical vorticity: ∂v/∂x - ∂u/∂y.
"""
function compute_vorticity_z!(dom::Domain, u, v, û, ω_z, tmp_spec)
    # ∂v/∂x
    rfft!(dom, v, û)
    ddx!(dom, û, tmp_spec)
    irfft!(dom, tmp_spec, ω_z)
    
    # ∂u/∂y
    rfft!(dom, u, û)
    ddy!(dom, û, tmp_spec)
    irfft!(dom, tmp_spec, tmp_spec)  # Reuse as real array
    
    # ω_z = ∂v/∂x - ∂u/∂y
    ω_z_local = ω_z.data
    tmp_spec_local = tmp_spec.data
    @. ω_z_local -= tmp_spec_local
    
    return ω_z
end

# =============================================================================
# ENHANCED DIAGNOSTICS AND FIELD ANALYSIS
# =============================================================================

"""
    compute_spectral_diagnostics(field_spec, dom::Domain) -> NamedTuple

Compute spectral diagnostics including energy in different wavenumber bands.
"""
function compute_spectral_diagnostics(field_spec, dom::Domain)
    field_local = field_spec.data
    local_ranges = local_range(field_spec.pencil)
    
    # Initialize energy counters for different scales
    large_scale_energy = 0.0   # Low wavenumbers
    small_scale_energy = 0.0   # High wavenumbers
    total_energy = 0.0
    
    # Define scale separation (can be adjusted)
    k_cutoff = min(dom.Nx, length(dom.ky)) ÷ 3
    
    for k in axes(field_local, 3)
        for (j_local, j_global) in enumerate(local_ranges[2])
            for (i_local, i_global) in enumerate(local_ranges[1])
                energy_density = abs2(field_local[i_local, j_local, k])
                
                # Apply conjugate symmetry factor for real FFT
                if size(field_local, 1) == dom.Nx && 1 < i_global < dom.Nx÷2 + 1
                    energy_density *= 2
                end
                
                total_energy += energy_density
                
                # Classify by scale
                k_mag = sqrt(dom.kx[i_global]^2 + dom.ky[j_global]^2)
                if k_mag < k_cutoff * 2π / max(dom.Lx, dom.Ly)
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
    norm_factor = (dom.Lx * dom.Ly * dom.Lz) / (dom.Nx^2 * dom.Ny^2 * dom.Nz)
    
    return (
        total_energy = total_energy * norm_factor,
        large_scale_energy = large_scale_energy * norm_factor,
        small_scale_energy = small_scale_energy * norm_factor,
        scale_ratio = small_scale_energy / max(large_scale_energy, 1e-16)
    )
end

"""
    print_conservation_summary(dom::Domain, fields::Fields; step::Int=0, time::Real=0.0)

Print a summary of conserved quantities for monitoring simulation health.
"""
function print_conservation_summary(dom::Domain, fields::Fields; step::Int=0, time::Real=0.0)
    if MPI.Comm_rank(dom.pr.comm) == 0
        # Compute conserved quantities
        energy = compute_energy(fields.u, fields.v, dom)
        total_buoyancy = compute_total_buoyancy(fields.b, dom)
        
        println("=" ^60)
        println("Conservation Summary - Step: $step, Time: $(round(time, digits=4))")
        println("  Kinetic Energy : $(round(energy, sigdigits=8))")
        println("  Total Buoyancy: $(round(total_buoyancy, sigdigits=8))")
        println("=" ^60)
    end
    
    return nothing
end

# =============================================================================
# FIELD STATISTICS AND ANALYSIS
# =============================================================================

"""
    enhanced_field_stats(fields::Fields) -> Dict

Compute enhanced statistics for all fields in Fields structure.
"""
function enhanced_field_stats(fields::Fields)
    stats = Dict{Symbol, NamedTuple}()
    
    for field_name in fieldnames(typeof(fields))
        field = getfield(fields, field_name)
        if isa(field, PencilArray) && eltype(field) <: Real
            field_data = field.data
            
            # Local statistics
            local_mean = mean(field_data)
            local_var = var(field_data)
            local_min = minimum(field_data)
            local_max = maximum(field_data)
            local_count = length(field_data)
            
            # Global statistics via MPI
            global_sum = MPI.Allreduce(local_mean * local_count, MPI.SUM, field.pencil.comm)
            global_count = MPI.Allreduce(local_count, MPI.SUM, field.pencil.comm)
            global_mean = global_sum / global_count
            
            global_min = MPI.Allreduce(local_min, MPI.MIN, field.pencil.comm)
            global_max = MPI.Allreduce(local_max, MPI.MAX, field.pencil.comm)
            
            # Global variance (simplified)
            local_var_contrib = sum((field_data .- global_mean).^2)
            global_var = MPI.Allreduce(local_var_contrib, MPI.SUM, field.pencil.comm) / global_count
            global_std = sqrt(global_var)
            
            stats[field_name] = (
                mean = global_mean,
                std = global_std,
                min = global_min,
                max = global_max,
                count = global_count
            )
        end
    end
    
    return stats
end

"""
    field_stats(field) -> (mean, std, min, max)

Compute basic statistics for a single field (backward compatibility).
"""
function field_stats(field)
    if isa(field, PencilArray)
        field_data = field.data
        
        # Local statistics
        local_mean = mean(field_data)
        local_var = var(field_data)
        local_min = minimum(field_data)
        local_max = maximum(field_data)
        local_count = length(field_data)
        
        # Global statistics via MPI
        global_sum = MPI.Allreduce(local_mean * local_count, MPI.SUM, field.pencil.comm)
        global_count = MPI.Allreduce(local_count, MPI.SUM, field.pencil.comm)
        global_mean = global_sum / global_count
        
        global_min = MPI.Allreduce(local_min, MPI.MIN, field.pencil.comm)
        global_max = MPI.Allreduce(local_max, MPI.MAX, field.pencil.comm)
        
        # Simplified global std calculation
        local_var_contrib = sum((field_data .- global_mean).^2)
        global_var = MPI.Allreduce(local_var_contrib, MPI.SUM, field.pencil.comm) / global_count
        global_std = sqrt(global_var)
        
        return (global_mean, global_std, global_min, global_max)
    else
        # Regular array
        vals = Array(field)
        return (mean(vals), std(vals), minimum(vals), maximum(vals))
    end
end

# =============================================================================
# GRID UTILITY FUNCTIONS (FROM UTILS.JL)
# =============================================================================

"""
    gridpoints(dom::Domain) -> (X, Y, Z)

Return 3D coordinate arrays for the domain.
"""
function gridpoints(dom::Domain)
    X = [dom.x[i] for i=1:dom.Nx, j=1:dom.Ny, k=1:dom.Nz]
    Y = [dom.y[j] for i=1:dom.Nx, j=1:dom.Ny, k=1:dom.Nz]
    Z = [dom.z[k] for i=1:dom.Nx, j=1:dom.Ny, k=1:dom.Nz]
    
    return X, Y, Z
end

"""
    gridpoints_2d(dom::Domain) -> (X, Y)

Return 2D horizontal coordinate arrays for the domain.
"""
function gridpoints_2d(dom::Domain)
    X = [dom.x[i] for i=1:dom.Nx, j=1:dom.Ny]
    Y = [dom.y[j] for i=1:dom.Nx, j=1:dom.Ny]
    
    return X, Y
end

# =============================================================================
# COMPATIBILITY AND UTILITY MACROS
# =============================================================================

"""
    @ensuresamegrid(a, b)

Macro to check that two arrays have the same size.
"""
macro ensuresamegrid(a, b)
    quote
        if size($(esc(a))) != size($(esc(b)))
            throw(ArgumentError("Grid mismatch: $(size($(esc(a)))) vs $(size($(esc(b))))"))
        end
    end
end

"""
    ensure_same_grid(dest::PencilArray, src::PencilArray)

Function version of grid compatibility check.
"""
function ensure_same_grid(dest::PencilArray, src::PencilArray)
    # Check compatible sizes
    if size(dest) != size(src)
        throw(ArgumentError("PencilArrays have incompatible sizes: $(size(dest)) vs $(size(src))"))
    end
    
    # Check MPI communicator compatibility
    if dest.pencil.comm != src.pencil.comm
        throw(ArgumentError("PencilArrays have different MPI communicators"))
    end
    
    return true
end