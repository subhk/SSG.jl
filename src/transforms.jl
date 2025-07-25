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
    apply_vertical_bc_ddz!(dom, A, out, bc, 1)
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
    apply_vertical_bc_ddz!(dom, A, out, bc, 2)
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
    apply_vertical_bc_ddz!(dom, A, out, bc, 3)
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
    
    # Boundary conditions
    if bc == :periodic
        # Handle periodic boundaries
        k = 1
        dzm = dz[Nz]
        dzp = dz[1]
        α = 2 / (dzm * (dzm + dzp))
        β = -2 / (dzm * dzp)
        γ = 2 / (dzp * (dzm + dzp))
        @views out_local[:, :, 1] = α .* A_local[:, :, Nz] + β .* A_local[:, :, 1] + γ .* A_local[:, :, 2]
        
        k = Nz
        dzm = dz[Nz-1]
        dzp = dz[1]  # wrap around
        α = 2 / (dzm * (dzm + dzp))
        β = -2 / (dzm * dzp)
        γ = 2 / (dzp * (dzm + dzp))
        @views out_local[:, :, Nz] = α .* A_local[:, :, Nz-1] + β .* A_local[:, :, Nz] + γ .* A_local[:, :, 1]
    else
        # Dirichlet or other BC: zero second derivative at boundaries
        @views out_local[:, :, 1] .= 0
        @views out_local[:, :, Nz] .= 0
    end
    
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