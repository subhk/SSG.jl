# src/transforms.jl
# FFT operations and spectral derivatives

"""
    rfft!(dom::Domain, realfield, specfield)
Forward real FFT: real space → spectral space.
"""
function rfft!(dom::Domain, realfield, specfield)
    mul!(specfield, dom.fplan, realfield)
    return specfield
end

"""
    irfft!(dom::Domain, specfield, realfield)
Inverse real FFT: spectral space → real space.
"""
function irfft!(dom::Domain, specfield, realfield)
    mul!(realfield, dom.iplan, specfield)
    return realfield
end

"""
    dealias!(dom::Domain, Â)
Apply two-thirds dealiasing rule to spectral field.
"""
function dealias!(dom::Domain, Â)
    local_ranges = local_range(dom.pc)
    mask_local = view(dom.mask, local_ranges[1], local_ranges[2])
    @. Â = ifelse(mask_local, Â, 0)
    return Â
end

"""
    ddx!(dom::Domain, Â, out̂)
Spectral derivative ∂/∂x: multiply by ik_x.
"""
function ddx!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    kx_local = view(dom.kx, local_ranges[1])
    
    @inbounds for (i_local, i_global) in enumerate(local_ranges[1])
        kx = kx_local[i_local]
        @views out̂[i_local, :] = (im * kx) .* Â[i_local, :]
    end
    return out̂
end

"""
    ddy!(dom::Domain, Â, out̂)
Spectral derivative ∂/∂y: multiply by ik_y.
"""
function ddy!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    ky_local = view(dom.ky, local_ranges[2])
    
    @inbounds for (j_local, j_global) in enumerate(local_ranges[2])
        ky = ky_local[j_local]
        @views out̂[:, j_local] = (im * ky) .* Â[:, j_local]
    end
    return out̂
end

"""
    laplacian!(dom::Domain, Â, out̂)
Spectral Laplacian: multiply by -(k_x² + k_y²).
"""
function laplacian!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    kx_local = view(dom.kx, local_ranges[1])
    ky_local = view(dom.ky, local_ranges[2])
    
    @inbounds for (i_local, i_global) in enumerate(local_ranges[1])
        kx2 = kx_local[i_local]^2
        for (j_local, j_global) in enumerate(local_ranges[2])
            ky2 = ky_local[j_local]^2
            out̂[i_local, j_local] = -(kx2 + ky2) * Â[i_local, j_local]
        end
    end
    return out̂
end

"""
    d2dxdy!(dom::Domain, Â, out̂)
Mixed derivative ∂²/∂x∂y: multiply by -k_x k_y.
"""
function d2dxdy!(dom::Domain, Â, out̂)
    local_ranges = local_range(dom.pc)
    kx_local = view(dom.kx, local_ranges[1])
    ky_local = view(dom.ky, local_ranges[2])
    
    @inbounds for (i_local, i_global) in enumerate(local_ranges[1])
        kx = kx_local[i_local]
        for (j_local, j_global) in enumerate(local_ranges[2])
            ky = ky_local[j_local]
            out̂[i_local, j_local] = (-kx * ky) * Â[i_local, j_global]
        end
    end
    return out̂
end

"""
    gradient!(dom::Domain, A, Â, grad_x, grad_y, grad_x̂, grad_ŷ)
Compute gradient (∂A/∂x, ∂A/∂y) using spectral derivatives.
"""
function gradient!(dom::Domain, A, Â, grad_x, grad_y, grad_x̂, grad_ŷ)
    # Forward transform
    rfft!(dom, A, Â)
    
    # x-derivative
    ddx!(dom, Â, grad_x̂)
    irfft!(dom, grad_x̂, grad_x)
    
    # y-derivative
    ddy!(dom, Â, grad_ŷ)
    irfft!(dom, grad_ŷ, grad_y)
    
    return grad_x, grad_y
end

"""
    second_derivatives!(dom::Domain, A, Â, Axx, Ayy, Axy, tmpc)
Compute all second derivatives of field A.
"""
function second_derivatives!(dom::Domain, A, Â, Axx, Ayy, Axy, tmpc)
    # Forward transform
    rfft!(dom, A, Â)
    
    # ∂²A/∂x²
    ddx!(dom, Â, tmpc)
    ddx!(dom, tmpc, tmpc)
    irfft!(dom, tmpc, Axx)
    
    # ∂²A/∂y²
    rfft!(dom, A, Â)
    ddy!(dom, Â, tmpc)
    ddy!(dom, tmpc, tmpc)
    irfft!(dom, tmpc, Ayy)
    
    # ∂²A/∂x∂y
    rfft!(dom, A, Â)
    d2dxdy!(dom, Â, tmpc)
    irfft!(dom, tmpc, Axy)
    
    return Axx, Ayy, Axy
end