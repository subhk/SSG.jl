# src/utils.jl
# Utility functions and macros for SSG solver

"""
    @ensuresamegrid(a, b)
Macro to check that two arrays have the same size.
"""
macro ensuresamegrid(a, b)
    :( @assert size($a) == size($b) "Grid mismatch: $(size($a)) vs $(size($b))" )
end

"""
    twothirds_mask(nx::Int, ny::Int) -> BitMatrix
Create a two-thirds dealiasing mask for spectral computations.
Zeros out the highest 1/3 of wavenumbers in each direction.
"""
function twothirds_mask(nx::Int, ny::Int)
    kx_cut = fld(nx, 3)
    ky_cut = fld(ny, 3)
    mask = ones(Bool, nx, ny)
    
    @inbounds for i in 1:nx
        for j in 1:ny
            if (i-1) > kx_cut && (i-1) < nx-kx_cut || 
               (j-1) > ky_cut && (j-1) < ny-ky_cut
                mask[i, j] = false
            end
        end
    end
    return mask
end

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