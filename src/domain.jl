# src/domain.jl
# 3D Domain setup with vertically bounded domain
"""
    struct Domain{T, PA, PC, PF, PB}

Holds 3D grid information, wavenumbers, FFT plans, and pencil descriptors.
Periodic in x,y directions; bounded in z direction.

# Fields
- `Nx, Ny, Nz`: Grid dimensions
- `Lx, Ly, Lz`: Domain size
- `x, y, z`: Coordinate vectors
- `kx, ky`: Horizontal wavenumber vectors (periodic directions)
- `dz`: Vertical grid spacing (non-uniform if z_grid != :uniform)
- `Krsq`: Array with squared total wavenumbers for real FFTs: kr² + ky²
- `invKrsq`: Array with inverse squared total wavenumbers: 1/(kr² + ky²)
- `mask`: Dealiasing mask for horizontal directions
- `z_boundary`: Boundary condition type (:dirichlet, :neumann, :free_slip, etc.)
- `z_grid`: Vertical grid type (:uniform, :stretched, :custom)
- `pr`: Real-space pencil descriptor
- `pc`: Spectral-space pencil descriptor (for horizontal FFTs)
- `fplan, iplan`: Forward and inverse FFT plans (horizontal only)
- `aliased_fraction`: Fraction of wavenumbers that are aliased (e.g., 1/3)
- `kxalias, kyalias`: Ranges of aliased wavenumber indices
"""
struct Domain{T, PA<:AbstractPencil, PC<:AbstractPencil, PF, PB}
    Nx::Int
    Ny::Int
    Nz::Int

    Lx::T
    Ly::T
    Lz::T

    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    dz::Vector{T}  # Vertical grid spacing (can be non-uniform)

    kx::Vector{T}
    ky::Vector{T}
    Krsq::Matrix{T}        # kr² + ky² for real FFTs
    invKrsq::Matrix{T}     # 1/(kr² + ky²)

    mask::BitMatrix
    z_boundary::Symbol
    z_grid::Symbol
    
    # Pencil descriptors
    pr::PA # real-space pencil
    pc::PC # complex/spectral pencil
    
    # FFT plans (horizontal only)
    fplan::PF
    iplan::PB
    
    # Dealiasing parameters
    aliased_fraction::T
    kxalias::UnitRange{Int}
    kyalias::UnitRange{Int}
end

"""
    make_domain(Nx, Ny, Nz; Lx=2π, Ly=2π, Lz=1.0, z_boundary=:dirichlet, 
                z_grid=:uniform, stretch_params=nothing, comm=MPI.COMM_WORLD) -> Domain

Create a 3D domain with periodic horizontal directions and bounded vertical direction.

# Arguments
- `Nx, Ny, Nz`: Grid points in x, y, and z directions
- `Lx, Ly, Lz`: Domain size (default: 2π × 2π × 1.0)
- `z_boundary`: Boundary condition type (:dirichlet, :neumann, :free_slip, :periodic)
- `z_grid`: Vertical grid type (:uniform, :stretched, :custom)
- `stretch_params`: Parameters for grid stretching (NamedTuple with specific fields for each type)
- `comm`: MPI communicator (default: MPI.COMM_WORLD)

# Stretch Parameters
For `z_grid = :stretched`, provide `stretch_params` as:
- `(type=:tanh, β=2.0)`: Hyperbolic tangent stretching
- `(type=:sinh, β=2.0)`: Hyperbolic sine stretching
- `(type=:power, α=1.5)`: Power law stretching
- `(type=:exponential, β=2.0)`: Exponential clustering

For `z_grid = :custom`, provide:
- `(z_coords=custom_z_vector)`: Direct specification of z coordinates

# Returns
- `Domain`: Configured 3D domain structure
"""
function make_domain(Nx::Int, Ny::Int, Nz::Int; 
                     Lx=2π, Ly=2π, Lz=1.0, 
                     z_boundary=:dirichlet,
                     z_grid=:uniform,
                     stretch_params=nothing,
                     comm=MPI.COMM_WORLD)
    
    # Initialize MPI if needed
    MPI.Initialized() || MPI.Init()
    
    # Create pencil descriptors for 3D arrays
    # Real-space: full (Nx, Ny, Nz) array
    pr = Pencil((Nx, Ny, Nz), comm)
    
    # Spectral-space: rFFT reduces y dimension to Ny÷2+1, z remains same
    Nyc = fld(Ny, 2) + 1
    pc = Pencil((Nx, Nyc, Nz), comm)
    
    # Create FFT plans using dummy arrays (only for horizontal directions)
    u_r = PencilArray(pr, zeros(FT, local_size(pr)))
    û_c = PencilArray(pc, zeros(Complex{FT}, local_size(pc)))
    
    # FFT only in horizontal directions (1,2), z remains in physical space
    fplan = PencilFFTs.plan_rfft(u_r, (1, 2))
    iplan = PencilFFTs.plan_irfft(û_c, (1, 2), Ny)
    
    # Coordinate arrays
    dx = Lx / Nx
    dy = Ly / Ny
    x = dx .* (0:(Nx-1))
    y = dy .* (0:(Ny-1))
    
    # Vertical coordinate based on grid type
    z, dz = make_vertical_grid(Nz, Lz, z_grid, z_boundary, stretch_params)
    
    # Horizontal wavenumber arrays (periodic)
    kx = [(i <= Nx÷2 ? i : i - Nx) * (2π/Lx) for i in 0:(Nx-1)]
    ky_full = [(j <= Ny÷2 ? j : j - Ny) * (2π/Ly) for j in 0:(Ny-1)]
    ky = ky_full[1:Nyc] # Truncated for rFFT
    
    # Dealiasing mask (only for horizontal directions)
    mask = twothirds_mask(Nx, Nyc)
    
    return Domain{FT, typeof(pr), typeof(pc), typeof(fplan), typeof(iplan)}(
        Nx, Ny, Nz, Lx, Ly, Lz, x, y, z, dz, kx, ky, mask, z_boundary, z_grid,
        pr, pc, fplan, iplan
    )
end

"""
    make_vertical_grid(Nz, Lz, z_grid, z_boundary, stretch_params) -> (z, dz)

Create vertical coordinate array and grid spacing based on grid type and boundary conditions.
Returns both the coordinate vector z and the spacing vector dz.
"""
function make_vertical_grid(Nz::Int, Lz, z_grid::Symbol, z_boundary::Symbol, stretch_params)
    if z_grid == :uniform
        if z_boundary == :periodic
            dz_uniform = Lz / Nz
            z = dz_uniform .* (0:(Nz-1))
            dz = fill(dz_uniform, Nz)
        else
            # Non-periodic: include boundaries
            z = collect(range(0, Lz, length=Nz))
            dz = fill(Lz / (Nz-1), Nz-1)
            push!(dz, dz[end])  # Extend dz to match z length
        end
        
    elseif z_grid == :stretched
        if stretch_params === nothing
            error("stretch_params required for :stretched grid. Provide (type=:tanh, β=2.0) or similar.")
        end
        
        z, dz = create_stretched_grid(Nz, Lz, stretch_params)
        
    elseif z_grid == :custom
        if stretch_params === nothing || !haskey(stretch_params, :z_coords)
            error("For :custom grid, provide stretch_params=(z_coords=your_z_vector,)")
        end
        
        z = collect(stretch_params.z_coords)
        if length(z) != Nz
            error("Custom z_coords length ($(length(z))) must match Nz ($Nz)")
        end
        
        # Compute grid spacing
        dz = zeros(Nz)
        for i in 2:Nz-1
            dz[i] = 0.5 * (z[i+1] - z[i-1])
        end
        dz[1] = z[2] - z[1]
        dz[Nz] = z[Nz] - z[Nz-1]
        
    else
        error("Unknown z_grid type: $z_grid. Use :uniform, :stretched, or :custom")
    end
    
    return z, dz
end

"""
    make_wavenumber_arrays(kx, ky, Nx, Nyc) -> (Krsq, invKrsq)

Create wavenumber magnitude arrays for real FFTs.
"""
function make_wavenumber_arrays(kx, ky, Nx, Nyc)
    # Create 2D arrays for wavenumber magnitudes
    Krsq = zeros(FT, Nx, Nyc)
    invKrsq = zeros(FT, Nx, Nyc)
    
    for i in 1:Nx, j in 1:Nyc
        Krsq[i, j] = kx[i]^2 + ky[j]^2
        if Krsq[i, j] > 0
            invKrsq[i, j] = 1.0 / Krsq[i, j]
        else
            invKrsq[i, j] = 0.0  # Avoid division by zero at k=0
        end
    end
    
    return Krsq, invKrsq
end

"""
    get_aliased_wavenumbers(Nx, Nyc, aliased_fraction) -> (kxalias, kyalias)

Get the ranges of wavenumber indices that should be aliased (set to zero).
"""
function get_aliased_wavenumbers(Nx, Nyc, aliased_fraction)
    # x-direction aliasing (for full FFT range)
    kxalias = get_alias_range(Nx, aliased_fraction)
    
    # y-direction aliasing (for real FFT range)  
    kyalias = get_alias_range_rfft(Nyc, aliased_fraction)
    
    return kxalias, kyalias
end

"""
    get_alias_range(N, aliased_fraction) -> UnitRange{Int}

Get aliasing range for full FFT (both positive and negative wavenumbers).
"""
function get_alias_range(N, aliased_fraction)
    if aliased_fraction <= 0
        return 1:0  # Empty range
    end
    
    L = (1 - aliased_fraction) / 2
    R = (1 + aliased_fraction) / 2
    
    iL = floor(Int, L * N) + 1
    iR = ceil(Int, R * N)
    
    return iL:iR
end

"""
    get_alias_range_rfft(Nyc, aliased_fraction) -> UnitRange{Int}

Get aliasing range for real FFT (only positive wavenumbers).
"""
function get_alias_range_rfft(Nyc, aliased_fraction)
    if aliased_fraction <= 0
        return 1:0  # Empty range
    end
    
    # For real FFT, we only need to dealias the high positive wavenumbers
    cutoff = ceil(Int, (1 - aliased_fraction) * Nyc)
    
    return (cutoff+1):Nyc
end

# =============================================================================
# DEALIASING FUNCTIONS
# =============================================================================

"""
    dealias!(domain::Domain, field_spec)

Apply dealiasing to a spectral field by zeroing out aliased wavenumbers.
"""
function dealias!(domain::Domain, field_spec)
    _dealias!(field_spec, domain)
    return nothing
end

"""
    _dealias!(field_spec, domain::Domain)

Internal dealiasing function that zeros out aliased wavenumbers.
"""
function _dealias!(field_spec, domain::Domain)
    # Get local array from PencilArray
    field_local = field_spec.data
    
    # Get local ranges for this MPI process
    local_ranges = local_range(domain.pc)
    
    # Map global alias ranges to local ranges
    kx_local_alias = intersect_ranges(domain.kxalias, local_ranges[1])
    ky_local_alias = intersect_ranges(domain.kyalias, local_ranges[2])
    
    # Zero out aliased wavenumbers
    if !isempty(kx_local_alias)
        kx_local_indices = [i for (i, ig) in enumerate(local_ranges[1]) if ig in kx_local_alias]
        @views @. field_local[kx_local_indices, :, :] = 0
    end
    
    if !isempty(ky_local_alias)
        ky_local_indices = [j for (j, jg) in enumerate(local_ranges[2]) if jg in ky_local_alias]
        @views @. field_local[:, ky_local_indices, :] = 0
    end
    
    return nothing
end

"""
    intersect_ranges(global_range, local_range) -> Vector{Int}

Find intersection between global aliasing range and local MPI range.
"""
function intersect_ranges(global_range, local_range)
    return [i for i in global_range if i in local_range]
end

# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

"""
    makefilter(dom::Domain; order=4, innerK=2/3, outerK=1, tol=1e-15) -> Array

Create a spectral filter for the domain.

# Arguments
- `dom`: Domain structure
- `order`: Filter order (higher = sharper transition)
- `innerK`: Inner wavenumber (filter inactive below this)
- `outerK`: Outer wavenumber (filter approaches tol at this value)
- `tol`: Filter tolerance at outer wavenumber

# Returns
- Filter array matching spectral space dimensions
"""
function makefilter(dom::Domain; order=4, innerK=2/3, outerK=1, tol=1e-15)
    # Create normalized wavenumber magnitude
    dx = dom.Lx / dom.Nx
    dy = dom.Ly / dom.Ny
    
    Nx, Nyc = length(dom.kx), length(dom.ky)
    K = zeros(FT, Nx, Nyc)
    
    for i in 1:Nx, j in 1:Nyc
        kx_norm = abs(dom.kx[i] * dx / π)
        ky_norm = abs(dom.ky[j] * dy / π)
        K[i, j] = sqrt(kx_norm^2 + ky_norm^2)
    end
    
    # Create filter
    decay = -log(tol) / (outerK - innerK)^order
    filter = exp.(-decay .* max.(K .- innerK, 0).^order)
    filter[K .< innerK] .= 1
    
    return filter
end

"""
    makefilter(equation; kwargs...)

Create a filter for an equation structure (assuming it has a domain field).
"""
function makefilter(equation; kwargs...)
    return makefilter(equation.domain; kwargs...)
end

"""
    apply_filter!(field_spec, filter)

Apply a spectral filter to a field in spectral space.
"""
function apply_filter!(field_spec, filter)
    field_local = field_spec.data
    local_ranges = local_range(field_spec.pencil)
    
    # Get local portion of filter
    filter_local = view(filter, local_ranges[1], local_ranges[2])
    
    # Apply filter to all z levels
    for k in axes(field_local, 3)
        @views @. field_local[:, :, k] *= filter_local
    end
    
    return nothing
end


"""
    twothirds_mask(Nx, Nyc) -> BitMatrix

Create 2/3 dealiasing mask for horizontal directions only.
"""
function twothirds_mask(Nx::Int, Nyc::Int)
    mask = trues(Nx, Nyc)
    
    # Dealias in x direction
    kx_max = Int(Nx ÷ 3)
    for i in (kx_max+1):(Nx-kx_max)
        mask[i, :] .= false
    end
    
    # Dealias in y direction
    ky_max = Int(2 * Nyc ÷ 3)
    for j in (ky_max+1):Nyc
        mask[:, j] .= false
    end
    
    return mask
end

"""
    Base.show(io::IO, dom::Domain)

Pretty print 3D domain information.
"""
function Base.show(io::IO, dom::Domain)
    println(io, "3D Domain (Periodic x,y; Bounded z):")
    println(io, "  Nx × Ny × Nz: $(dom.Nx) × $(dom.Ny) × $(dom.Nz)")
    println(io, "  Lx × Ly × Lz: $(dom.Lx) × $(dom.Ly) × $(dom.Lz)")
    println(io, "  dx × dy × dz: $((domain.Lx/domain.Nx)) × $((domain.Ly/domain.Ny)) × $((domain.Lz/domain.Nz))")
    println(io, "  Z boundary  : $(domain.z_boundary)")
    println(io, "  Z grid type : $(domain.chebyshev === nothing ? "uniform/finite-difference" : "Chebyshev spectral")")
    println(io, "  Real pencil : $(typeof(domain.pr))")
    println(io, "  Spectral pencil: $(typeof(domain.pc))")
    println(io, "  MPI processes: $(MPI.Comm_size(domain.pr.comm))")
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

"""
    Base.eltype(domain::Domain) -> Type

Return the element type of the domain coordinates.
"""
Base.eltype(domain::Domain) = eltype(domain.x)

# Add FT constant if not defined elsewhere
const FT = Float64