# src/domain.jl
# 3D Domain setup with vertically bounded domain

using PencilArrays: size_local, size_global, range_local, Pencil
using PencilFFTs: PencilFFTPlan, Transforms
using AbstractFFTs: fftfreq, rfftfreq

const FT = Float64

"""
    struct Domain{T, PR, PC, PFP}

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
struct Domain{T, PR3D, PC3D, PFP3D, PR2D, PC2D, PFP2D}
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
    
    # 3D Pencil descriptors
    pr3d::PR3D      # real-space pencil
    pc3d::PC3D      # complex/spectral pencil
    
    # FFT plans (horizontal only)
    fplan::PFP3D
    iplan::PFP3D

    # 2D surface field support
    pr2d::PR2D      # 2D real-space pencil for surface
    pc2d::PC2D      # 2D complex/spectral pencil for surface

    fplan_2d::PFP2D  # 2D FFT plans for surface
    iplan_2d::PFP2D
    
    # Dealiasing parameters
    aliased_fraction::T
    kxalias::UnitRange{Int}
    kyalias::UnitRange{Int}
end

"""
    Domain(Nx, Ny, Nz; Lx=2π, Ly=2π, Lz=1.0, z_boundary=:dirichlet, 
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
function Domain(Nx::Int, Ny::Int, Nz::Int; 
                Lx=2π, Ly=2π, Lz=1.0, 
                z_boundary=:dirichlet,
                z_grid=:uniform,
                stretch_params=nothing,
                comm=MPI.COMM_WORLD)
    
    T = typeof(float(Lx))  # Infer type properly

    # Initialize MPI if needed
    MPI.Initialized() || MPI.Init()
    
    # Create pencil descriptors for 3D arrays
    # Real-space: full (Nx, Ny, Nz) array
    pr3d  = Pencil((Nx, Ny, Nz),  comm)
    
    # Spectral-space: rFFT reduces y dimension to Ny÷2+1, z remains same
    Nyc   = fld(Ny, 2) + 1
    pc3d  = Pencil((Nx, Nyc, Nz), comm)
    
    # FFT plans: FFT on x, RFFT on y, no transform on z
    fplan = PencilFFTPlan(
        pr3d,
        (Transforms.FFT(),   # complex → complex on x
        Transforms.RFFT(),   # real → complex on y
        Transforms.NoTransform());
        fftw_flags = FFTW.MEASURE
    )
    iplan = fplan  # same plan used for inverse (via ldiv! or \)

    pr2d = Pencil((Nx, Ny),  comm)
    pc2d = Pencil((Nx, Nyc), comm)
    
    # Create 2D FFT plans
    fplan_2d = PencilFFTPlan(
        pr2d,
        (Transforms.FFT(),     # FFT on x
         Transforms.RFFT());   # RFFT on y
        fftw_flags = FFTW.MEASURE
    )
    iplan_2d = fplan_2d

    # Coordinate arrays
    dx = Lx / Nx
    dy = Ly / Ny
    x = dx .* (0:(Nx-1))
    y = dy .* (0:(Ny-1))
    
    # sampling rates (reciprocal of spacing)
    fs_x = 1/dx
    fs_y = 1/dy

    # full (signed) frequencies in x → angular wavenumbers kx
    # AbstractFFTs.fftfreq(n, fs) returns [0, 1, …, n÷2, −n÷2+1, …, −1] * (fs/n) 
    kx = 2π .* fftfreq(Nx, fs_x)

    # full (signed) frequencies in y (if you ever need both positive & negative)
    ky_full = 2π .* fftfreq(Ny, fs_y)

    # one‐sided frequencies in y for an RFFT → angular wavenumbers ky
    # AbstractFFTs.rfftfreq(n, fs) returns [0, 1, …, n÷2] * (fs/n) 
    ky = 2π .* rfftfreq(Ny, fs_y)

    # Vertical coordinate based on grid type
    z, dz = make_vertical_grid(Nz, Lz, z_grid, z_boundary, stretch_params)
    
    # Dealiasing mask (only for horizontal directions)
    mask = twothirds_mask(Nx, Nyc)

    Krsq, invKrsq = make_wavenumber_arrays(kx, ky, Nx, Nyc)

    aliased_fraction = T(1/3)
    kxalias, kyalias =  get_aliased_wavenumbers(Nx, Nyc, aliased_fraction)
    
    return Domain{T, typeof(pr3d), typeof(pc3d), typeof(fplan), 
            typeof(pr2d), typeof(pc2d), typeof(fplan_2d)}(
            Nx, Ny, Nz, 
            T(Lx), T(Ly), T(Lz), 
            x, y, z, dz, 
            kx, ky, Krsq, invKrsq, 
            mask, z_boundary, z_grid,
            pr3d, pc3d, fplan, iplan, 
            pr2d, pc2d, fplan_2d, iplan_2d, 
            aliased_fraction, kxalias, kyalias
    )
end

"""
    make_domain(Nx, Ny, Nz; kwargs...) -> Domain

Convenience constructor for creating a Domain with common defaults.
This is the main interface used in examples and applications.

# Arguments
- `Nx, Ny, Nz`: Grid dimensions  
- `Lx, Ly, Lz`: Domain size (default: 2π, 2π, 1.0)
- `z_boundary`: Vertical boundary condition (default: :dirichlet)
- `z_grid`: Vertical grid type (default: :uniform)  
- `stretch_params`: Grid stretching parameters (default: nothing)
- `comm`: MPI communicator (default: MPI.COMM_WORLD)

# Examples
```julia
# Basic uniform domain
domain = make_domain(64, 64, 16)

# Ocean-like domain with surface stretching
domain = make_domain(256, 256, 32; 
                    Lx=100e3, Ly=100e3, Lz=1000.0,
                    z_grid=:stretched,
                    stretch_params=(type=:exponential, β=2.0, surface_concentration=true))

# Custom MPI communicator
domain = make_domain(128, 128, 20; comm=my_comm)
```
"""
function make_domain(Nx::Int, Ny::Int, Nz::Int; 
                    Lx=2π, Ly=2π, Lz=1.0,
                    z_boundary=:dirichlet, 
                    z_grid=:uniform,
                    stretch_params=nothing,
                    comm=MPI.COMM_WORLD)
    
    return Domain(Nx, Ny, Nz; 
                 Lx=Lx, Ly=Ly, Lz=Lz,
                 z_boundary=z_boundary,
                 z_grid=z_grid, 
                 stretch_params=stretch_params,
                 comm=comm)
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
    T = eltype(kx)
    # Create 2D arrays for wavenumber magnitudes
    Krsq    = zeros(T, Nx, Nyc)
    invKrsq = zeros(T, Nx, Nyc)
    
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
    # Get local array from PencilArray
    field_local = field_spec.data
    
    # Get local ranges for this MPI process
    range_locals = range_local(domain.pc3d)
    
    # Apply mask to local data
    mask_local = view(domain.mask, range_locals[1], range_locals[2])
    
    @inbounds for k in axes(field_local, 3)
        @views @. field_local[:, :, k] = ifelse(mask_local, field_local[:, :, k], 0)
    end
    
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
    range_locals = range_local(domain.pc3d)
    
    # Map global alias ranges to local ranges
    kx_local_alias = intersect_ranges(domain.kxalias, range_locals[1])
    ky_local_alias = intersect_ranges(domain.kyalias, range_locals[2])
    
    # Zero out aliased wavenumbers
    if !isempty(kx_local_alias)
        kx_local_indices = [i for (i, ig) in enumerate(range_locals[1]) if ig in kx_local_alias]
        @views @. field_local[kx_local_indices, :, :] = 0
    end
    
    if !isempty(ky_local_alias)
        ky_local_indices = [j for (j, jg) in enumerate(range_locals[2]) if jg in ky_local_alias]
        @views @. field_local[:, ky_local_indices, :] = 0
    end
    
    return nothing
end

"""
    intersect_ranges(global_range, range_local) -> Vector{Int}

Find intersection between global aliasing range and local MPI range.
"""
function intersect_ranges(global_range, range_local)
    return [i for i in global_range if i in range_local]
end

# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

"""
    makefilter(domain::Domain; order=4, innerK=2/3, outerK=1, tol=1e-15) -> Array

Create a spectral filter for the domain.

# Arguments
- `domain`: Domain structure
- `order`: Filter order (higher = sharper transition)
- `innerK`: Inner wavenumber (filter inactive below this)
- `outerK`: Outer wavenumber (filter approaches tol at this value)
- `tol`: Filter tolerance at outer wavenumber

# Returns
- Filter array matching spectral space dimensions
"""
function makefilter(domain::Domain; order=4, innerK=2/3, outerK=1, tol=1e-15)
    T = eltype(domain.Lx)
    # Create normalized wavenumber magnitude
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    
    Nx, Nyc = length(domain.kx), length(domain.ky)
    K = zeros(T, Nx, Nyc)
    
    for i in 1:Nx, j in 1:Nyc
        kx_norm = abs(domain.kx[i] * dx / π)
        ky_norm = abs(domain.ky[j] * dy / π)
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
    range_locals = range_local(field_spec.pencil)
    
    # Get local portion of filter
    filter_local = view(filter, range_locals[1], range_locals[2])
    
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
    dealias_2d!(domain::Domain, field_spec_2d)

Apply 2D dealiasing to a 2D spectral field (e.g., surface buoyancy field).
Uses the horizontal dealiasing mask to zero out aliased wavenumbers.
"""
function dealias_2d!(domain::Domain, field_spec_2d)
    # Get local array from PencilArray
    field_local = field_spec_2d.data
    
    # Get local ranges for this MPI process
    range_locals = range_local(domain.pc2d)
    
    # Apply mask to local data
    mask_local = view(domain.mask, range_locals[1], range_locals[2])
    
    @inbounds @views @. field_local = ifelse(mask_local, field_local, 0)
    
    return nothing
end

"""
    Base.show(io::IO, domain::Domain)

Pretty print 3D domain information.
"""
function Base.show(io::IO, domain::Domain)
    println(io, "3D Domain (Periodic x,y; Bounded z):")
    println(io, "  Nx × Ny × Nz: $(domain.Nx) × $(domain.Ny) × $(domain.Nz)")
    println(io, "  Lx × Ly × Lz: $(domain.Lx) × $(domain.Ly) × $(domain.Lz)")
    println(io, "  dx × dy × dz: $((domain.Lx/domain.Nx)) × $((domain.Ly/domain.Ny)) × $((domain.Lz/domain.Nz))")
    println(io, "  Z boundary  : $(domain.z_boundary)")
    println(io, "  Z grid type : $(domain.z_grid)")
    println(io, "  Real pencil : $(typeof(domain.pr3d))")
    println(io, "  Spectral pencil: $(typeof(domain.pc3d))")
    println(io, "  MPI processes: $(MPI.Comm_size(domain.pr3d.comm))")
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

# =============================================================================
# STRETCHED GRID IMPLEMENTATIONS
# =============================================================================

"""
    create_stretched_grid(Nz, Lz, stretch_params) -> (z, dz)

Create a stretched vertical grid based on the specified parameters.
"""
function create_stretched_grid(Nz::Int, Lz, stretch_params)
    stretch_type = get(stretch_params, :type, :tanh)
    
    if stretch_type == :tanh
        β = get(stretch_params, :β, 2.0)
        surface_concentration = get(stretch_params, :surface_concentration, true)
        return create_tanh_grid(Nz, Lz, β, surface_concentration)
        
    elseif stretch_type == :exponential
        β = get(stretch_params, :β, 2.0) 
        surface_concentration = get(stretch_params, :surface_concentration, true)
        return create_exponential_grid(Nz, Lz, β, surface_concentration)
        
    elseif stretch_type == :sinh
        β = get(stretch_params, :β, 2.0)
        return create_sinh_grid(Nz, Lz, β)
        
    elseif stretch_type == :power
        α = get(stretch_params, :α, 1.5)
        return create_power_grid(Nz, Lz, α)
        
    else
        error("Unknown stretch type: $stretch_type")
    end
end

"""
    create_tanh_grid(Nz, Lz, β, surface_concentration) -> (z, dz)

Create hyperbolic tangent stretched grid.
"""
function create_tanh_grid(Nz::Int, Lz, β, surface_concentration::Bool)
    T = eltype(Lz)
    # Uniform grid in computational space
    ξ = collect(range(0, 1, length=Nz))
    
    if surface_concentration
        # Concentrate points near surface (z = Lz)
        z = Lz .* (1 .- 0.5 .* (1 .+ tanh.(β .* (2 .* ξ .- 1)) ./ tanh(β)))
    else
        # Concentrate points near bottom (z = 0)
        z = Lz .* 0.5 .* (1 .+ tanh.(β .* (2 .* ξ .- 1)) ./ tanh(β))
    end
    
    # Compute grid spacing
    dz = zeros(T, Nz)
    for i in 2:Nz-1
        dz[i] = 0.5 * (z[i+1] - z[i-1])
    end
    dz[1] = z[2] - z[1]
    dz[Nz] = z[Nz] - z[Nz-1]
    
    return z, dz
end

"""
    create_exponential_grid(Nz, Lz, β, surface_concentration) -> (z, dz)

Create exponentially stretched grid.
"""
function create_exponential_grid(Nz::Int, Lz, β, surface_concentration::Bool)
    T = eltype(Lz)
    ξ = collect(range(0, 1, length=Nz))
    
    if surface_concentration
        # Exponential clustering at surface
        z = Lz .* (1 .- exp.(-β .* ξ) ./ (1 - exp(-β)))
    else
        # Exponential clustering at bottom
        z = Lz .* (exp.(β .* ξ) .- 1) ./ (exp(β) - 1)
    end
    
    # Compute spacing
    dz = zeros(T, Nz)
    for i in 2:Nz-1
        dz[i] = 0.5 * (z[i+1] - z[i-1])
    end
    dz[1] = z[2] - z[1] 
    dz[Nz] = z[Nz] - z[Nz-1]
    
    return z, dz
end

"""
    create_sinh_grid(Nz, Lz, β) -> (z, dz)

Create hyperbolic sine stretched grid.
"""
function create_sinh_grid(Nz::Int, Lz, β)
    T = eltype(Lz)
    ξ = collect(range(-1, 1, length=Nz))
    
    # Symmetric stretching about middle
    z = Lz .* (0.5 .+ sinh.(β .* ξ) ./ (2 * sinh(β)))
    
    # Compute spacing
    dz = zeros(T, Nz)
    for i in 2:Nz-1
        dz[i] = 0.5 * (z[i+1] - z[i-1])
    end
    dz[1] = z[2] - z[1]
    dz[Nz] = z[Nz] - z[Nz-1]
    
    return z, dz
end

"""
    create_power_grid(Nz, Lz, α) -> (z, dz)

Create power-law stretched grid.
"""
function create_power_grid(Nz::Int, Lz, α)
    T = eltype(Lz)
    ξ = collect(range(0, 1, length=Nz))
    
    # Power law stretching
    z = Lz .* ξ.^α
    
    # Compute spacing
    dz = zeros(T, Nz)
    for i in 2:Nz-1
        dz[i] = 0.5 * (z[i+1] - z[i-1])
    end
    dz[1] = z[2] - z[1]
    dz[Nz] = z[Nz] - z[Nz-1]
    
    return z, dz
end