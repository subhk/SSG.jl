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
- `mask`: Dealiasing mask for horizontal directions
- `z_boundary`: Boundary condition type (:dirichlet, :neumann, :free_slip, etc.)
- `z_grid`: Vertical grid type (:uniform, :stretched, :custom)
- `pr`: Real-space pencil descriptor
- `pc`: Spectral-space pencil descriptor (for horizontal FFTs)
- `fplan, iplan`: Forward and inverse FFT plans (horizontal only)
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
    
    mask::BitMatrix
    z_boundary::Symbol
    z_grid::Symbol
    
    # Pencil descriptors
    pr::PA # real-space pencil
    pc::PC # complex/spectral pencil
    
    # FFT plans (horizontal only)
    fplan::PF
    iplan::PB
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
    create_stretched_grid(Nz, Lz, stretch_params) -> (z, dz)

Create various types of stretched grids for clustering points near boundaries.
"""
function create_stretched_grid(Nz::Int, Lz, stretch_params)
    stretch_type = stretch_params.type
    
    # Create normalized coordinate η ∈ [-1, 1] or [0, 1]
    if stretch_type in [:tanh, :sinh]
        η = range(-1, 1, length=Nz)
        β = stretch_params.β
        
        if stretch_type == :tanh
            # Hyperbolic tangent stretching - clusters at boundaries
            ξ = tanh.(β .* η) ./ tanh(β)
        elseif stretch_type == :sinh
            # Hyperbolic sine stretching - clusters at center
            ξ = sinh.(β .* η) ./ sinh(β)
        end
        
        # Map from [-1,1] to [0,Lz]
        z = 0.5 * Lz .* (ξ .+ 1)
        
    elseif stretch_type == :power
        η = range(0, 1, length=Nz)
        α = stretch_params.α
        
        # Power law stretching
        ξ = η.^α
        z = Lz .* ξ
        
    elseif stretch_type == :exponential
        η = range(0, 1, length=Nz)
        β = stretch_params.β
        
        # Exponential clustering toward bottom boundary
        ξ = (exp.(β .* η) .- 1) ./ (exp(β) - 1)
        z = Lz .* ξ
        
    else
        error("Unknown stretch type: $stretch_type. Use :tanh, :sinh, :power, or :exponential")
    end
    
    # Compute grid spacing
    dz = zeros(Nz)
    for i in 2:Nz-1
        dz[i] = 0.5 * (z[i+1] - z[i-1])
    end
    dz[1] = z[2] - z[1]
    dz[Nz] = z[Nz] - z[Nz-1]
    
    return collect(z), dz
end

# """
#     make_chebyshev_matrices(Nz, z_boundary) -> NamedTuple

# Create Chebyshev differentiation matrices for bounded domain.
# Returns first and second derivative matrices with boundary conditions applied.
# """
# function make_chebyshev_matrices(Nz::Int, z_boundary::Symbol)
#     # Create Chebyshev differentiation matrix
#     D1, D2 = chebyshev_diff_matrices(Nz)
    
#     # Apply boundary conditions
#     if z_boundary == :dirichlet
#         # Zero at both boundaries: remove first and last rows/columns
#         D1_bc = D1[2:end-1, 2:end-1]
#         D2_bc = D2[2:end-1, 2:end-1]
#         bc_indices = 2:Nz-1
#     elseif z_boundary == :neumann
#         # Zero derivative at boundaries
#         D1_bc, D2_bc, bc_indices = apply_neumann_bc(D1, D2, Nz)
#     elseif z_boundary == :free_slip
#         # w=0, ∂u/∂z=∂v/∂z=0 at boundaries
#         D1_bc, D2_bc, bc_indices = apply_free_slip_bc(D1, D2, Nz)
#     else
#         # No boundary conditions applied
#         D1_bc = D1
#         D2_bc = D2
#         bc_indices = 1:Nz
#     end
    
#     return (D1=D1_bc, D2=D2_bc, indices=bc_indices, type=z_boundary)
# end

# """
#     chebyshev_diff_matrices(N) -> (D1, D2)

# Compute Chebyshev differentiation matrices for first and second derivatives.
# """
# function chebyshev_diff_matrices(N::Int)
#     # Chebyshev points
#     θ = π .* (0:(N-1)) ./ (N-1)
#     x = -cos.(θ)
    
#     # First derivative matrix
#     c = [i == 1 || i == N ? 2.0 : 1.0 for i in 1:N]
#     c[1] *= (-1)^(1-1)
#     c[N] *= (-1)^(N-1)
    
#     D1 = zeros(N, N)
#     for i in 1:N, j in 1:N
#         if i ≠ j
#             D1[i,j] = (c[i]/c[j]) * (-1)^(i+j) / (x[i] - x[j])
#         end
#     end
    
#     # Diagonal elements
#     for i in 1:N
#         D1[i,i] = -sum(D1[i, 1:N .≠ i])
#     end
    
#     # Second derivative matrix
#     D2 = D1^2
    
#     return D1, D2
# end

# """
#     apply_neumann_bc(D1, D2, N) -> (D1_bc, D2_bc, indices)

# Apply Neumann boundary conditions to differentiation matrices.
# """
# function apply_neumann_bc(D1, D2, N)
#     # For Neumann BC: ∂u/∂z = 0 at boundaries
#     # Modify the boundary rows to enforce this condition
#     D1_bc = copy(D1)
#     D2_bc = copy(D2)
    
#     # Set boundary conditions in first and last rows
#     D1_bc[1, :] = D1[1, :]  # ∂u/∂z = 0 at bottom
#     D1_bc[N, :] = D1[N, :]  # ∂u/∂z = 0 at top
    
#     return D1_bc, D2_bc, 1:N
# end

# """
#     apply_free_slip_bc(D1, D2, N) -> (D1_bc, D2_bc, indices)

# Apply free-slip boundary conditions to differentiation matrices.
# For free-slip: w=0 and ∂u/∂z=∂v/∂z=0 at boundaries.
# """
# function apply_free_slip_bc(D1, D2, N)
#     # Similar to Neumann but may need special treatment for different variables
#     # This is a simplified version - actual implementation depends on the equations
#     return apply_neumann_bc(D1, D2, N)
# end

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
    println(io, "  dx × dy × dz: $((dom.Lx/dom.Nx)) × $((dom.Ly/dom.Ny)) × $((dom.Lz/dom.Nz))")
    println(io, "  Z boundary  : $(dom.z_boundary)")
    println(io, "  Z grid type : $(dom.chebyshev === nothing ? "uniform/finite-difference" : "Chebyshev spectral")")
    println(io, "  Real pencil : $(typeof(dom.pr))")
    println(io, "  Spectral pencil: $(typeof(dom.pc))")
    println(io, "  MPI processes: $(MPI.Comm_size(dom.pr.comm))")
end

# Add FT constant if not defined elsewhere
const FT = Float64