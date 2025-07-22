# src/domain.jl
# Domain setup and grid management

"""
    struct Domain{T, PA, PC, PF, PB}
Holds grid information, wavenumbers, FFT plans, and pencil descriptors.

# Fields
- `Nx, Ny`: Grid dimensions
- `Lx, Ly`: Domain size
- `x, y`: Coordinate vectors
- `kx, ky`: Wavenumber vectors
- `mask`: Dealiasing mask
- `pr`: Real-space pencil descriptor
- `pc`: Spectral-space pencil descriptor  
- `fplan, iplan`: Forward and inverse FFT plans
"""
struct Domain{T, PA<:AbstractPencil, PC<:AbstractPencil, PF, PB}
    Nx::Int
    Ny::Int
    Lx::T
    Ly::T
    x::Vector{T}
    y::Vector{T}
    kx::Vector{T}
    ky::Vector{T}
    mask::BitMatrix
    # Pencil descriptors
    pr::PA        # real-space pencil
    pc::PC        # complex/spectral pencil
    # FFT plans
    fplan::PF
    iplan::PB
end

"""
    make_domain(Nx, Ny; Lx=2π, Ly=2π, comm=MPI.COMM_WORLD) -> Domain
Create a 2D periodic domain with PencilArrays decomposition and FFT plans.

# Arguments
- `Nx, Ny`: Grid points in x and y directions
- `Lx, Ly`: Domain size (default: 2π × 2π)
- `comm`: MPI communicator (default: MPI.COMM_WORLD)

# Returns
- `Domain`: Configured domain structure
"""
function make_domain(Nx::Int, Ny::Int; Lx=2π, Ly=2π, comm=MPI.COMM_WORLD)
    # Initialize MPI if needed
    MPI.Initialized() || MPI.Init()

    # Create pencil descriptors
    # Real-space: full (Nx, Ny) array
    pr = Pencil((Nx, Ny), comm)
    
    # Spectral-space: rFFT reduces y dimension to Ny÷2+1
    Nyc = fld(Ny, 2) + 1
    pc = Pencil((Nx, Nyc), comm)

    # Create FFT plans using dummy arrays
    u_r = PencilArray(pr, zeros(FT, local_size(pr)))
    û_c = PencilArray(pc, zeros(Complex{FT}, local_size(pc)))

    fplan = PencilFFTs.plan_rfft(u_r, (1, 2))
    iplan = PencilFFTs.plan_irfft(û_c, (1, 2), Ny)

    # Coordinate arrays
    dx = Lx / Nx
    dy = Ly / Ny
    x = dx .* (0:(Nx-1))
    y = dy .* (0:(Ny-1))

    # Wavenumber arrays
    kx = [(i <= Nx÷2 ? i : i - Nx) * (2π/Lx) for i in 0:(Nx-1)]
    ky_full = [(j <= Ny÷2 ? j : j - Ny) * (2π/Ly) for j in 0:(Ny-1)]
    ky = ky_full[1:Nyc]  # Truncated for rFFT

    # Dealiasing mask
    mask = twothirds_mask(Nx, Nyc)

    return Domain{FT, typeof(pr), typeof(pc), typeof(fplan), typeof(iplan)}(
        Nx, Ny, Lx, Ly, x, y, kx, ky, mask, pr, pc, fplan, iplan
    )
end

"""
    Base.show(io::IO, dom::Domain)
Pretty print domain information.
"""
function Base.show(io::IO, dom::Domain)
    println(io, "Domain:")
    println(io, "  Nx × Ny       : $(dom.Nx) × $(dom.Ny)")
    println(io, "  Lx × Ly       : $(dom.Lx) × $(dom.Ly)")
    println(io, "  dx × dy       : $((dom.Lx/dom.Nx)) × $((dom.Ly/dom.Ny))")
    println(io, "  Real pencil   : $(typeof(dom.pr))")
    println(io, "  Spectral pencil: $(typeof(dom.pc))")
    println(io, "  MPI processes : $(MPI.Comm_size(dom.pr.comm))")
end