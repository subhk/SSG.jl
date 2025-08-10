# Code Structure and Architecture

## 🏗️ Overview

This document describes the **software architecture** and **code organization** of SSG.jl. The package follows a **modular design** with clear separation of concerns, making it maintainable, extensible, and suitable for high-performance computing applications.

## 📚 Table of Contents

1. [Package Structure](#package-structure)
2. [Core Data Structures](#core-data-structures)
3. [Module Dependencies](#module-dependencies)
4. [Memory Management](#memory-management)
5. [MPI Parallelization](#mpi-parallelization)
6. [Design Patterns](#design-patterns)
7. [Extension Points](#extension-points)

---

## Package Structure

### Directory Layout

```
SSG.jl/
├── src/                          # Source code
│   ├── SSG.jl                    # Main module and exports
│   ├── domain.jl                 # 3D domain setup and grid management  
│   ├── fields.jl                 # Field allocation and management
│   ├── transforms.jl             # FFT operations and spectral derivatives
│   ├── poisson.jl                # 3D SSG equation multigrid solver
│   ├── timestep.jl               # Time integration schemes
│   ├── nonlinear.jl              # Nonlinear terms and tendency computation
│   ├── filter.jl                 # Spectral filtering
│   ├── params.jl                 # Parameter structures
│   ├── utils.jl                  # Utility functions
│   ├── output.jl                 # I/O and data management
│   ├── setICs.jl                 # Initial condition setters
│   ├── coarse_domain.jl          # Multigrid domain coarsening
│   ├── extras/                   # Optional/experimental features
│   │   ├── mapping.jl            # Coordinate transformations
│   │   ├── omega_eq.jl           # Alternative formulations
│   │   └── solution_b4.md        # Mathematical derivations
│   └── docs/                     # Internal documentation
│       └── ssg_multigrid.md      # Multigrid solver details
├── examples/                     # Example simulations
│   ├── ssg_example.jl            # Main JFM paper example
│   ├── badin_example.jl          # Badin et al. reproduction
│   └── tmp.jl                    # Development/testing scripts
├── test/                         # Test suite
│   ├── runtests.jl               # Main test runner
│   ├── test_*.jl                 # Individual test modules
│   └── test_ssg_jfm_example.jl   # Integration tests
├── Project.toml                  # Package dependencies
├── Manifest.toml                 # Dependency lock file
└── README.md                     # Package overview
```

### File Responsibilities

| File | Primary Purpose | Key Functions |
|------|----------------|---------------|
| `SSG.jl` | Module definition, exports | Main module, constant definitions |
| `domain.jl` | Grid setup, wavenumbers | `make_domain()`, coordinate generation |
| `fields.jl` | Memory allocation, field management | `allocate_fields()`, field operations |
| `transforms.jl` | Spectral operations, FFTs | `rfft!()`, `ddx!()`, `jacobian!()` |
| `poisson.jl` | 3D SSG equation solver | `solve_ssg_equation()`, multigrid |
| `timestep.jl` | Time integration, problem setup | `timestep!()`, `SemiGeostrophicProblem` |
| `nonlinear.jl` | Tendency computation | `compute_tendency!()`, `compute_jacobian!()` |
| `filter.jl` | Spectral filtering | `apply_spectral_filter!()` |
| `params.jl` | Parameter structures | `TimeParams`, validation |
| `utils.jl` | Utility functions, diagnostics | Grid utilities, field operations |
| `output.jl` | I/O operations | File saving/loading, MPI coordination |
| `setICs.jl` | Initial conditions | Built-in initial condition functions |

---

## Core Data Structures

### Domain Structure

The `Domain` structure encapsulates all **grid information** and **computational resources**:

```julia
struct Domain{T, PR3D, PC3D, PFP3D, PR2D, PC2D, PFP2D}
    # Grid dimensions
    Nx::Int                       # Points in x-direction
    Ny::Int                       # Points in y-direction  
    Nz::Int                       # Points in z-direction

    # Physical domain size
    Lx::T                         # Domain length in x
    Ly::T                         # Domain length in y
    Lz::T                         # Domain depth in z

    # Coordinate vectors
    x::Vector{T}                  # x-coordinates [0, Lx]
    y::Vector{T}                  # y-coordinates [0, Ly] 
    z::Vector{T}                  # z-coordinates [-Lz, 0]
    dz::Vector{T}                 # Vertical spacing (non-uniform)

    # Spectral space arrays
    kx::Vector{T}                 # x-wavenumbers
    ky::Vector{T}                 # y-wavenumbers
    Krsq::Matrix{T}               # k²ᵣ + k²ᵧ for real FFTs
    invKrsq::Matrix{T}            # 1/(k²ᵣ + k²ᵧ)
    mask::BitMatrix               # Dealiasing mask

    # Grid properties
    z_boundary::Symbol            # Boundary condition type
    z_grid::Symbol                # Grid type (:uniform, :stretched)
    
    # Parallel decomposition (3D)
    pr3d::PR3D                    # Real-space pencil descriptor
    pc3d::PC3D                    # Complex-space pencil descriptor
    fplan::PFP3D                  # 3D FFT plan

    # Surface field support (2D)
    pr2d::PR2D                    # 2D real-space pencil  
    pc2d::PC2D                    # 2D complex-space pencil
    fplan_2d::PFP2D               # 2D FFT plan
    
    # Dealiasing parameters
    aliased_fraction::T           # Fraction of aliased modes
    kxalias::UnitRange{Int}       # x-wavenumber alias range
    kyalias::UnitRange{Int}       # y-wavenumber alias range
end
```

**Key Design Decisions:**
- **Type parameters** ensure zero-cost abstractions
- **Non-uniform vertical grids** via `dz` vector
- **Dual pencil support** for 2D surface and 3D volume operations
- **Pre-computed spectral arrays** for performance

### Fields Structure

The `Fields` structure manages **all simulation arrays**:

```julia
struct Fields{T, PR2D, PR3D, PC2D, PC3D}
    # Prognostic fields (2D surface evolution)
    bₛ::PencilArray{T, 2, PR2D}           # Surface buoyancy anomaly
    φₛ::PencilArray{T, 2, PR2D}           # Surface streamfunction
    
    # Diagnostic fields (3D for solver)
    φ::PencilArray{T, 3, PR3D}            # 3D streamfunction
    u::PencilArray{T, 3, PR3D}            # x-velocity component
    v::PencilArray{T, 3, PR3D}            # y-velocity component
    
    # Solver workspace (3D multigrid)
    φ_mg::PencilArray{T, 3, PR3D}         # Multigrid solution space
    b_mg::PencilArray{T, 3, PR3D}         # Multigrid RHS space
    
    # Computational workspace (2D surface)
    R::PencilArray{T, 2, PR2D}            # Residual array
    tmp::PencilArray{T, 2, PR2D}          # General purpose scratch
    tmp2::PencilArray{T, 2, PR2D}         # Additional scratch
    tmp3::PencilArray{T, 2, PR2D}         # Third scratch array
    
    # Spectral workspace (2D)
    bshat::PencilArray{Complex{T}, 2, PC2D}    # Spectral buoyancy
    φshat::PencilArray{Complex{T}, 2, PC2D}    # Spectral streamfunction
    tmpc_2d::PencilArray{Complex{T}, 2, PC2D}  # Complex scratch 2D
    tmpc2_2d::PencilArray{Complex{T}, 2, PC2D} # Second complex scratch 2D

    # Spectral workspace (3D) 
    φhat::PencilArray{Complex{T}, 3, PC3D}     # 3D spectral streamfunction
    tmpc_3d::PencilArray{Complex{T}, 3, PC3D}  # Complex scratch 3D
    tmpc2_3d::PencilArray{Complex{T}, 3, PC3D} # Second complex scratch 3D
end
```

**Memory Organization:**
- **Prognostic vs Diagnostic**: Clear separation of evolved vs computed fields
- **Dimension-specific**: 2D surface fields separate from 3D volume fields  
- **Workspace management**: Pre-allocated scratch arrays prevent allocations
- **Spectral/Physical pairs**: Matching arrays for transform operations

### Time Integration Structures

#### TimeParams
```julia
struct TimeParams{T<:AbstractFloat}
    dt::T                        # Base time step size
    scheme::TimeScheme           # Integration method
    filter_freq::Int             # Filtering frequency  
    filter_strength::T           # Filter amplitude
    cfl_safety::T                # CFL safety factor
    max_dt::T                    # Maximum time step
    min_dt::T                    # Minimum time step
    adaptive_dt::Bool            # Enable adaptive stepping
end
```

#### TimeState
```julia
mutable struct TimeState{T, PA2D}
    t::T                         # Current simulation time
    step::Int                    # Current step number
    
    # Multi-step method storage
    db_dt_old::PA2D              # Previous tendency (Adams-Bashforth)
    b_stage::PA2D                # RK intermediate stages
    k1::PA2D                     # RK stage derivatives  
    k2::PA2D
    k3::PA2D
    
    # Diagnostics
    dt_actual::T                 # Last time step used
    cfl_max::T                   # Maximum CFL number
end
```

#### SemiGeostrophicProblem
```julia
mutable struct SemiGeostrophicProblem{T<:AbstractFloat}
    # Core simulation components
    fields::Fields{T}            # All field arrays
    domain::Domain               # Grid and spectral info
    timestepper::TimeParams{T}   # Integration parameters
    clock::TimeState{T}          # Time and step tracking
    
    # Optional components
    diagnostics::Union{DiagnosticTimeSeries{T}, Nothing}
    output_settings::NamedTuple  # I/O configuration
end
```

---

## Module Dependencies

### Dependency Graph

```
         SSG.jl (main)
             │
    ┌────────┼────────┐
    │        │        │
utils.jl  params.jl  domain.jl
    │        │        │
    └────────┼────────┴──┐
             │           │
        fields.jl    transforms.jl
             │           │
    ┌────────┼───────────┴──┐
    │        │              │
filter.jl  poisson.jl   nonlinear.jl
    │        │              │
    └────────┼──────────────┴──┐
             │                 │
        timestep.jl         output.jl
             │                 │
        setICs.jl              │
                               │
                        coarse_domain.jl
```

### Dependency Rules

1. **No circular dependencies**: Clean hierarchical structure
2. **Minimal coupling**: Modules depend only on necessary components
3. **Interface stability**: Public APIs remain consistent
4. **Type stability**: Generic programming with concrete dispatch

### Import Strategy

Each module follows this pattern:
```julia
# External dependencies
using MPI
using PencilArrays: PencilArray, range_local
using LinearAlgebra: mul!, ldiv!

# Internal dependencies (explicit)
# Note: Functions used from other modules via qualified names
```

---

## Memory Management

### Allocation Strategy

**Pre-allocation Principle**: All major arrays allocated once during setup.

```julia
function allocate_fields(domain::Domain{T}) where T
    # 2D surface fields
    bₛ = PencilArray{T}(undef, domain.pr2d)
    φₛ = PencilArray{T}(undef, domain.pr2d)
    
    # 3D volume fields
    φ = PencilArray{T}(undef, domain.pr3d)
    u = PencilArray{T}(undef, domain.pr3d)
    v = PencilArray{T}(undef, domain.pr3d)
    
    # Workspace arrays (prevent allocations)
    tmp = PencilArray{T}(undef, domain.pr2d)
    tmp2 = PencilArray{T}(undef, domain.pr2d)
    
    # Spectral arrays
    bshat = PencilArray{Complex{T}}(undef, domain.pc2d)
    φhat = PencilArray{Complex{T}}(undef, domain.pc3d)
    
    return Fields{T}(bₛ, φₛ, φ, u, v, tmp, tmp2, bshat, φhat, ...)
end
```

### Memory Layout

**PencilArrays** provide **distributed arrays** with optimal memory layout:

```julia
# Example: 3D field on 4 processes
# Global size: 256×256×64
# Process 0: holds [1:128,   1:128,   1:64]
# Process 1: holds [129:256, 1:128,   1:64] 
# Process 2: holds [1:128,   129:256, 1:64]
# Process 3: holds [129:256, 129:256, 1:64]

field_local = field.data  # Access local portion
nx_local, ny_local, nz_local = size(field_local)
```

### Workspace Management

Critical performance optimization - **no allocations in time-stepping loops**:

```julia
function compute_tendency!(db_dt, fields, domain, params)
    # Use pre-allocated workspace
    rfft_2d!(domain, fields.bₛ, fields.bshat)      # ✓ No allocation
    ddx_2d!(domain, fields.bshat, fields.tmpc_2d)  # ✓ Use workspace
    irfft_2d!(domain, fields.tmpc_2d, fields.tmp)  # ✓ Use workspace
    
    # Avoid this anti-pattern:
    # temp = similar(fields.bₛ)  # ✗ Allocation in hot loop
end
```

### Memory Footprint

For a simulation with **Nx×Ny×Nz** grid points:

| Field Type | Count | Memory per Field | Total Memory |
|------------|-------|------------------|--------------|
| 2D Real | 8 | `8 * Nx * Ny` bytes | `64 * Nx * Ny` |
| 3D Real | 6 | `8 * Nx * Ny * Nz` bytes | `48 * Nx * Ny * Nz` |
| 2D Complex | 4 | `16 * (Nx/2+1) * Ny` bytes | `64 * (Nx/2+1) * Ny` |
| 3D Complex | 4 | `16 * (Nx/2+1) * Ny * Nz` bytes | `64 * (Nx/2+1) * Ny * Nz` |

**Total**: ~20× base field size for complete workspace.

---

## MPI Parallelization

### Domain Decomposition

SSG.jl uses **PencilArrays.jl** for **2D domain decomposition**:

```julia
# Automatic decomposition
comm = MPI.COMM_WORLD
pencil_2d = Pencil((Nx, Ny), comm)
pencil_3d = Pencil((Nx, Ny, Nz), comm)

# Fields automatically distributed
field_2d = PencilArray{Float64}(undef, pencil_2d)
field_3d = PencilArray{Float64}(undef, pencil_3d)
```

### Communication Patterns

#### FFT Transposes
```julia
# Forward FFT: X-pencils → Y-pencils
# Backward FFT: Y-pencils → X-pencils
# Handled transparently by PencilFFTs.jl

mul!(spec_field, fft_plan, real_field)  # Includes MPI transposes
```

#### Global Reductions
```julia
# Energy computation with MPI reduction
function compute_energy(field::PencilArray{T, 2}) where T
    local_energy = sum(abs2, field.data)
    global_energy = MPI.Allreduce(local_energy, MPI.SUM, field.pencil.comm)
    return global_energy
end
```

#### Boundary Exchange
```julia
# Boundary conditions applied independently on each process
function apply_boundary_conditions!(field::PencilArray{T, 3}) where T
    field_local = field.data
    # Each process handles its local boundary points
    # No MPI communication needed for periodic boundaries
end
```

### Load Balancing

**Automatic load balancing** through PencilArrays:
- **Equal distribution**: Each process gets ≈equal number of grid points
- **Memory balanced**: Arrays distributed to minimize memory imbalance
- **Communication optimized**: Minimal data movement during transposes

### Scalability Considerations

```julia
# Good: Scales well with increasing problem size
Nx = Ny = 1024, Nz = 64, processes = 256  # ~16K points per process

# Poor: Too many processes for problem size  
Nx = Ny = 128, Nz = 8, processes = 64     # ~256 points per process
```

**Rule of thumb**: Aim for **>1000 grid points per MPI process**.

---

## Design Patterns

### Functional Programming Style

**Pure functions** where possible:
```julia
# Good: Pure function
function compute_derivative(input_field, wavenumbers)
    output_field = similar(input_field)
    # ... computation ...
    return output_field
end

# Preferred: In-place for performance
function compute_derivative!(output_field, input_field, wavenumbers)
    # ... computation directly in output_field ...
    return nothing
end
```

### Type Stability

**Concrete types** for performance-critical code:
```julia
# Type-stable dispatch
function solve_equation(x::Vector{Float64}, params::TimeParams{Float64})
    # ... computation with concrete types
end

# Generic interface for flexibility
function solve_equation(x::AbstractVector{T}, params::TimeParams{T}) where T
    # ... delegates to concrete implementation
end
```

### Error Handling

**Defensive programming** with informative errors:
```julia
function make_domain(Nx, Ny, Nz; kwargs...)
    # Validate inputs
    Nx > 0 || throw(ArgumentError("Nx must be positive, got $Nx"))
    ispow2(Nx) || @warn "Nx=$Nx is not a power of 2, FFT performance may be suboptimal"
    
    # Proceed with construction
    # ...
end
```

### Configuration Pattern

**Flexible parameter handling**:
```julia
# Default parameters with keyword override
function TimeParams{T}(dt::T; 
                      scheme::TimeScheme=AB2_LowStorage,
                      filter_freq::Int=10,
                      adaptive_dt::Bool=false,
                      kwargs...) where T
    # Handle additional parameters
    return TimeParams{T}(dt, scheme, filter_freq, adaptive_dt, ...)
end
```

---

## Extension Points

### Adding New Time Integration Schemes

1. **Define scheme enum**:
```julia
@enum TimeScheme begin
    AB2_LowStorage
    RK3
    YourNewScheme  # Add here
end
```

2. **Implement stepping function**:
```julia
function timestep_your_scheme!(fields, domain, params, state)
    # Your implementation
    return dt_used
end
```

3. **Add to dispatcher**:
```julia
function timestep!(fields, domain, params, state)
    if params.scheme == YourNewScheme
        return timestep_your_scheme!(fields, domain, params, state)
    elseif ...
end
```

### Adding New Initial Conditions

```julia
function initialize_your_condition!(fields::Fields{T}, domain::Domain; 
                                   amplitude::T=T(1.0),
                                   custom_param::T=T(0.5)) where T
    
    range_locals = range_local(fields.bₛ.pencil)
    b_local = fields.bₛ.data
    
    # Your initialization logic
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            # Set b_local[i_local, j_local] = ...
        end
    end
    
    return nothing
end
```

### Adding New Boundary Conditions

1. **Modify Domain structure** (if needed):
```julia
# Add new boundary type
z_boundary_types = [:dirichlet, :neumann, :your_boundary, :mixed]
```

2. **Implement boundary application**:
```julia
function apply_your_boundary!(field, domain)
    # Your boundary condition logic
end
```

3. **Integrate in solver**:
```julia
function apply_boundary_conditions!(field, domain)
    if domain.z_boundary == :your_boundary
        apply_your_boundary!(field, domain)
    elseif ...
end
```

### Adding New Spectral Filters

```julia
struct YourCustomFilter <: AbstractSpectralFilter
    strength::Float64
    cutoff::Float64
    # Your filter parameters
end

function apply_filter!(field_hat, filter::YourCustomFilter, domain)
    # Apply your filter in spectral space
end
```

---

*This document describes the current architecture of SSG.jl. For implementation details of specific algorithms, see the Theory and Mathematics documentation.*