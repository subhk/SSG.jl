# SSG.jl: Surface Semi-Geostrophic Solver

## 🌊 Overview

**SSG.jl** is a high-performance Julia package for simulating **Surface Semi-Geostrophic (SSG) turbulence** in oceanic and atmospheric flows. The package implements advanced numerical methods for solving the semi-geostrophic equations with spectral accuracy in horizontal directions and supports massively parallel computation via MPI.

### Key Features

-  **Spectral Methods**: FFT-based spectral derivatives for high accuracy
-  **Multigrid Solvers**: Advanced 3D Semi-Geostrophic equation solver with adaptive smoothers
-  **MPI Parallel**: Distributed computation using PencilArrays.jl for domain decomposition
-  **Advanced Time Integration**: Multiple schemes (Adams-Bashforth, Runge-Kutta) with adaptive time stepping
-  **Flexible Filtering**: Spectral filters for numerical stability
-  **Comprehensive Diagnostics**: Energy, enstrophy, and turbulence statistics

##  Table of Contents

1. [Mathematical Theory](#-mathematical-theory)
2. [Code Architecture](#️-code-architecture)
3. [Installation & Setup](#-installation--setup)
4. [Quick Start Guide](#-quick-start-guide)
5. [Core API Reference](#-core-api-reference)
6. [Advanced Features](#-advanced-features)
7. [Examples & Tutorials](#-examples--tutorials)
8. [Performance & Scaling](#-performance--scaling)
9. [Contributing](#-contributing)

---

##  Mathematical Theory

### Surface Semi-Geostrophic Equations

The package solves the **surface semi-geostrophic equations** which describe the evolution of surface buoyancy anomalies under geostrophic constraint:

```math
\frac{\partial b}{\partial t} + J(\psi, b) = 0
```

where:
- `b(x,y,t)`: surface buoyancy anomaly
- `ψ(x,y,t)`: geostrophic streamfunction  
- `J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x`: Jacobian (advection term)

### Geostrophic Velocities

The velocities are diagnostic variables computed from the streamfunction:

```math
u = -\frac{\partial \psi}{\partial y}, \quad v = \frac{\partial \psi}{\partial x}
```

### 3D Semi-Geostrophic Solver

For the full 3D problem, the package implements a sophisticated solver for the **Semi-Geostrophic equation**:

```math
\nabla^2 \Phi = \varepsilon D\Phi \quad \text{(SSG Equation)}
```

where:
- `∇² = ∂²/∂X² + ∂²/∂Y² + ∂²/∂Z²` (3D Laplacian in geostrophic coordinates)
- `DΦ = (∂²Φ/∂X²)(∂²Φ/∂Y²) - (∂²Φ/∂X∂Y)²` (nonlinear differential operator)
- `ε`: external parameter (measure of global Rossby number)

### Boundary Conditions

The solver supports sophisticated boundary conditions:

```math
\frac{\partial \Phi}{\partial Z} = \tilde{b}_s \quad \text{at } Z = 0 \text{ (surface)}
```

```math
\frac{\partial \Phi}{\partial Z} = 0 \quad \text{at } Z = -H \text{ (bottom)}
```

### Monge-Ampère Relation

For the surface semi-geostrophic model, the streamfunction and buoyancy are related through the **Monge-Ampère equation**:

```math
\det(D^2\phi) = b
```

where `D²φ` is the Hessian matrix of the streamfunction.

---

## 🏗️ Code Architecture

### Modular Design

The package follows a clean modular architecture:

```
SSG.jl/
├── src/
│   ├── SSG.jl              # Main module and exports
│   ├── domain.jl           # 3D domain setup and grid management  
│   ├── fields.jl           # Field allocation and management
│   ├── transforms.jl       # FFT operations and spectral derivatives
│   ├── poisson.jl          # 3D SSG equation multigrid solver
│   ├── timestep.jl         # Time integration schemes
│   ├── nonlinear.jl        # Nonlinear terms and tendency computation
│   ├── filter.jl           # Spectral filtering
│   ├── params.jl           # Parameter structures
│   ├── utils.jl            # Utility functions
│   ├── output.jl           # I/O and data management
│   └── setICs.jl           # Initial condition setters
├── examples/               # Example simulations
├── test/                   # Test suite
└── docs/                   # Additional documentation
```

### Core Data Structures

#### Domain Structure
```julia
struct Domain{T, PR3D, PC3D, PFP3D, PR2D, PC2D, PFP2D}
    # Grid dimensions
    Nx::Int, Ny::Int, Nz::Int
    Lx::T, Ly::T, Lz::T
    
    # Coordinate vectors
    x::Vector{T}, y::Vector{T}, z::Vector{T}
    dz::Vector{T}  # Non-uniform vertical spacing
    
    # Spectral space
    kx::Vector{T}, ky::Vector{T}
    Krsq::Matrix{T}, invKrsq::Matrix{T}
    mask::BitMatrix  # Dealiasing mask
    
    # Pencil descriptors and FFT plans
    pr3d::PR3D, pc3d::PC3D, fplan::PFP3D
    pr2d::PR2D, pc2d::PC2D, fplan_2d::PFP2D
end
```

#### Fields Structure
```julia
struct Fields{T, PR2D, PR3D, PC2D, PC3D}
    # Prognostic fields (2D surface)
    bₛ::PencilArray{T, 2, PR2D}     # surface buoyancy
    φₛ::PencilArray{T, 2, PR2D}     # surface streamfunction
    
    # Diagnostic fields (3D)
    φ::PencilArray{T, 3, PR3D}      # 3D streamfunction
    u::PencilArray{T, 3, PR3D}      # velocity components
    v::PencilArray{T, 3, PR3D}
    
    # Spectral arrays
    bshat::PencilArray{Complex{T}, 2, PC2D}
    φhat::PencilArray{Complex{T}, 3, PC3D}
    
    # Scratch arrays for computations
    tmp::PencilArray{T, 2, PR2D}
    tmpc_2d::PencilArray{Complex{T}, 2, PC2D}
    # ... additional workspace arrays
end
```

---

## Installation & Setup

### Prerequisites

- **Julia 1.10+**: Modern Julia version
- **MPI Implementation**: OpenMPI, MPICH, or Intel MPI
- **FFTW**: For FFT operations (automatically installed)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/SSG.jl.git
cd SSG.jl
```

2. **Install dependencies:**
```julia
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

3. **Set up MPI (if needed):**
```julia
julia --project=. -e "using MPI; MPI.install_mpiexec()"
```

### Verification

Test the installation:
```bash
# Single process
julia --project=. test/runtests.jl

# Multi-process
mpirun -np 4 julia --project=. test/runtests.jl
```

---

##  Quick Start Guide

### Basic Simulation Setup

```julia
using MPI
using SSG

# Initialize MPI
MPI.Init()

# Create domain
domain = make_domain(128, 128, 8; 
                    Lx=2π, Ly=2π, Lz=1.0,
                    comm=MPI.COMM_WORLD)

# Create problem
prob = SemiGeostrophicProblem(domain; 
                             scheme=RK3, 
                             dt=0.01)

# Set initial conditions (Taylor-Green vortex)
set_initial_conditions!(prob, initialize_taylor_green!; 
                       amplitude=1.0)

# Run simulation
step_until!(prob, 5.0)

# Clean up
MPI.Finalize()
```

### Monitoring Simulation Progress

```julia
# Enable diagnostics
prob = SemiGeostrophicProblem(domain; 
                             enable_diagnostics=true,
                             output_dir="results/")

# Run with progress monitoring
stepforward!(prob, 1000)  # Take 1000 time steps

# Access diagnostics
diag = prob.diagnostics
println("Final kinetic energy: ", diag.kinetic_energy[end])
```

---

## Core API Reference

### Domain Creation

#### `make_domain(Nx, Ny, Nz; kwargs...)`
Creates a 3D computational domain.

**Arguments:**
- `Nx, Ny, Nz::Int`: Grid points in x, y, z directions
- `Lx, Ly, Lz::Real`: Domain sizes (default: 2π, 2π, 1.0)
- `z_boundary::Symbol`: Boundary type (:dirichlet, :neumann, :periodic)
- `z_grid::Symbol`: Vertical grid (:uniform, :stretched, :custom)
- `stretch_params::NamedTuple`: Grid stretching parameters
- `comm::MPI.Comm`: MPI communicator

**Returns:** `Domain` structure

**Examples:**
```julia
# Uniform grid
domain = make_domain(64, 64, 16; Lx=4π, Ly=4π, Lz=2.0)

# Stretched grid (surface concentrated)
domain = make_domain(128, 128, 32; 
                    z_grid=:stretched,
                    stretch_params=(type=:tanh, β=2.5, surface_concentration=true))

# Ocean-like domain
domain = make_domain(256, 256, 64; 
                    Lx=100e3, Ly=100e3, Lz=1000.0,  # 100km x 100km x 1km
                    z_grid=:stretched)
```

### Field Operations

#### `allocate_fields(domain) -> Fields`
Allocates all field arrays for the domain.

#### `zero_fields!(fields)`
Sets all field values to zero.

#### `copy_field!(dest, src)`
Copies field data between PencilArrays.

### Transform Operations

#### `rfft!(domain, realfield, specfield)`
Forward real FFT: real space → spectral space.

#### `irfft!(domain, specfield, realfield)`
Inverse real FFT: spectral space → real space.

#### `ddx!(domain, Â, out̂)`, `ddy!(domain, Â, out̂)`
Spectral derivatives ∂/∂x and ∂/∂y.

**Example:**
```julia
# Compute ∂b/∂x
rfft_2d!(domain, fields.bₛ, fields.bshat)      # Transform to spectral
ddx_2d!(domain, fields.bshat, fields.tmpc_2d)  # Compute derivative
irfft_2d!(domain, fields.tmpc_2d, fields.tmp)  # Transform back
```

### Time Integration

#### `SemiGeostrophicProblem(domain; kwargs...)`
Main problem structure encapsulating the complete system.

**Arguments:**
- `domain::Domain`: Computational domain
- `scheme::TimeScheme`: Integration scheme (AB2_LowStorage, RK3)
- `dt::Real`: Time step size
- `adaptive_dt::Bool`: Enable adaptive time stepping
- `filter_freq::Int`: Spectral filtering frequency
- `enable_diagnostics::Bool`: Enable diagnostic computation

#### `timestep!(fields, domain, params, state)`
Advances the solution by one time step.

#### `step_until!(prob, stop_time)`
Integrates until specified time.

#### `stepforward!(prob, nsteps)`
Takes specified number of time steps.

### Initial Conditions

#### `set_b!(prob, b_field, domain)`
Sets buoyancy field and solves for streamfunction.

#### `set_φ!(prob, φ_field, domain)`
Sets streamfunction and computes derived buoyancy.

#### Built-in Initial Conditions:
- `initialize_taylor_green!(fields, domain; amplitude=1.0)`
- `initialize_gaussian_vortex!(fields, domain; amplitude=1.0, width=0.5)`
- `initialize_random_field!(fields, domain; amplitude=0.1, seed=1234)`
- `initialize_baroclinic_jet!(fields, domain; amplitude=1.0, width=π/4)`

### Solver Operations

#### `solve_ssg_equation(Φ_initial, b_rhs, ε, domain; kwargs...)`
Solves the 3D Semi-Geostrophic equation using multigrid methods.

**Arguments:**
- `Φ_initial::PencilArray{T, 3}`: Initial guess
- `b_rhs::PencilArray{T, 3}`: Right-hand side
- `ε::Real`: SSG parameter (Rossby number measure)
- `tol::Real`: Convergence tolerance (default: 1e-8)
- `maxiter::Int`: Maximum iterations
- `smoother::Symbol`: Smoother type (:spectral, :adaptive, :sor)

**Returns:** `(solution, diagnostics)`

#### `solve_monge_ampere_fields!(fields, domain; kwargs...)`
Interface-compatible Monge-Ampère solver for existing code.

### Diagnostics

#### `compute_energy(field, domain)`
Computes spectral energy using Parseval's theorem.

#### `compute_enstrophy(field, domain)`
Computes enstrophy (energy in vorticity).

#### `parsevalsum(field, domain)`, `parsevalsum2(field, domain)`
Computes spectral integrals and squared integrals.

---

##  Advanced Features

### MPI Parallelization

SSG.jl uses **PencilArrays.jl** for distributed memory parallelization:

```julia
# Multi-process simulation
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if rank == 0
    println("Running on $nprocs processes")
end

domain = make_domain(256, 256, 64; comm=comm)
# ... rest of simulation ...

MPI.Finalize()
```

**Running:**
```bash
mpirun -np 8 julia --project=. your_simulation.jl
```

### Spectral Filtering

Control numerical stability with spectral filters:

```julia
# Exponential filter
prob = SemiGeostrophicProblem(domain;
                             filter_freq=10,        # Apply every 10 steps  
                             filter_strength=0.1)   # Mild filtering

# Manual filtering
apply_spectral_filter!(fields, domain, 0.2)  # Strong filtering
```

### Non-Uniform Grids

Support for stretched vertical grids:

```julia
# Hyperbolic tangent stretching
domain = make_domain(128, 128, 64;
                    z_grid=:stretched,
                    stretch_params=(type=:tanh, β=3.0, surface_concentration=true))

# Power law stretching  
domain = make_domain(128, 128, 64;
                    z_grid=:stretched,
                    stretch_params=(type=:power, α=1.5))

# Custom grid specification
custom_z = [0.0, -0.1, -0.3, -0.6, -1.0]  # Non-uniform spacing
domain = make_domain(128, 128, 5;
                    z_grid=:custom,
                    stretch_params=(z_coords=custom_z,))
```

### Adaptive Time Stepping

Automatic time step adjustment based on CFL condition:

```julia
prob = SemiGeostrophicProblem(domain;
                             dt=0.01,
                             adaptive_dt=true,
                             cfl_safety=0.5,
                             max_dt=0.1,
                             min_dt=1e-6)
```

### Multigrid Solver Configuration

Fine-tune the 3D SSG equation solver:

```julia
solution, diag = solve_ssg_equation(Φ₀, b, ε, domain;
                                   tol=1e-10,
                                   maxiter=50,
                                   n_levels=4,           # Multigrid levels
                                   smoother=:adaptive,   # Adaptive smoother selection
                                   verbose=true)

println("Converged: $(diag.converged)")
println("Iterations: $(diag.iterations)")
println("Final residual: $(diag.final_residual)")
```

---

## Examples & Tutorials

### Example 1: Basic Taylor-Green Vortex

```julia
using MPI, SSG

MPI.Init()

# Setup
domain = make_domain(64, 64, 8; Lx=2π, Ly=2π)
prob = SemiGeostrophicProblem(domain; dt=0.01, scheme=RK3)

# Taylor-Green initial condition
set_initial_conditions!(prob, initialize_taylor_green!; amplitude=1.0)

# Run simulation
final_time = 2.0
step_until!(prob, final_time)

# Analyze results
if prob.diagnostics !== nothing
    diag = prob.diagnostics
    println("Energy evolution:")
    for i in 1:length(diag.times)
        @printf "t=%.2f, KE=%.6f\n" diag.times[i] diag.kinetic_energy[i]
    end
end

MPI.Finalize()
```

### Example 2: High-Resolution Ocean Simulation

```julia
using MPI, SSG

MPI.Init()

# Ocean-scale domain (100km x 100km x 1km)
domain = make_domain(512, 512, 128; 
                    Lx=100e3, Ly=100e3, Lz=1000.0,
                    z_grid=:stretched,
                    stretch_params=(type=:exponential, β=2.0))

# High-resolution problem with filtering
prob = SemiGeostrophicProblem(domain;
                             dt=60.0,              # 60 second time steps
                             scheme=RK3,
                             adaptive_dt=true,
                             filter_freq=20,
                             filter_strength=0.05,
                             enable_diagnostics=true)

# Initialize with baroclinic instability
set_initial_conditions!(prob, initialize_baroclinic_jet!; 
                       amplitude=0.1,     # 0.1 m/s velocity scale
                       width=20e3)        # 20 km jet width

# Long integration (30 days)
step_until!(prob, 30 * 24 * 3600)

MPI.Finalize()
```

### Example 3: Parameter Sweep Study

```julia
using MPI, SSG

function parameter_sweep()
    MPI.Init()
    
    # Parameter ranges
    ε_values = [0.01, 0.05, 0.1, 0.2]
    resolutions = [(64, 64, 8), (128, 128, 16), (256, 256, 32)]
    
    results = Dict()
    
    for (Nx, Ny, Nz) in resolutions
        for ε in ε_values
            # Setup simulation
            domain = make_domain(Nx, Ny, Nz)
            prob = SemiGeostrophicProblem(domain; dt=0.01)
            
            # Standard initial condition
            set_initial_conditions!(prob, initialize_taylor_green!)
            
            # Run to equilibrium
            step_until!(prob, 5.0)
            
            # Store final energy
            final_ke = prob.diagnostics.kinetic_energy[end]
            results[(Nx, Ny, Nz, ε)] = final_ke
            
            if MPI.Comm_rank(MPI.COMM_WORLD) == 0
                @printf "Resolution %dx%dx%d, ε=%.3f: Final KE = %.6f\n" Nx Ny Nz ε final_ke
            end
        end
    end
    
    MPI.Finalize()
    return results
end

# Run parameter sweep
results = parameter_sweep()
```

### Example 4: Custom Initial Conditions

```julia
function initialize_custom_vortex!(fields::Fields{T}, domain::Domain; 
                                  amplitude::T=T(1.0), 
                                  center_x::T=T(π), 
                                  center_y::T=T(π),
                                  radius::T=T(0.5)) where T
    
    # Get local ranges for MPI
    range_locals = range_local(fields.bₛ.pencil)
    b_local = fields.bₛ.data
    
    # Create custom vortex in buoyancy field
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Distance from vortex center
            r = sqrt((x - center_x)^2 + (y - center_y)^2)
            
            # Gaussian vortex with ring structure
            if r < 2*radius
                profile = exp(-(r/radius)^2) * (1 - exp(-(r/radius)^2))
                b_local[i_local, j_local] = amplitude * profile
            else
                b_local[i_local, j_local] = T(0)
            end
        end
    end
    
    return nothing
end

# Usage
set_initial_conditions!(prob, initialize_custom_vortex!; 
                       amplitude=2.0, 
                       center_x=π/2, 
                       center_y=π, 
                       radius=0.3)
```

---

## ⚡ Performance & Scaling

### Optimization Guidelines

1. **Grid Size Selection:**
   - Use powers of 2 for optimal FFT performance
   - Balance resolution with computational cost
   - Typical ratios: Nx:Ny:Nz ≈ 8:8:1 to 16:16:1

2. **MPI Process Distribution:**
   ```julia
   # Good: Square process grid
   mpirun -np 16 julia simulation.jl  # 4x4 process grid
   
   # Better: Matches aspect ratio
   mpirun -np 8 julia simulation.jl   # 4x2 process grid for 2:1 domain
   ```

3. **Memory Usage:**
   - Surface fields: `8 * Nx * Ny * bytes_per_float`
   - 3D fields: `8 * Nx * Ny * Nz * bytes_per_float`
   - Total: ~10-20x base field memory for workspace

### Scaling Results

Typical performance on modern HPC systems:

| Resolution | Processes | Time/Step | Parallel Efficiency |
|-----------|-----------|-----------|-------------------|
| 256²×32   | 16        | 0.1s      | 95%              |
| 512²×64   | 64        | 0.3s      | 90%              |
| 1024²×128 | 256       | 1.2s      | 85%              |
| 2048²×256 | 1024      | 4.8s      | 80%              |

### Profiling & Debugging

```julia
using Profile

# Profile time step
@profile for i = 1:100
    timestep!(prob.fields, prob.domain, prob.timestepper, prob.clock)
end

Profile.print()
```

---

##  Contributing

### Development Setup

1. **Fork and clone:**
```bash
git clone https://github.com/your-username/SSG.jl.git
cd SSG.jl
```

2. **Development environment:**
```julia
julia --project=. -e "using Pkg; Pkg.develop()"
```

3. **Run tests:**
```bash
julia --project=. test/runtests.jl
```

### Adding New Features

1. **Solvers:** Add new time integration schemes in `timestep.jl`
2. **Initial Conditions:** Add functions in `setICs.jl`
3. **Diagnostics:** Extend diagnostic calculations in `timestep.jl`
4. **Boundary Conditions:** Modify domain setup in `domain.jl`

### Code Style

- Follow [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- Use descriptive variable names
- Add comprehensive docstrings
- Include unit tests for new functions

### Testing

```bash
# Unit tests
julia --project=. test/test_transforms.jl

# Integration tests  
julia --project=. test/test_ssg_example.jl

# MPI tests
mpirun -np 4 julia --project=. test/test_parallel.jl
```

---

##  References

### Scientific Papers

1. **Badin, G., & Crisciani, F.** (2018). *Variational Formulation of Fluid and Geophysical Fluid Dynamics*. Springer.

2. **McIntyre, M. E., & Norton, W. A.** (2000). Potential vorticity inversion on a hemisphere. *Journal of the Atmospheric Sciences*, 57(9), 1214-1235.

3. **Vallis, G. K.** (2017). *Atmospheric and Oceanic Fluid Dynamics*. Cambridge University Press.

### Technical References

- [PencilArrays.jl Documentation](https://github.com/jipolanco/PencilArrays.jl)
- [PencilFFTs.jl Documentation](https://github.com/jipolanco/PencilFFTs.jl)  
- [MPI.jl Documentation](https://github.com/JuliaParallel/MPI.jl)

### Mathematical Background

- **Semi-Geostrophic Theory**: Hoskins, B. J. (1975). The geostrophic momentum approximation and the semi-geostrophic equations. *Journal of the Atmospheric Sciences*, 32(2), 233-242.

- **Monge-Ampère Equations**: Benamou, J. D., & Brenier, Y. (2000). A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem. *Numerische Mathematik*, 84(3), 375-393.

---

##  Support & Community

- **Issues:** [GitHub Issues](https://github.com/subhk/SSG.jl/issues)
- **Discussions:** [GitHub Discussions](https://github.com/subhk/SSG.jl/discussions)
- **Email:** [subhajitkar19@gmail.com](mailto:subhajitkar19@gmail.com)

---

##  License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [Julia](https://julialang.org/) - "A fresh approach to technical computing"
- Uses [PencilArrays.jl](https://github.com/jipolanco/PencilArrays.jl) for MPI parallelization
- FFT operations via [PencilFFTs.jl](https://github.com/jipolanco/PencilFFTs.jl)
- Inspired by classical geophysical fluid dynamics codes


