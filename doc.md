# SSG.jl Compatibility Issues and Fixes

## Summary of Changes Made

I've made the following key compatibility fixes to make all the files work together:

### 1. **Domain Structure** (`domain.jl`)
- **Fixed**: Consistent constructor pattern `Domain(Nx, Ny, Nz, Lx, Ly, Lz, comm)`
- **Fixed**: Proper wavenumber array creation for real FFTs
- **Fixed**: Dealiasing mask creation compatible with PencilArrays
- **Added**: Helper functions for grid creation and validation

### 2. **Fields Structure** (`fields.jl`)
- **Fixed**: Consistent field allocation using `allocate_fields(domain)`
- **Fixed**: MPI-aware field statistics computation
- **Fixed**: Grid compatibility checking with `@ensuresamegrid` macro
- **Added**: Enhanced field statistics for debugging

### 3. **Transform Operations** (`transforms.jl`)
- **Fixed**: Spectral derivatives with proper MPI range handling
- **Fixed**: Parseval sum functions for energy/enstrophy calculations
- **Fixed**: Jacobian computation for advection terms
- **Added**: Vertical finite difference derivatives for 3D domains

### 4. **Time Integration** (`timestep.jl`)
- **Fixed**: Compatible problem structure `SemiGeostrophicProblem`
- **Fixed**: `step_until!` interface matching expected behavior
- **Fixed**: Diagnostic time series with MPI reductions
- **Added**: Initial condition functions (Taylor-Green, Gaussian vortex, etc.)

### 5. **Main Module** (`SSG.jl`)
- **Fixed**: Dependency order in include statements
- **Fixed**: Consistent exports across all modules
- **Added**: Module initialization for MPI and FFTW

## Key Compatibility Features

### MPI Support
- All functions work with PencilArrays and MPI domain decomposition
- Proper local range handling for distributed arrays
- MPI reductions for global statistics and norms

### Spectral Methods
- Real FFTs in horizontal directions (x,y) 
- Finite differences in vertical direction (z)
- Proper dealiasing for nonlinear terms
- Consistent wavenumber array handling

### Memory Management
- Efficient field allocation and reuse
- Scratch arrays for intermediate calculations
- Low-storage time integration schemes

## Usage Example

```julia
using MPI
include("SSG.jl")
using .SSG

# Initialize MPI
MPI.Init()

# Create domain
domain = make_domain(128, 128, 8; Lx=2π, Ly=2π, Lz=1.0)

# Create problem
prob = SemiGeostrophicProblem(domain; scheme=RK3, dt=0.01)

# Set initial conditions
set_initial_conditions!(prob, initialize_taylor_green!; amplitude=1.0)

# Run simulation
step_until!(prob, 5.0)

# Clean up
MPI.Finalize()
```

## Testing the Compatibility

The provided `example_usage.jl` file contains comprehensive tests:

1. **Basic simulation**: Taylor-Green vortex evolution
2. **Advanced simulation**: High-resolution with filtering
3. **Performance benchmark**: Scaling tests
4. **Integration schemes**: Comparison of time steppers

## Known Limitations

1. **SSG Solver**: Currently uses Poisson approximation (`Δφ = b`) instead of full Monge-Ampère equation (`det(D²φ) = b`)
2. **Vertical discretization**: Basic finite differences (could be enhanced with spectral methods)
3. **Output system**: Simplified version of the full I/O system
4. **Filter system**: Basic exponential filter (could add more sophisticated options)

## Future Enhancements

1. **Full Monge-Ampère solver** with multigrid methods
2. **Advanced vertical discretization** (Chebyshev, stretched grids)
3. **Comprehensive I/O system** with HDF5/NetCDF support
4. **Performance optimizations** with LoopVectorization.jl
5. **Adaptive mesh refinement** for high-resolution features

## File Dependencies

```
SSG.jl (main)
├── utils.jl (utilities, no dependencies)
├── domain.jl (depends on: utils)
├── fields.jl (depends on: domain)  
├── params.jl (standalone)
├── transforms.jl (depends on: domain, fields, utils)
├── coarse_domain.jl (depends on: domain)
├── filter.jl (depends on: domain, transforms)
├── poisson.jl (depends on: domain, fields, transforms)
├── timestep.jl (depends on: fields, domain, transforms, poisson)
├── output.jl (depends on: fields, domain, transforms)
└── setICs.jl (depends on: timestep)
```
