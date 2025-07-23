# ============================================================================
# SSG Solver - Modular File Structure
# ============================================================================

# src/SSG.jl - Main module file
module SSG

using MPI
using PencilArrays
using PencilFFTs
using FFTW
using LinearAlgebra
using Statistics
using Printf
using KrylovKit
using Random
using Dates
using NetCDF

# Constants
const FT = Float64

# Include all submodules
include("utils.jl")
include("domain.jl")
include("fields.jl")
include("transforms.jl")
include("params.jl")
include("physics.jl")
include("monge_ampere.jl")
include("timestep.jl")
include("filter.jl")
include("hyperdiffusion.jl")
include("io.jl")
include("run.jl")

# Export main functions
export Domain, Fields, Params
export make_domain, allocate_fields
export run_ssg
export save_snapshot, create_output_file

end # module SSG

# ============================================================================
# File structure that should be created:
# ============================================================================

#=
src/
├── SSGPencil.jl          # Main module (above)
├── utils.jl              # Utility functions and macros
├── domain.jl             # Domain setup and grid management
├── fields.jl             # Field allocation and management
├── transforms.jl         # FFT operations and spectral derivatives
├── params.jl             # Parameter structures
├── physics.jl            # Physical computations (velocities, Jacobian)
├── monge_ampere.jl       # Monge-Ampère equation solver
├── timestep.jl           # Time stepping schemes
├── filter.jl             # Spectral filtering
├── hyperdiffusion.jl     # Hyperdiffusion operators
├── io.jl                 # Input/output operations
├── run.jl                # Main simulation driver

examples/
├── basic_run.jl      # Basic example
=#