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
    using Random
    using Dates
    using JLD2

    using LoopVectorization  # For SIMD optimization
    using StaticArrays
    using Parameters
    using BenchmarkTools

    # Constants
    const FT = Float64


    # Export main functions

    # domain.jl
    export Domain, 
            Fields, 
            Params,
            make_domain, 
            allocate_fields,
            run_ssg
            save_snapshot, 
            create_output_file

    # coarse_domain.jl export
    export 


    # fields.jl export
    export 
        allocate_fields,
        zero_fields!


    # transforms.jl export
    export rfft!, irfft!, dealias!, ddx!, ddy!, ddz!, d2dxdy!, d2dz2!,
       laplacian_h!, laplacian_3d!, divergence_3d!, gradient_h!,
       compute_vorticity_z!, parsevalsum, parsevalsum2

    # utils.jl exports
    export jacobian, jacobianh, advection_term!, compute_energy, compute_enstrophy,
       compute_total_buoyancy, compute_buoyancy_variance, compute_cfl_number,
       norm_field, inner_product, create_real_field, create_spectral_field

    # timestep.jl exports
    export TimeScheme, TimeParams, TimeState, SemiGeostrophicProblem,
       timestep!, integrate!, step_until!, stepforward!, run!,
       AB2_LowStorage, RK3, RK3_LowStorage


    # mapping.jl
    export 
        create_geostrophic_mapping,
        map_geostrophic

    # omega_eq.jl
    export 
        create_grid,
        compute_vertical_velocity


    # Include all submodules
    include("utils.jl")
    include("domain.jl")
    include("fields.jl")
    include("transforms.jl")
    include("params.jl")
    include("poisson.jl")
    include("timestep.jl")
    include("filter.jl")
    include("output.jl")
    include("coarse_domain.jl")
    include("omega_eq.jl")


end # module SSG

# ============================================================================
# File structure that should be created:
# ============================================================================

#=
src/
├── SSG.jl                # Main module (above)
├── utils.jl              # Utility functions and macros
├── domain.jl             # Domain setup and grid management
├── fields.jl              # Field allocation and management
├── transforms.jl         # FFT operations and spectral derivatives
├── params.jl             # Parameter structures
├── poisson.jl            # 3D Poisson solver
├── timestep.jl           # Time stepping schemes
├── filter.jl              # Spectral filtering
├── output.jl             # Input/output operations
├── run.jl                # Main simulation driver

examples/
├── basic_run.jl          # Basic example
=#