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

    # domain.jl exports
    export Domain, make_domain, gridpoints, gridpoints_2d

    # fields.jl exports  
    export Fields, allocate_fields, zero_fields!, copy_field!, field_stats

    # params.jl exports
    export Params, has_diffusion, has_hyperdiffusion, has_filter

    # coarse_domain.jl exports
    export create_coarse_domain, validate_coarsening, estimate_mg_memory


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

    # nonlinear.jl exports
    export compute_tendency!, compute_geostrophic_velocities!

    # poisson.jl exports
    export solve_ssg_equation, solve_monge_ampere_fields!, compute_ma_residual_fields!,
        SSGLevel, SSGMultigridSolver, solve_poisson_simple,
        ssg_sor_smoother_enhanced!, ssg_sor_smoother_adaptive!,
        demo_ssg_solver, demo_nonuniform_grid_ssg, test_poisson_solver

    # output.jl exports
    export OutputFrequency, OutputManager, save_simulation_state_full,
        save_snapshot, save_spectral_snapshot, process_all_outputs!,
        load_simulation_state_full, DiagnosticTimeSeries


    # filter.jl exports
    export AbstractSpectralFilter, ExponentialFilter, HyperviscosityFilter,
        CutoffFilter, CesaroFilter, apply_spectral_filter!, create_filter

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
    include("nonlinear.jl")


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