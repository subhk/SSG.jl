# ============================================================================
# SSG Solver - Modular File Structure
# ============================================================================

# # src/SSG.jl - Main module file
# module SSG

#     using MPI
#     using PencilArrays
#     using PencilFFTs
#     using FFTW
#     using LinearAlgebra
#     using Statistics
#     using Printf
#     using Random
#     using Dates
#     using JLD2

#     using LoopVectorization  # For SIMD optimization
#     using StaticArrays
#     using Parameters
#     using BenchmarkTools

#     # Constants
#     const FT = Float64


#     # Export main functions

#     # fields.jl exports  
#     export Fields, allocate_fields, zero_fields!, copy_field!, field_stats

#    # domain.jl exports
#     export Domain, gridpoints, gridpoints_2d

#     # params.jl exports
#     export Params, validate_params, has_diffusion, has_hyperdiffusion, has_filter

#     # coarse_domain.jl exports
#     export create_coarse_domain, validate_coarsening, estimate_mg_memory


#     # transforms.jl export
#     export rfft!, irfft!, dealias!, ddx!, ddy!, ddz!, d2dxdy!, d2dz2!,
#         jacobian!, laplacian_h!, laplacian_3d!, divergence_3d!, gradient_h!,
#         compute_vorticity_z!, parsevalsum, parsevalsum2

#     # utils.jl exports
#     export jacobian, jacobianh, advection_term!, compute_energy, compute_enstrophy,
#         compute_total_buoyancy, compute_buoyancy_variance, compute_cfl_number,
#         norm_field, inner_product, create_real_field, create_spectral_field

#     # timestep.jl exports
#     export TimeScheme, TimeParams, TimeState, SemiGeostrophicProblem,
#         timestep!, integrate!, step_until!, stepforward!, run!,
#         AB2_LowStorage, RK3, RK3_LowStorage

#     # nonlinear.jl exports
#     export compute_tendency!, compute_geostrophic_velocities!

#     # poisson.jl exports
#     export solve_ssg_equation, solve_monge_ampere_fields!, compute_ma_residual_fields!,
#         SSGLevel, SSGMultigridSolver, solve_poisson_simple,
#         ssg_sor_smoother_enhanced!, ssg_sor_smoother_adaptive!,
#         demo_ssg_solver, demo_nonuniform_grid_ssg, test_poisson_solver

#     # output.jl exports
#     export OutputFrequency, OutputManager, save_simulation_state_full,
#         save_snapshot, save_spectral_snapshot, process_all_outputs!,
#         load_simulation_state_full, DiagnosticTimeSeries


#     # filter.jl exports
#     export AbstractSpectralFilter, ExponentialFilter, HyperviscosityFilter,
#         CutoffFilter, CesaroFilter, apply_spectral_filter!, create_filter

#     # mapping.jl
#     export 
#         create_geostrophic_mapping,
#         map_geostrophic

#     # omega_eq.jl
#     export 
#         create_grid,
#         compute_vertical_velocity


#     # Include all submodules
#     include("utils.jl")
#     include("domain.jl")
#     include("fields.jl")
#     include("transforms.jl")
#     include("params.jl")
#     include("poisson.jl")
#     include("timestep.jl")
#     include("filter.jl")
#     include("output.jl")
#     include("coarse_domain.jl")
#     include("omega_eq.jl")
#     include("nonlinear.jl")


# end # module SSG


# module SSG

#     using MPI
#     using PencilArrays
#     using PencilFFTs
#     using FFTW
#     using LinearAlgebra
#     using Statistics
#     using Printf
#     using Random
#     using Dates
#     using JLD2

#     # Constants
#     const FT = Float64

#     # Include files in dependency order
#     include("params.jl")           # No dependencies
#     include("domain.jl")           # Depends on params (FT constant)
#     include("fields.jl")           # Depends on domain
#     include("transforms.jl")       # Depends on domain, fields
#     include("utils.jl")            # Depends on domain, fields, transforms
#     include("coarse_domain.jl")    # Depends on domain
#     include("poisson.jl")          # Depends on all above
#     include("filter.jl")           # Depends on domain, fields
#     include("timestep.jl")         # Depends on fields, domain, transforms
#     include("nonlinear.jl")        # Depends on fields, domain, transforms
#     include("output.jl")           # Depends on fields, domain
#     include("setICs.jl")           # Depends on all above

#     # Export main functions

#     # domain.jl exports
#     export Domain, make_domain, gridpoints, gridpoints_2d

#     # fields.jl exports  
#     export Fields, allocate_fields, zero_fields!, copy_field!, field_stats

#     # params.jl exports
#     export Params, validate_params, has_diffusion, has_hyperdiffusion, has_filter

#     # transforms.jl exports
#     export rfft!, irfft!, dealias!, ddx!, ddy!, ddz!, d2dxdy!, d2dz2!,
#         jacobian!, laplacian_h!, laplacian_3d!, divergence_3d!, gradient_h!,
#         compute_vorticity_z!, parsevalsum, parsevalsum2

#     # utils.jl exports
#     export jacobian, jacobianh, advection_term!, compute_energy, compute_enstrophy,
#         compute_total_buoyancy, compute_buoyancy_variance, compute_cfl_number,
#         norm_field, inner_product, create_real_field, create_spectral_field,
#         enhanced_field_stats

#     # timestep.jl exports
#     export TimeScheme, TimeParams, TimeState, SemiGeostrophicProblem,
#         timestep!, integrate!, step_until!, stepforward!, run!,
#         AB2_LowStorage, RK3, RK3_LowStorage, DiagnosticTimeSeries, update_diagnostics!

#     # nonlinear.jl exports
#     export compute_tendency!, compute_geostrophic_velocities!

#     # poisson.jl exports
#     export solve_ssg_equation, solve_monge_ampere_fields!, compute_ma_residual_fields!,
#         SSGLevel, SSGMultigridSolver

#     # output.jl exports
#     export OutputFrequency, OutputManager, save_simulation_state_full,
#         save_snapshot, save_spectral_snapshot, process_all_outputs!,
#         load_simulation_state_full

#     # filter.jl exports
#     export AbstractSpectralFilter, ExponentialFilter, HyperviscosityFilter,
#         CutoffFilter, CesaroFilter, apply_spectral_filter!, create_filter

#     # setICs.jl exports
#     export set_b!, set_φ!

#     # coarse_domain.jl exports
#     export create_coarse_domain

# end # module SSG



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
    using JLD2

    # Constants
    const FT = Float64

    # Include files in order
    include("utils.jl")
    include("domain.jl")
    include("fields.jl")
    include("transforms.jl")
    include("params.jl")
    include("filter.jl")
    include("poisson.jl")
    include("timestep.jl")
    include("output.jl")
    include("nonlinear.jl")
    
    # Optional files (if they exist)
    #isfile("nonlinear.jl") && include("nonlinear.jl")

    # Exports by file:

    # utils.jl
    export @ensuresamegrid, twothirds_mask, create_real_field, create_spectral_field,
           copy_field!, zero_field!, norm_field, inner_product

    # domain.jl  
    export Domain, make_domain, gridpoints, gridpoints_2d, dealias!

    # fields.jl
    export Fields, allocate_fields, zero_fields!, field_stats, enhanced_field_stats

    # transforms.jl
    export rfft!, irfft!, ddx!, ddy!, ddz!, laplacian_h!, jacobian!, jacobianh,
           parsevalsum, parsevalsum2, compute_energy, compute_enstrophy

    # params.jl
    export Params, validate_params, has_diffusion, has_hyperdiffusion, has_filter

    # filter.jl
    export ExponentialFilter, HyperviscosityFilter, CutoffFilter, 
           apply_spectral_filter!, create_filter

    # poisson.jl
    export solve_ssg_equation, SSGLevel, solve_monge_ampere_fields!, 
           solve_poisson_simple

    # timestep.jl
    export TimeParams, TimeState, SemiGeostrophicProblem, AB2_LowStorage, RK3,
           timestep!, step_until!, run!

    # output.jl
    export save_simulation_state_full, load_simulation_state_full, OutputManager

    # nonlinear.jl (if exists)
    #if isdefined(SSG, :compute_tendency!)
    export compute_tendency!, compute_geostrophic_velocities!
    
    #end

end # module SSG


# File structure that should be:
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