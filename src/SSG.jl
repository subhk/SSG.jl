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
       using JLD2

       # Constants
       const FT = Float64

       # Include files in order
       include("domain.jl")
       include("fields.jl")
       include("utils.jl")
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

       # domain.jl  
       export Domain, make_domain, gridpoints, gridpoints_2d, dealias!

       # utils.jl
       export @ensuresamegrid, twothirds_mask, create_real_field, create_spectral_field,
              copy_field!, zero_field!, norm_field, inner_product

       # fields.jl
       export Fields, allocate_fields, zero_fields!, field_stats, enhanced_field_stats

       # transforms.jl
       export rfft!, irfft!, ddx!, ddy!, ddz!, laplacian_h!, jacobian!, jacobianh,
              parsevalsum, parsevalsum2, compute_energy, compute_enstrophy

       export rfft_2d!, irfft_2d!, ddx_2d!, ddy_2d!, jacobian_2d!

       # params.jl
       export Params, validate_params, has_diffusion, has_hyperdiffusion, has_filter

       # filter.jl
       export ExponentialFilter, HyperviscosityFilter, CutoffFilter, 
              CesaroFilter, CustomFilter, AbstractSpectralFilter,
              apply_spectral_filter!, create_filter, apply_filter_to_field!

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
       export compute_tendency!, compute_geostrophic_velocities!, 
              compute_surface_geostrophic_velocities!, compute_jacobian!,
              set_b!, set_φ!, compute_kinetic_energy, compute_surface_kinetic_energy
    
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