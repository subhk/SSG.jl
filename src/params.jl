# src/params.jl
# Parameter structures and constructors

"""
    struct Params{T}
Container for all physical and numerical parameters.

# Physical Parameters
- `κ`: Laplacian diffusivity coefficient
- `νh`: Hyperdiffusivity coefficient  
- `p_h`: Order of hyperdiffusion operator
- `ε`: External parameter for SSG equation: Rossby number measure

# Numerical Parameters
- `MA_tol`: Tolerance for Monge-Ampère solver
- `MA_maxiter`: Maximum iterations for Monge-Ampère solver

# Simulation Parameters
- `t_end`: End time of simulation
- `nsave`: Frequency of output (save every nsave timesteps)

# Optional Components
- `filter`: Spectral filter (or nothing)
"""
struct Params{T}
    # Physical parameters
    κ::T            # Laplacian diffusivity
    νh::T           # hyperdiffusivity coefficient
    p_h::Int        # order of hyperdiffusion
    ε::T            # Global Rossby number
    
    # Numerical parameters  
    MA_tol::T       # Monge-Ampere tolerance
    MA_maxiter::Int # Monge-Ampere max iterations
    
    # Simulation parameters
    t_end::T        # end time
    nsave::Int      # save frequency
    
    # Optional components
    filter::Any     # spectral filter (or nothing)
end

"""
    Params(; kwargs...)
Constructor for Params with keyword arguments and defaults.

# Keyword Arguments
- `κ=1e-4`: Laplacian diffusivity
- `νh=0.0`: Hyperdiffusivity coefficient
- `p_h=4`: Hyperdiffusion order
- `ε=0.1`: SSG external parameter (Rossby number measure)
- `MA_tol=1e-8`: Monge-Ampère tolerance
- `MA_maxiter=20`: Monge-Ampère max iterations
- `t_end=1.0`: End time
- `nsave=50`: Save frequency
- `filter=nothing`: Spectral filter

# Notes on ε Parameter
The parameter ε appears in the SSG equation (A1): ∇²Φ = εDΦ
- ε ≈ 0: Reduces to Poisson equation (surface quasi-geostrophic limit)
- ε ~ O(1): Full semi-geostrophic dynamics
- ε >> 1: Strong ageostrophic effects
- Typical values: ε ∈ [0.1, 1.0] for oceanic applications
"""
function Params(;
    κ::Real=1e-4,
    νh::Real=0.0,
    p_h::Integer=4,
    MA_tol::Real=1e-8,
    MA_maxiter::Integer=20,
    t_end::Real=1.0,
    nsave::Integer=50,
    filter=nothing
)
    T = promote_type(typeof(κ), typeof(νh), typeof(MA_tol), typeof(t_end))
    return Params{T}(T(κ), T(νh), Int(p_h), T(MA_tol), Int(MA_maxiter), 
                     T(t_end), Int(nsave), filter)
end

"""
    has_diffusion(params::Params) -> Bool
Check if standard diffusion is enabled.
"""
has_diffusion(params::Params) = params.κ != 0

"""
    has_hyperdiffusion(params::Params) -> Bool  
Check if hyperdiffusion is enabled.
"""
has_hyperdiffusion(params::Params) = params.νh != 0 && params.p_h > 1

"""
    has_filter(params::Params) -> Bool
Check if spectral filtering is enabled.
"""
has_filter(params::Params) = params.filter !== nothing

"""
    Base.show(io::IO, params::Params)
Pretty print parameter information.
"""
function Base.show(io::IO, params::Params)
    println(io, "Simulation Parameters:")
    println(io, "  Physical:")
    println(io, "    κ (diffusivity)     : $(params.κ)")
    println(io, "    νh (hyperdiffusion) : $(params.νh)")
    println(io, "    p_h (hyper order)   : $(params.p_h)")
    println(io, "  Numerical:")
    println(io, "    MA tolerance        : $(params.MA_tol)")
    println(io, "    MA max iterations   : $(params.MA_maxiter)")
    println(io, "  Simulation:")
    println(io, "    End time           : $(params.t_end)")
    println(io, "    Save frequency     : $(params.nsave)")
    println(io, "  Optional:")
    println(io, "    Spectral filter    : $(has_filter(params) ? "enabled" : "disabled")")
end

"""
    validate_params(params::Params) -> Bool
Validate parameter values and return true if all are reasonable.
"""
function validate_params(params::Params)
    valid = true
    
    if params.κ < 0
        @warn "Negative diffusivity κ = $(params.κ)"
        valid = false
    end
    
    if params.νh < 0
        @warn "Negative hyperdiffusivity νh = $(params.νh)"  
        valid = false
    end
    
    if params.p_h < 1
        @warn "Hyperdiffusion order p_h = $(params.p_h) should be ≥ 1"
        valid = false
    end
    
    if params.MA_tol <= 0
        @warn "Non-positive MA tolerance $(params.MA_tol)"
        valid = false
    end
    
    if params.MA_maxiter < 1
        @warn "MA max iterations $(params.MA_maxiter) should be ≥ 1"
        valid = false
    end
    
    if params.t_end <= 0
        @warn "Non-positive end time $(params.t_end)"
        valid = false
    end
    
    if params.nsave < 1
        @warn "Save frequency $(params.nsave) should be ≥ 1"
        valid = false
    end
    
    return valid
end