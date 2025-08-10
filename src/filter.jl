# src/filter.jl
# Spectral filtering for surface semi-geostrophic equations
# Supports exponential, hyperviscosity, and custom filter functions

using PencilArrays: range_local
using LinearAlgebra: mul!, ldiv!

###############################################################################
# SPECTRAL FILTERING FOR NUMERICAL STABILITY
###############################################################################
#
# FILTERING FUNCTIONS:
# 1. Exponential filter: exp(-α(k/k_c)^p) for high-wavenumber damping
# 2. Hyperviscosity filter: Based on (-1)^p ∇^{2p} operator
# 3. Sharp cutoff filter: Hard truncation at specified wavenumber
# 4. Cesàro filter: Smooth polynomial rolloff
# 5. Custom transfer functions
#
# APPLICATIONS:
# - Remove spurious high-frequency oscillations
# - Stabilize time integration schemes
# - Model subgrid-scale dissipation
# - Prevent aliasing errors in nonlinear terms
#
# IMPLEMENTATION:
# - MPI-compatible with PencilArrays
# - Efficient application in spectral space
# - Configurable filter parameters
# - Support for anisotropic filtering
#
###############################################################################

# ============================================================================
# FILTER STRUCTURES AND TYPES
# ============================================================================

"""
Abstract base type for spectral filters
"""
abstract type AbstractSpectralFilter{T<:AbstractFloat} end

"""
Exponential filter with configurable parameters
"""
struct ExponentialFilter{T<:AbstractFloat} <: AbstractSpectralFilter{T}
    strength::T      # Filter strength parameter α
    order::Int       # Filter order p  
    cutoff::T        # Cutoff wavenumber ratio (fraction of Nyquist)
    
    function ExponentialFilter{T}(strength::T=T(1.0), order::Int=4, cutoff::T=T(0.65)) where T
        @assert strength >= 0 "Filter strength must be non-negative"
        @assert order >= 1 "Filter order must be positive"
        @assert 0 < cutoff <= 1 "Cutoff must be between 0 and 1"
        new{T}(strength, order, cutoff)
    end
end

"""
Hyperviscosity filter: (-1)^p ∇^{2p}
"""
struct HyperviscosityFilter{T<:AbstractFloat} <: AbstractSpectralFilter{T}
    coefficient::T   # Hyperviscosity coefficient ν_p
    order::Int       # Order p (p=1 is regular viscosity)
    
    function HyperviscosityFilter{T}(coefficient::T, order::Int=2) where T
        @assert coefficient >= 0 "Hyperviscosity coefficient must be non-negative"
        @assert order >= 1 "Hyperviscosity order must be positive"
        new{T}(coefficient, order)
    end
end

"""
Sharp cutoff filter (ideal low-pass)
"""
struct CutoffFilter{T<:AbstractFloat} <: AbstractSpectralFilter{T}
    cutoff::T        # Cutoff wavenumber ratio
    
    function CutoffFilter{T}(cutoff::T=T(2/3)) where T
        @assert 0 < cutoff <= 1 "Cutoff must be between 0 and 1"
        new{T}(cutoff)
    end
end

"""
Cesàro filter with polynomial rolloff
"""
struct CesaroFilter{T<:AbstractFloat} <: AbstractSpectralFilter{T}
    cutoff::T        # Start of filter rolloff
    order::Int       # Polynomial order
    
    function CesaroFilter{T}(cutoff::T=T(0.5), order::Int=2) where T
        @assert 0 < cutoff <= 1 "Cutoff must be between 0 and 1"
        @assert order >= 1 "Cesàro order must be positive"
        new{T}(cutoff, order)
    end
end

"""
Custom filter with user-defined transfer function
"""
struct CustomFilter{T<:AbstractFloat, F} <: AbstractSpectralFilter{T}
    transfer_function::F
    parameters::NamedTuple
    
    function CustomFilter{T}(transfer_func::F, params::NamedTuple=NamedTuple()) where {T, F}
        new{T, F}(transfer_func, params)
    end
end

# ============================================================================
# FILTER TRANSFER FUNCTIONS
# ============================================================================

"""
    exponential_transfer(k_norm, filter::ExponentialFilter)

Compute exponential filter transfer function: exp(-α(k/k_c)^p)
"""
function exponential_transfer(k_norm::T, filter::ExponentialFilter{T}) where T
    if k_norm <= filter.cutoff
        return one(T)
    else
        ratio = (k_norm - filter.cutoff) / (one(T) - filter.cutoff)
        return exp(-filter.strength * ratio^filter.order)
    end
end

"""
    hyperviscosity_transfer(k_mag, dt, filter::HyperviscosityFilter)

Compute hyperviscosity filter transfer function: exp(-ν_p k^{2p} dt)
"""
function hyperviscosity_transfer(k_mag::T, dt::T, filter::HyperviscosityFilter{T}) where T
    return exp(-filter.coefficient * k_mag^(2*filter.order) * dt)
end

"""
    cutoff_transfer(k_norm, filter::CutoffFilter)

Compute sharp cutoff transfer function
"""
function cutoff_transfer(k_norm::T, filter::CutoffFilter{T}) where T
    return k_norm <= filter.cutoff ? one(T) : zero(T)
end

"""
    cesaro_transfer(k_norm, filter::CesaroFilter)

Compute Cesàro filter transfer function with polynomial rolloff
"""
function cesaro_transfer(k_norm::T, filter::CesaroFilter{T}) where T
    if k_norm <= filter.cutoff
        return one(T)
    elseif k_norm >= one(T)
        return zero(T)
    else
        ratio = (k_norm - filter.cutoff) / (one(T) - filter.cutoff)
        return (one(T) - ratio)^filter.order
    end
end

"""
    custom_transfer(k_norm, k_x, k_y, filter::CustomFilter)

Apply custom transfer function
"""
function custom_transfer(k_norm::T, k_x::T, k_y::T, filter::CustomFilter{T}) where T
    return filter.transfer_function(k_norm, k_x, k_y; filter.parameters...)
end

# ============================================================================
# MAIN FILTERING FUNCTIONS
# ============================================================================

"""
    apply_spectral_filter!(fields::Fields, domain::Domain, filter::AbstractSpectralFilter; dt=nothing)

Apply spectral filter to all prognostic fields in Fields structure.
For surface SSG, this filters the 2D surface buoyancy field.
"""
function apply_spectral_filter!(fields::Fields{T}, domain::Domain, 
                               filter::AbstractSpectralFilter{T}; 
                               dt::Union{T, Nothing}=nothing) where T
    
    # Apply filter to surface buoyancy field (2D)
    apply_filter_to_field!(fields.bₛ, fields.bshat, domain, filter; dt=dt)
    
    # Apply filter to surface streamfunction if needed (2D)
    apply_filter_to_field!(fields.φₛ, fields.φshat, domain, filter; dt=dt)
    
    return nothing
end

"""
    apply_filter_to_field!(field_real, field_spec, domain::Domain, filter::AbstractSpectralFilter; dt=nothing)

Apply spectral filter to a single field with proper 2D/3D handling.
"""
function apply_filter_to_field!(field_real::PencilArray{T, 2}, 
                               field_spec::PencilArray{Complex{T}, 2},
                               domain::Domain, 
                               filter::AbstractSpectralFilter{T};
                               dt::Union{T, Nothing}=nothing) where {T}
    
    # Transform to spectral space (2D)
    rfft_2d!(domain, field_real, field_spec)
    
    # Apply filter in spectral space
    apply_filter_spectral!(field_spec, domain, filter; dt=dt)
    
    # Transform back to physical space (2D)
    irfft_2d!(domain, field_spec, field_real)
    
    return nothing
end

# 3D version for compatibility with other parts of the code
function apply_filter_to_field!(field_real::PencilArray{T, 3}, 
                               field_spec::PencilArray{Complex{T}, 3},
                               domain::Domain, 
                               filter::AbstractSpectralFilter{T};
                               dt::Union{T, Nothing}=nothing) where {T}
    
    # Transform to spectral space (3D)
    rfft!(domain, field_real, field_spec)
    
    # Apply filter in spectral space
    apply_filter_spectral!(field_spec, domain, filter; dt=dt)
    
    # Transform back to physical space (3D)
    irfft!(domain, field_spec, field_real)
    
    return nothing
end

"""
    apply_filter_spectral!(field_spec, domain::Domain, filter::AbstractSpectralFilter; dt=nothing)

Apply filter directly to field in spectral space (most efficient).
"""
function apply_filter_spectral!(field_spec::PencilArray{Complex{T}, N}, 
                               domain::Domain,
                               filter::AbstractSpectralFilter{T};
                               dt::Union{T, Nothing}=nothing) where {T, N}
    
    # Get local data and ranges for MPI compatibility
    field_local = field_spec.data
    range_locals = range_local(field_spec.pencil)
    
    # Apply filter based on type
    if filter isa ExponentialFilter
        apply_exponential_filter!(field_local, range_locals, domain, filter)

    elseif filter isa HyperviscosityFilter
        @assert dt !== nothing "Hyperviscosity filter requires time step dt"
        apply_hyperviscosity_filter!(field_local, range_locals, domain, filter, dt)

    elseif filter isa CutoffFilter
        apply_cutoff_filter!(field_local, range_locals, domain, filter)

    elseif filter isa CesaroFilter
        apply_cesaro_filter!(field_local, range_locals, domain, filter)

    elseif filter isa CustomFilter
        apply_custom_filter!(field_local, range_locals, domain, filter)

    else
        error("Unknown filter type: $(typeof(filter))")
    end
    
    return nothing
end

# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

"""
    apply_exponential_filter!(field_local, range_locals, domain, filter)

Apply exponential filter with MPI domain decomposition.
"""
function apply_exponential_filter!(field_local::AbstractArray{Complex{T}, N},
                                  range_locals, domain::Domain,
                                  filter::ExponentialFilter{T}) where {T, N}
    
    # Precompute normalization factors
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    k_max_x = π / dx
    k_max_y = π / dy
    k_max = sqrt(k_max_x^2 + k_max_y^2)
    
    if N == 2
        # 2D fields (surface)
        @inbounds for (j_local, j_global) in enumerate(range_locals[2])
            ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
            for (i_local, i_global) in enumerate(range_locals[1])
                kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                
                # Compute normalized wavenumber magnitude
                k_mag = sqrt(kx^2 + ky^2)
                k_norm = k_mag / k_max
                
                # Apply exponential filter
                transfer = exponential_transfer(k_norm, filter)
                field_local[i_local, j_local] *= transfer
            end
        end
    else
        # 3D fields
        @inbounds for k in axes(field_local, 3)
            for (j_local, j_global) in enumerate(range_locals[2])
                ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
                for (i_local, i_global) in enumerate(range_locals[1])
                    kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                    
                    # Compute normalized wavenumber magnitude
                    k_mag = sqrt(kx^2 + ky^2)
                    k_norm = k_mag / k_max
                    
                    # Apply exponential filter
                    transfer = exponential_transfer(k_norm, filter)
                    field_local[i_local, j_local, k] *= transfer
                end
            end
        end
    end
end

"""
    apply_hyperviscosity_filter!(field_local, range_locals, domain, filter, dt)

Apply hyperviscosity filter: exp(-ν_p k^{2p} dt)
"""
function apply_hyperviscosity_filter!(field_local::AbstractArray{Complex{T}, N},
                                     range_locals, domain::Domain,
                                     filter::HyperviscosityFilter{T},
                                     dt::T) where {T, N}
    
    if N == 2
        # 2D fields (surface)
        @inbounds for (j_local, j_global) in enumerate(range_locals[2])
            ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
            for (i_local, i_global) in enumerate(range_locals[1])
                kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                
                # Compute wavenumber magnitude
                k_mag = sqrt(kx^2 + ky^2)
                
                # Apply hyperviscosity filter
                transfer = hyperviscosity_transfer(k_mag, dt, filter)
                field_local[i_local, j_local] *= transfer
            end
        end
    else
        # 3D fields
        @inbounds for k in axes(field_local, 3)
            for (j_local, j_global) in enumerate(range_locals[2])
                ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
                for (i_local, i_global) in enumerate(range_locals[1])
                    kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                    
                    # Compute wavenumber magnitude
                    k_mag = sqrt(kx^2 + ky^2)
                    
                    # Apply hyperviscosity filter
                    transfer = hyperviscosity_transfer(k_mag, dt, filter)
                    field_local[i_local, j_local, k] *= transfer
                end
            end
        end
    end
end

"""
    apply_cutoff_filter!(field_local, range_locals, domain, filter)

Apply sharp cutoff filter (2/3 rule or custom cutoff).
"""
function apply_cutoff_filter!(field_local::AbstractArray{Complex{T}, N},
                             range_locals, domain::Domain,
                             filter::CutoffFilter{T}) where {T, N}
    
    # Precompute normalization
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    k_max_x = π / dx
    k_max_y = π / dy
    k_max = sqrt(k_max_x^2 + k_max_y^2)
    
    if N == 2
        # 2D fields (surface)
        @inbounds for (j_local, j_global) in enumerate(range_locals[2])
            ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
            for (i_local, i_global) in enumerate(range_locals[1])
                kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                
                # Compute normalized wavenumber
                k_mag = sqrt(kx^2 + ky^2)
                k_norm = k_mag / k_max
                
                # Apply cutoff filter
                transfer = cutoff_transfer(k_norm, filter)
                field_local[i_local, j_local] *= transfer
            end
        end
    else
        # 3D fields
        @inbounds for k in axes(field_local, 3)
            for (j_local, j_global) in enumerate(range_locals[2])
                ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
                for (i_local, i_global) in enumerate(range_locals[1])
                    kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                    
                    # Compute normalized wavenumber
                    k_mag = sqrt(kx^2 + ky^2)
                    k_norm = k_mag / k_max
                    
                    # Apply cutoff filter
                    transfer = cutoff_transfer(k_norm, filter)
                    field_local[i_local, j_local, k] *= transfer
                end
            end
        end
    end
end

"""
    apply_cesaro_filter!(field_local, range_locals, domain, filter)

Apply Cesàro filter with polynomial rolloff.
"""
function apply_cesaro_filter!(field_local::AbstractArray{Complex{T}, N},
                             range_locals, domain::Domain,
                             filter::CesaroFilter{T}) where {T, N}
    
    # Precompute normalization
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny

    k_max_x = π / dx
    k_max_y = π / dy
    k_max = sqrt(k_max_x^2 + k_max_y^2)
    
    @inbounds for k in axes(field_local, 3)
        for (j_local, j_global) in enumerate(range_locals[2])
            ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
            for (i_local, i_global) in enumerate(range_locals[1])
                kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                
                # Compute normalized wavenumber
                k_mag = sqrt(kx^2 + ky^2)
                k_norm = k_mag / k_max
                
                # Apply Cesàro filter
                transfer = cesaro_transfer(k_norm, filter)
                field_local[i_local, j_local, k] *= transfer
            end
        end
    end
end

"""
    apply_custom_filter!(field_local, range_locals, domain, filter)

Apply user-defined custom filter.
"""
function apply_custom_filter!(field_local::AbstractArray{Complex{T}, N},
                             range_locals, domain::Domain,
                             filter::CustomFilter{T}) where {T, N}
    
    # Precompute normalization if needed
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny

    k_max_x = π / dx
    k_max_y = π / dy
    k_max = sqrt(k_max_x^2 + k_max_y^2)
    
    @inbounds for k in axes(field_local, 3)
        for (j_local, j_global) in enumerate(range_locals[2])
            ky = j_global <= length(domain.ky) ? domain.ky[j_global] : zero(T)
            for (i_local, i_global) in enumerate(range_locals[1])
                kx = i_global <= length(domain.kx) ? domain.kx[i_global] : zero(T)
                
                # Compute normalized wavenumber
                k_mag = sqrt(kx^2 + ky^2)
                k_norm = k_mag / k_max
                
                # Apply custom filter
                transfer = custom_transfer(k_norm, kx, ky, filter)
                field_local[i_local, j_local, k] *= transfer
            end
        end
    end
end

# # ============================================================================
# # LEGACY INTERFACE FROM TIMESTEP.JL
# # ============================================================================

# """
#     apply_spectral_filter!(fields::Fields, domain::Domain, filter_strength::Real)

# Legacy interface matching timestep.jl - applies exponential filter.
# """
# function apply_spectral_filter!(fields::Fields{T}, domain::Domain, 
#                                filter_strength::Real) where T
    
#     # Create exponential filter with legacy parameters
#     filter = ExponentialFilter{T}(T(filter_strength), 4, T(0.65))
    
#     # Apply to buoyancy field
#     apply_spectral_filter!(fields, domain, filter)
    
#     return nothing
# end

# """
#     apply_exponential_filter!(bhat, domain::Domain, strength::Real)

# Direct exponential filter application (from timestep.jl).
# """
# function apply_exponential_filter!(bhat::PencilArray{Complex{T}, N}, 
#                                   domain::Domain, strength::T) where {T, N}
    
#     filter = ExponentialFilter{T}(strength, 4, T(0.65))
#     apply_filter_spectral!(bhat, domain, filter)
    
#     return nothing
# end

# ============================================================================
# FILTER UTILITIES AND DIAGNOSTICS
# ============================================================================

"""
    create_filter(filter_type::Symbol, params...; kwargs...)

Factory function for creating filters.
"""
function create_filter(filter_type::Symbol, T::Type=Float64; kwargs...)
    if filter_type == :exponential
        return ExponentialFilter{T}(; kwargs...)
    elseif filter_type == :hyperviscosity
        return HyperviscosityFilter{T}(; kwargs...)
    elseif filter_type == :cutoff
        return CutoffFilter{T}(; kwargs...)
    elseif filter_type == :cesaro
        return CesaroFilter{T}(; kwargs...)
    else
        error("Unknown filter type: $filter_type")
    end
end

"""
    filter_energy_removal(field_before, field_after, domain::Domain)

Compute energy removed by filtering operation.
"""
function filter_energy_removal(field_before::PencilArray{T, N}, 
                              field_after::PencilArray{T, N}, 
                              domain::Domain) where {T, N}
    
    # Create temporary spectral arrays
    spec_before = similar(field_before, Complex{T})
    spec_after = similar(field_after, Complex{T})
    
    # Transform to spectral space
    rfft!(domain, field_before, spec_before)
    rfft!(domain, field_after,  spec_after)
    
    # Compute energy using Parseval's theorem
    energy_before = 0.5 * parsevalsum2(spec_before, domain)
    energy_after = 0.5 * parsevalsum2(spec_after, domain)
    
    return energy_before - energy_after
end

"""
    plot_filter_response(filter::AbstractSpectralFilter, domain::Domain)

Generate filter response function for visualization.
"""
function plot_filter_response(filter::AbstractSpectralFilter{T}, 
                             domain::Domain; npoints::Int=1000) where T
    
    # Create wavenumber range
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny

    k_max = π / min(dx, dy)
    k_range = range(zero(T), k_max, length=npoints)
    
    # Compute transfer function
    transfer = similar(k_range)
    for (i, k) in enumerate(k_range)
        k_norm = k / k_max
        if filter isa ExponentialFilter
            transfer[i] = exponential_transfer(k_norm, filter)
        elseif filter isa CutoffFilter
            transfer[i] = cutoff_transfer(k_norm, filter)
        elseif filter isa CesaroFilter
            transfer[i] = cesaro_transfer(k_norm, filter)
        else
            transfer[i] = one(T)  # Default
        end
    end
    
    return collect(k_range), collect(transfer)
end

"""
    validate_filter(filter::AbstractSpectralFilter)

Validate filter parameters and warn about potential issues.
"""
function validate_filter(filter::AbstractSpectralFilter{T}) where T
    if filter isa ExponentialFilter
        if filter.strength > 10
            @warn "Very strong exponential filter (α=$(filter.strength)) may cause excessive damping"
        end
        if filter.cutoff < 0.3
            @warn "Low cutoff frequency ($(filter.cutoff)) may affect resolved scales"
        end
    elseif filter isa HyperviscosityFilter
        if filter.order > 4
            @warn "High hyperviscosity order ($(filter.order)) may cause numerical instability"
        end
    end
    
    return true
end

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

"""
    dealiasing_filter(domain::Domain)

Create standard 2/3 dealiasing filter.
"""
function dealiasing_filter(domain::Domain, T::Type=Float64)
    return CutoffFilter{T}(T(2/3))
end

"""
    stabilizing_filter(domain::Domain; strength=0.1)

Create moderate exponential filter for numerical stability.
"""
function stabilizing_filter(domain::Domain, T::Type=Float64; strength::Real=0.1)
    return ExponentialFilter{T}(T(strength), 4, T(0.8))
end

"""
    subgrid_filter(domain::Domain; coefficient=1e-4, order=2)

Create hyperviscosity filter for subgrid-scale modeling.
"""
function subgrid_filter(domain::Domain, T::Type=Float64; coefficient::Real=1e-4, order::Int=2)
    return HyperviscosityFilter{T}(T(coefficient), order)
end

"""
    Base.show(io::IO, filter::AbstractSpectralFilter)

Pretty print filter information.
"""
function Base.show(io::IO, filter::ExponentialFilter{T}) where T
    println(io, "ExponentialFilter{$T}:")
    println(io, "  strength: $(filter.strength)")
    println(io, "  order   : $(filter.order)")
    println(io, "  cutoff  : $(filter.cutoff)")
end

function Base.show(io::IO, filter::HyperviscosityFilter{T}) where T
    println(io, "HyperviscosityFilter{$T}:")
    println(io, "  coefficient: $(filter.coefficient)")
    println(io, "  order      : $(filter.order)")
end

function Base.show(io::IO, filter::CutoffFilter{T}) where T
    println(io, "CutoffFilter{$T}:")
    println(io, "  cutoff: $(filter.cutoff)")
end


