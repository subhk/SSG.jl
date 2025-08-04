# This code implements a coordinate transformation between Eulerian (physical) 
# coordinates (x,y) and geostrophic coordinates (X,Y) for geophysical fluid dynamics.
#
# The transformation is defined by:
#   X = x + ε ∂Φ/∂x
#   Y = y + ε ∂Φ/∂y
#
# where:
# - Φ is the Montgomery potential (streamfunction for geostrophic flow)
# - ε is a small parameter controlling the coordinate transformation strength
# - The gradients ∂Φ/∂x, ∂Φ/∂y represent geostrophic velocity components
#
# MATHEMATICAL APPROACH:
# The transformation is solved iteratively using Newton's method since the 
# mapping X,Y depends on gradients of Φ evaluated at the unknown transformed 
# coordinates. The algorithm:
# 1. Computes spatial derivatives of Φ using FFT-based spectral differentiation
# 2. Creates periodic extensions to handle boundary conditions properly
# 3. Iteratively solves for X,Y using interpolated gradient fields
# 4. Maps fields (buoyancy, potential, vorticity) onto the new coordinate system


# using FFTW
# using Statistics: mean
# using LinearAlgebra
# using Interpolations
# using Base.Threads
# using JLD2

# Custom exception types for better error handling
struct ConvergenceError <: Exception
    msg::String
end

struct InterpolationError <: Exception
    msg::String
end

"""
    GeostrophicMapping

Struct to hold precomputed data and configurations for geostrophic mapping.
This allows for better memory management and reuse of expensive computations.
"""
struct GeostrophicMapping{T<:AbstractFloat}
    # Grid parameters
    nx::Int
    ny::Int
    dx::T
    dy::T
    
    # FFT plans (thread-safe when created)
    plan_forward::FFTW.rFFTWPlan{T,-1,false,2}
    plan_backward::FFTW.rFFTWPlan{Complex{T},1,false,2}
    
    # Wavenumber arrays
    kr::Vector{T}
    l::Vector{T}
    
    # Preallocated work arrays
    work_complex::Matrix{Complex{T}}
    work_real1::Matrix{T}
    work_real2::Matrix{T}
    
    function GeostrophicMapping(xg::AbstractVector{T}, yg::AbstractVector{T}, 
                               kr::AbstractVector{T}, l::AbstractVector{T}) where {T<:AbstractFloat}
        nx, ny = length(xg), length(yg)
        
        # Validate inputs
        @assert nx > 1 && ny > 1 "Grid must have at least 2 points in each dimension"
        @assert length(kr) == nx÷2 + 1 "kr length must match rfft output size"
        @assert length(l) == ny "l length must match y dimension"
        
        # Check for uniform spacing (within tolerance)
        dx = xg[2] - xg[1]
        dy = yg[2] - yg[1]
        if nx > 2
            dx_check = maximum(diff(xg)) - minimum(diff(xg))
            @assert dx_check < 1e-12 * abs(dx) "x grid must be uniformly spaced"
        end
        if ny > 2
            dy_check = maximum(diff(yg)) - minimum(diff(yg))
            @assert dy_check < 1e-12 * abs(dy) "y grid must be uniformly spaced"
        end
        
        # Create optimized FFT plans
        dummy_real = zeros(T, nx, ny)
        dummy_complex = zeros(Complex{T}, nx÷2+1, ny)
        
        plan_forward = plan_rfft(dummy_real; flags=FFTW.MEASURE)
        plan_backward = plan_irfft(dummy_complex, nx; flags=FFTW.MEASURE)
        
        # Preallocate work arrays
        work_complex = similar(dummy_complex)
        work_real1 = similar(dummy_real)
        work_real2 = similar(dummy_real)
        
        new{T}(nx, ny, dx, dy, plan_forward, plan_backward, 
               Vector{T}(kr), Vector{T}(l), work_complex, work_real1, work_real2)
    end
end

"""
    compute_derivatives!(mapping, field, ∂x, ∂y, ∂xx, ∂yy)

Efficiently compute spatial derivatives using FFT with preallocated arrays.
"""
function compute_derivatives!(mapping::GeostrophicMapping{T}, 
                             field::AbstractMatrix{T},
                             ∂x::AbstractMatrix{T}, 
                             ∂y::AbstractMatrix{T},
                             ∂xx::AbstractMatrix{T}, 
                             ∂yy::AbstractMatrix{T}) where {T}
    
    # Forward FFT
    copyto!(mapping.work_real1, field)
    mul!(mapping.work_complex, mapping.plan_forward, mapping.work_real1)
    
    # Compute ∂/∂x
    @. mapping.work_complex = im * mapping.kr * mapping.work_complex
    mul!(∂x, mapping.plan_backward, mapping.work_complex)
    
    # Compute ∂²/∂x²
    @. mapping.work_complex = im * mapping.kr * mapping.work_complex
    mul!(∂xx, mapping.plan_backward, mapping.work_complex)
    
    # Recompute FFT for y derivatives
    copyto!(mapping.work_real1, field)
    mul!(mapping.work_complex, mapping.plan_forward, mapping.work_real1)
    
    # Compute ∂/∂y
    @. mapping.work_complex = im * mapping.l' * mapping.work_complex
    mul!(∂y, mapping.plan_backward, mapping.work_complex)
    
    # Compute ∂²/∂y²
    @. mapping.work_complex = im * mapping.l' * mapping.work_complex
    mul!(∂yy, mapping.plan_backward, mapping.work_complex)
    
    return nothing
end

"""
    create_periodic_extension(field, pad_factor=1)

Create periodic extension of field with specified padding factor.
Returns extended field and corresponding coordinate arrays.
"""
function create_periodic_extension(field::AbstractMatrix{T}, 
                                 xg::AbstractVector{T}, 
                                 yg::AbstractVector{T},
                                 pad_factor::Int=1) where {T}
    
    nx, ny = size(field)
    dx, dy = xg[2] - xg[1], yg[2] - yg[1]
    
    # Create extended field
    ext_size = (2*pad_factor + 1)
    field_ext = repeat(field, ext_size, ext_size)
    
    # Create extended coordinate arrays
    Lx, Ly = (nx-1)*dx, (ny-1)*dy
    x_ext = range(xg[1] - pad_factor*Lx, xg[end] + pad_factor*Lx, 
                  length=ext_size*nx)
    y_ext = range(yg[1] - pad_factor*Ly, yg[end] + pad_factor*Ly, 
                  length=ext_size*ny)
    
    return field_ext, x_ext, y_ext
end

"""
    newton_iteration!(X, Y, xg, yg, itp_∂Φx, itp_∂Φy, ε, tol, maxiter)

Perform Newton iterations with convergence checking.
"""
function newton_iteration!(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
                          xg::AbstractVector{T}, yg::AbstractVector{T},
                          itp_∂Φx, itp_∂Φy, ε::T, 
                          tol::T=1e-10, maxiter::Int=50) where {T}
    
    ∂Φx_interp = similar(X)
    ∂Φy_interp = similar(Y)
    
    converged = false
    
    for iter in 1:maxiter
        # Interpolate gradients at current positions
        @threads for I in CartesianIndices(X)
            try
                ∂Φx_interp[I] = itp_∂Φx(X[I], Y[I])
                ∂Φy_interp[I] = itp_∂Φy(X[I], Y[I])
            catch e
                throw(InterpolationError("Interpolation failed at iteration $iter, index $I: $e"))
            end
        end
        
        # Store previous values for convergence check
        X_old = copy(X)
        Y_old = copy(Y)
        
        # Newton update
        @. X = xg' + ε * ∂Φx_interp
        @. Y = yg  + ε * ∂Φy_interp
        
        # Check convergence
        max_change = max(maximum(abs.(X - X_old)), maximum(abs.(Y - Y_old)))
        
        if max_change < tol
            @info "Newton iteration converged after $iter iterations (max change: $(max_change))"
            converged = true
            break
        end
        
        if iter % 10 == 0
            @info "Newton iteration $iter, max change: $(max_change)"
        end
    end
    
    if !converged
        @warn "Newton iteration did not converge after $maxiter iterations"
    end
    
    return converged
end

"""
    map_geostrophic(xg, yg, b_g, Φ_g, mapping, ε; kwargs...)

Main function for geostrophic coordinate mapping with improved error handling and performance.
"""
function map_geostrophic(xg::AbstractVector{T},
                         yg::AbstractVector{T},
                         b_g::AbstractMatrix{T},
                         Φ_g::AbstractMatrix{T},
                         mapping::GeostrophicMapping{T},
                         ε::T;
                         maxiter::Int = 50,
                         tol::T = 1e-10,
                         pad_factor::Int = 1,
                         use_scipy::Bool = true) where {T<:AbstractFloat}

    @assert size(Φ_g) == size(b_g) "Φ_g and b_g must have same dimensions"
    @assert length(xg) == size(Φ_g, 1) "xg length must match first dimension of Φ_g"
    @assert length(yg) == size(Φ_g, 2) "yg length must match second dimension of Φ_g"
    @assert abs(ε) < 1.0 "ε should be much less than 1 for convergence"

    nx, ny = length(xg), length(yg)

    # ------------------------------------------------------------------
    # 1. Compute gradients and vorticity efficiently
    # ------------------------------------------------------------------
    ∂Φx = zeros(T, nx, ny)
    ∂Φy = zeros(T, nx, ny)
    ∂²Φx² = zeros(T, nx, ny)
    ∂²Φy² = zeros(T, nx, ny)
    
    compute_derivatives!(mapping, Φ_g, ∂Φx, ∂Φy, ∂²Φx², ∂²Φy²)
    
    ζ_g = ∂²Φx² .+ ∂²Φy²  # Vorticity

    # ------------------------------------------------------------------
    # 2. Create periodic extensions
    # ------------------------------------------------------------------
    ∂Φx_ext, x_ext, y_ext = create_periodic_extension(∂Φx, xg, yg, pad_factor)
    ∂Φy_ext, _, _ = create_periodic_extension(∂Φy, xg, yg, pad_factor)
    Φ_ext, _, _ = create_periodic_extension(Φ_g, xg, yg, pad_factor)
    b_ext, _, _ = create_periodic_extension(b_g, xg, yg, pad_factor)
    ζ_ext, _, _ = create_periodic_extension(ζ_g, xg, yg, pad_factor)

    # Create interpolation objects (thread-safe)
    itp_∂Φx = interpolate((x_ext, y_ext), ∂Φx_ext, Gridded(Linear()))
    itp_∂Φy = interpolate((x_ext, y_ext), ∂Φy_ext, Gridded(Linear()))

    # ------------------------------------------------------------------
    # 3. Newton iterations with convergence monitoring
    # ------------------------------------------------------------------
    X = repeat(xg', ny, 1)  # Initialize coordinate arrays
    Y = repeat(yg, 1, nx)

    converged = newton_iteration!(X, Y, xg, yg, itp_∂Φx, itp_∂Φy, ε, tol, maxiter)
    
    if !converged
        throw(ConvergenceError("Newton iteration failed to converge"))
    end

    # ------------------------------------------------------------------
    # 4. Final interpolation onto the non-uniform grid
    # ------------------------------------------------------------------
    if use_scipy
        # Use SciPy for scattered interpolation (more robust)
        try
            using PyCall
            si = pyimport("scipy.interpolate")
            
            # Flatten coordinates for scipy
            x_flat = vec(repeat(x_ext', length(y_ext), 1))
            y_flat = vec(repeat(y_ext, 1, length(x_ext)))
            coords = hcat(x_flat, y_flat)
            
            Φ_result = reshape(si.griddata(coords, vec(Φ_ext), hcat(vec(X), vec(Y)), 
                                         method="cubic"), size(X))
            b_result = reshape(si.griddata(coords, vec(b_ext), hcat(vec(X), vec(Y)), 
                                         method="cubic"), size(X))
            ζ_result = reshape(si.griddata(coords, vec(ζ_ext), hcat(vec(X), vec(Y)), 
                                         method="cubic"), size(X))
        catch e
            @warn "SciPy interpolation failed, falling back to Julia interpolation: $e"
            use_scipy = false
        end
    end
    
    if !use_scipy
        # Fallback to Julia interpolation
        itp_Φ = interpolate((x_ext, y_ext), Φ_ext, Gridded(Linear()))
        itp_b = interpolate((x_ext, y_ext), b_ext, Gridded(Linear()))
        itp_ζ = interpolate((x_ext, y_ext), ζ_ext, Gridded(Linear()))
        
        Φ_result = [itp_Φ(X[i,j], Y[i,j]) for i in 1:nx, j in 1:ny]
        b_result = [itp_b(X[i,j], Y[i,j]) for i in 1:nx, j in 1:ny]
        ζ_result = [itp_ζ(X[i,j], Y[i,j]) for i in 1:nx, j in 1:ny]
    end

    return Φ_result, b_result, ζ_result, X, Y
end

"""
    create_geostrophic_mapping(xg, yg, Lx, Ly)

Convenience function to create a GeostrophicMapping object.
"""
function create_geostrophic_mapping(xg::AbstractVector{T}, yg::AbstractVector{T}, 
                                   Lx::T, Ly::T) where {T<:AbstractFloat}
    nx, ny = length(xg), length(yg)
    kr = rfftfreq(nx, 2π/Lx*nx)
    l = fftfreq(ny, 2π/Ly*ny)
    
    return GeostrophicMapping(xg, yg, kr, l)
end

# --------------------------------------------------------------------------
# Example 
# --------------------------------------------------------------------------
# function demo(; fname="ep_0.6/SurfaceSG_nx_512.jld2", snap=20001, ε=0.2,
#               maxiter=50, tol=1e-10, use_scipy=true)
    
#     # Set up threading
#     FFTW.set_num_threads(Threads.nthreads())
#     @info "Running with $(Threads.nthreads()) threads, FFTW threads: $(FFTW.get_num_threads())"

#     # Load data with error handling
#     try
#         jldopen(fname, "r") do file
#             x = file["/grid/x"]
#             y = file["/grid/y"]
#             Φ = file["/snapshots/Φ/$(snap)"]
#             b = file["/snapshots/b/$(snap)"]
#             Lx = file["/grid/Lx"]
#             Ly = file["/grid/Ly"]
            
#             @info "Loaded data: nx=$(length(x)), ny=$(length(y)), ε=$ε"
            
#             # Create mapping object
#             mapping = create_geostrophic_mapping(x, y, Lx, Ly)
            
#             # Perform mapping
#             @time Φg, bg, ζg, X, Y = map_geostrophic(x, y, b, Φ, mapping, ε; 
#                                                     maxiter=maxiter, tol=tol, use_scipy=use_scipy)
            
#             @info "Mapping completed successfully"
#             @info "Final grid displacement: max(|X-x|) = $(maximum(abs.(X .- x')))"
#             @info "                        max(|Y-y|) = $(maximum(abs.(Y .- y)))"
            
#             return Φg, bg, ζg, X, Y
#         end
#     catch e
#         if isa(e, SystemError)
#             error("Could not open file $fname: $(e.msg)")
#         else
#             rethrow(e)
#         end
#     end
# end
