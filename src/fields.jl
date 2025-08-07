# src/fields.jl
# Field allocation and management

using PencilArrays: PencilArray

"""
    struct Fields{T, PA, PC}
Container for all simulation fields (prognostic, diagnostic, and scratch).

# Fields
## Prognostic
- `bₛ`: Surface buoyancy anomaly
- `φₛ`: Surface streamfunction extracted from 3D φ 

## Diagnostic  
- `φ`: Geostrophic streamfunction
- `u, v`: Semi-geostrophic velocity components

## Scratch arrays (real space)
- `R`: Residual for Monge-Ampère equation
- `tmp, tmp2`: General scratch arrays

## Spectral arrays
- `bhat, φhat`: Spectral versions of b and φ
- `tmpc`: Spectral scratch array
"""
struct Fields{T, PR2D, PR3D, PC2D, PC3D}
    # Prognostic fields (2D surface)
    bₛ::PencilArray{T, 2, PR2D}                     # surface buoyancy anomaly (2D)
    φₛ::PencilArray{T, 2, PR2D}                     # surface streamfunction (2D)
    
    # Diagnostic fields (3D for solver)
    φ::PencilArray{T, 3, PR3D}                      # 3D streamfunction for solver

    u::PencilArray{T, 3, PR3D}                      # surface u-velocity (2D)
    v::PencilArray{T, 3, PR3D}                      # surface v-velocity (2D)
    
    # Multigrid workspace (3D)
    φ_mg::PencilArray{T, 3, PR3D}                   # Multigrid solution workspace
    b_mg::PencilArray{T, 3, PR3D}                   # Multigrid RHS workspace
    
    # Scratch arrays (2D for surface)
    R::PencilArray{T,    2, PR2D}                   # residual for MA equation (2D)
    tmp::PencilArray{T,  2, PR2D}                   # general scratch (2D)
    tmp2::PencilArray{T, 2, PR2D}                   # additional scratch (2D)
    tmp3::PencilArray{T, 2, PR2D}                   # additional scratch (2D)
    
    # Spectral arrays (2D)
    bshat::PencilArray{Complex{T}, 2, PC2D}         # spectral surface buoyancy
    φshat::PencilArray{Complex{T}, 2, PC2D}         # spectral surface streamfunction

    φhat::PencilArray{Complex{T},  3, PC3D}         # spectral surface streamfunction

    tmpc_2d::PencilArray{Complex{T},  2, PC2D}      # spectral scratch
    tmpc1_2d::PencilArray{Complex{T}, 2, PC2D}      # spectral scratch

    tmpc_3d::PencilArray{Complex{T},  3, PC3D}      # spectral scratch
    tmpc1_3d::PencilArray{Complex{T}, 3, PC3D}      # spectral scratch
end

"""
    allocate_fields(domain::Domain{T}) -> Fields{T}
Allocate all field arrays for the given domain.

# Arguments
- `domain`: Domain structure

# Returns
- `Fields`: Allocated field structure
"""
function allocate_fields(domain::Domain{T}) where T
    # 2D fields for surface
    bₛ   = PencilArray{T}(undef, domain.pr2d)
    φₛ   = PencilArray{T}(undef, domain.pr2d)
    
    # 3D field for solver
    φ    = PencilArray{T}(undef, domain.pr3d)
    
    # 3D velocity fields
    u    = PencilArray{T}(undef, domain.pr3d)
    v    = PencilArray{T}(undef, domain.pr3d)
    
    # 3D arrays for multigrid
    φ_mg = PencilArray{T}(undef, domain.pr3d)
    b_mg = PencilArray{T}(undef, domain.pr3d)

    # 2D scratch arrays
    R    = PencilArray{T}(undef, domain.pr2d)
    tmp  = PencilArray{T}(undef, domain.pr2d)
    tmp2 = PencilArray{T}(undef, domain.pr2d)
    tmp3 = PencilArray{T}(undef, domain.pr2d)
    
    # 2D Spectral fields
    bshat  = PencilArray{Complex{T}}(undef, domain.pc2d)
    φshat  = PencilArray{Complex{T}}(undef, domain.pc2d)

    φhat   = PencilArray{Complex{T}}(undef, domain.pc3d)

    tmpc_2d  = PencilArray{Complex{T}}(undef, domain.pc2d)
    tmpc2_2d = PencilArray{Complex{T}}(undef, domain.pc2d)

    tmpc_3d  = PencilArray{Complex{T}}(undef, domain.pc3d)
    tmpc2_3d = PencilArray{Complex{T}}(undef, domain.pc3d)
    
    return Fields{T, typeof(domain.pr2d), typeof(domain.pr3d), 
                  typeof(domain.pc2d), typeof(domain.pc3d)}(
        bₛ, φₛ, φ, u, v, 
        φ_mg, b_mg, 
        R, tmp, tmp2, tmp3,
        bshat, φshat, φhat, 
        tmpc_2d, tmpc2_2d,
        tmpc_3d, tmpc2_3d,
    )
end


"""
    zero_fields!(fields::Fields)

Set all fields to zero.
"""
function zero_fields!(fields::Fields)
    for f in fieldnames(Fields)
        arr = getfield(fields, f)
        arr.data .= zero(eltype(arr))
    end
    return fields
end


function ensure_same_grid(dest::PencilArray, src::PencilArray)
    # Check compatible sizes
    if size(dest) != size(src)
        throw(ArgumentError("PencilArrays have incompatible sizes: $(size(dest)) vs $(size(src))"))
    end
    
    # Check MPI communicator compatibility
    if dest.pencil.comm != src.pencil.comm
        throw(ArgumentError("PencilArrays have different MPI communicators"))
    end
    
    return true
end


macro ensuresamegrid(dest, src)
    return quote
        ensure_same_grid($(esc(dest)), $(esc(src)))
    end
end


"""
    copy_field!(dest, src)
Copy one field to another (must be same type and size).
"""
function copy_field!(dest::PencilArray, src::PencilArray)
    @ensuresamegrid(dest, src)
    dest.data .= src.data
    #return dest
end


"""
    zero_field!(field)
Set all values in a PencilArray to zero.
"""
zero_field!(field::PencilArray) = field.data .= zero(eltype(field))


"""
    field_stats(field) -> NamedTuple

Compute basic statistics for a single field.
"""
function field_stats(field::PencilArray{T}) where T
    field_data = field.data
    
    # Local statistics
    local_mean  = mean(field_data)
    local_min   = minimum(field_data)
    local_max   = maximum(field_data)
    local_count = length(field_data)
    
    # Global statistics via MPI (if available)
    if MPI.Initialized() && MPI.Comm_size(field.pencil.comm) > 1
        global_sum   = MPI.Allreduce(local_mean * local_count, MPI.SUM, field.pencil.comm)
        global_count = MPI.Allreduce(local_count, MPI.SUM, field.pencil.comm)
        global_mean  = global_sum / global_count
        
        global_min = MPI.Allreduce(local_min, MPI.MIN, field.pencil.comm)
        global_max = MPI.Allreduce(local_max, MPI.MAX, field.pencil.comm)
        
        # Global variance
        local_var_contrib = sum((field_data .- global_mean).^2)
        global_var = MPI.Allreduce(local_var_contrib, MPI.SUM, field.pencil.comm) / global_count
        global_std = sqrt(global_var)
    else
        # Single process case
        global_mean = local_mean
        global_min = local_min
        global_max = local_max
        global_std = std(field_data)
        global_count = local_count
    end
    
    return (
        mean = global_mean,
        std = global_std,
        min = global_min,
        max = global_max,
        count = global_count
    )
end


"""
    enhanced_field_stats(fields::Fields) -> Dict

Compute enhanced statistics for all fields in Fields structure.
"""
function enhanced_field_stats(fields::Fields)
    stats = Dict{Symbol, NamedTuple}()
    
    # Get all field names that are PencilArrays with real element types
    for field_name in fieldnames(typeof(fields))
        field = getfield(fields, field_name)
        if isa(field, PencilArray) && eltype(field) <: Real
            stats[field_name] = field_stats(field)
        end
    end
    
    return stats
end


"""
    Base.show(io::IO, fields::Fields)
Pretty print field information.
"""
function Base.show(io::IO, fields::Fields)
    T = eltype(fields.bₛ)
    println(io, "Fields{$T}:")
    println(io, "  Prognostic: bₛ (buoyancy)")
    println(io, "  Diagnostic: φ (streamfunction), u, v (velocities)")
    println(io, "  Scratch   : R, tmp, tmp2 (real), tmpc (spectral)")
    println(io, "  Local size: $(size(fields.bₛ))")
end