# src/fields.jl
# Field allocation and management

using PencilArrays: local_size, PencilArray

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
struct Fields{T, PR, PC}
    # Prognostic fields
    bₛ::PencilArray{T, PR}           # surface buoyancy anomaly
    φₛ::PencilArray{T, PR}           # 3D streamfunction at the surface
    

    # Diagnostic fields
    φ::PencilArray{T, PR}           # 3D streamfunction
    u::PencilArray{T, PR}           # geostrophic x-velocity (3D)
    v::PencilArray{T, PR}           # geostrophic y-velocity (3D)

    # Multigrid workspace
    φ_mg::PencilArray{T, PR}           # Multigrid solution workspace
    b_mg::PencilArray{T, PR}           # Multigrid RHS workspace
    
    # Scratch arrays (real space)
    R::PencilArray{T, PR}           # residual for MA equation
    tmp::PencilArray{T, PR}         # general scratch
    tmp2::PencilArray{T, PR}        # additional scratch
    tmp3::PencilArray{T, PR}        # additional scratch
    
    # Spectral arrays
    bhat::PencilArray{Complex{T},PC}        # spectral buoyancy
    φhat::PencilArray{Complex{T},PC}        # spectral streamfunction
    tmpc::PencilArray{Complex{T},PC}        # spectral scratch
    tmpc2::PencilArray{Complex{T},PC}       # spectral scratch
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
    # Real-space fields
    bₛ   = PencilArray{T}(undef, domain.pr)
    φₛ   = PencilArray{T}(undef, domain.pr)
    φ    = PencilArray{T}(undef, domain.pr)

    u    = PencilArray{T}(undef, domain.pr)
    v    = PencilArray{T}(undef, domain.pr)
    
    R    = PencilArray{T}(undef, domain.pr)
    tmp  = PencilArray{T}(undef, domain.pr)
    tmp2 = PencilArray{T}(undef, domain.pr)
    tmp3 = PencilArray{T}(undef, domain.pr)
    
    φ_mg = PencilArray{T}(undef, domain.pr)
    b_mg = PencilArray{T}(undef, domain.pr)

    # Spectral fields
    bhat  = PencilArray{Complex{T}}(undef, domain.pc)
    φhat  = PencilArray{Complex{T}}(undef, domain.pc)
    tmpc  = PencilArray{Complex{T}}(undef, domain.pc)
    tmpc2 = PencilArray{Complex{T}}(undef, domain.pc)
    
    return Fields{T, typeof(b), typeof(bhat)}(
        bₛ, φₛ, φ, u, v, 
        φ_mg, b_mg, R, 
        tmp, tmp2, tmp3, 
        bhat, φhat, 
        tmpc, tmpc2
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