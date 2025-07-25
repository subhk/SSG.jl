# src/fields.jl
# Field allocation and management

"""
    struct Fields{T, PA, PC}
Container for all simulation fields (prognostic, diagnostic, and scratch).

# Fields
## Prognostic
- `b`: Surface buoyancy anomaly

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
struct Fields{T, PA, PC}
    # Prognostic fields
    b::PA           # surface buoyancy anomaly
    
    # Diagnostic fields
    φ::PA           # streamfunction
    u::PA           # x-velocity
    v::PA           # y-velocity

    # Multigrid workspace
    φ_mg::PA           # Multigrid solution workspace
    b_mg::PA           # Multigrid RHS workspace
    
    # Scratch arrays (real space)
    R::PA           # residual for MA equation
    tmp::PA         # general scratch
    tmp2::PA        # additional scratch
    tmp3::PA        # additional scratch
    
    # Spectral arrays
    bhat::PC        # spectral buoyancy
    φhat::PC        # spectral streamfunction
    tmpc::PC        # spectral scratch
    tmpc2::PC       # spectral scratch
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
    b    = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    φ    = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    u    = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    v    = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    ω_z  = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    R    = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    tmp  = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    tmp2 = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    tmp3 = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    
    φ_mg = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))
    b_mg = PencilArray(domain.pr, zeros(T, local_size(domain.pr)))

    # Spectral fields
    bhat = PencilArray(domain.pc, zeros(Complex{T}, local_size(domain.pc)))
    φhat = PencilArray(domain.pc, zeros(Complex{T}, local_size(domain.pc)))
    tmpc = PencilArray(domain.pc, zeros(Complex{T}, local_size(domain.pc)))
    tmpc2 = PencilArray(domain.pc, zeros(Complex{T}, local_size(domain.pc)))
    
    return Fields{T, typeof(b), typeof(bhat)}(
        b, φ, u, v, ω_z, 
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
function zero_fields!(domain::Fields)
    fields.b    .= 0
    fields.φ    .= 0
    fields.u    .= 0
    fields.v    .= 0

    fields.φ_mg .= 0
    fields.b_mg .= 0
    fields.R    .= 0

    fields.tmp  .= 0
    fields.tmp2 .= 0
    fields.tmp3 .= 0

    fields.bhat .= 0
    fields.φhat .= 0
    
    fields.tmpc  .= 0
    fields.tmpc2 .= 0

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
function copy_field!(dest, src)
    @ensuresamegrid(dest, src)
    dest .= src
    return dest
end


"""
    field_stats(field) -> (mean, std, min, max)
Compute basic statistics for a field.
"""
"""
    field_stats(field) -> (mean, std, min, max)

Compute basic statistics for a field.
"""
function field_stats(field)
    stats = Dict{Symbol, NamedTuple}()
    for field_name in fieldnames(typeof(fields))
        field = getfield(fields, field_name)
        if isa(field, PencilArray) && eltype(field) <: Real
            field_data = field.data
            
            # Local statistics
            local_mean = mean(field_data)
            local_min   = minimum(field_data)
            local_max   = maximum(field_data)
            local_count = length(field_data)
            
            # Global statistics via MPI
            global_sum   = MPI.Allreduce(local_mean * local_count, MPI.SUM, field.pencil.comm)
            global_count = MPI.Allreduce(local_count, MPI.SUM, field.pencil.comm)
            global_mean  = global_sum / global_count
            
            global_min = MPI.Allreduce(local_min, MPI.MIN, field.pencil.comm)
            global_max = MPI.Allreduce(local_max, MPI.MAX, field.pencil.comm)
            
            # Global variance
            local_var_contrib = sum((field_data .- global_mean).^2)
            global_var = MPI.Allreduce(local_var_contrib, MPI.SUM, field.pencil.comm) / global_count
            global_std = sqrt(global_var)
            
            stats[field_name] = (
                mean = global_mean,
                std = global_std,
                min = global_min,
                max = global_max,
                count = global_count
            )
        end
    end
    
    return stats
end

"""
    Base.show(io::IO, fields::Fields)
Pretty print field information.
"""
function Base.show(io::IO, fields::Fields)
    T = eltype(fields.b)
    println(io, "Fields{$T}:")
    println(io, "  Prognostic: b (buoyancy)")
    println(io, "  Diagnostic: φ (streamfunction), u, v (velocities)")
    println(io, "  Scratch   : R, tmp, tmp2 (real), tmpc (spectral)")
    println(io, "  Local size: $(size(fields.b))")
end